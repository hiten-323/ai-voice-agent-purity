/**
 * Glitch Voice — generic LiveKit voice agent worker (engine).
 *
 * Long-running worker process. Registers with LiveKit Cloud, subscribes to
 * dispatch requests, and handles each inbound call with a profile-driven
 * agent. The engine itself is profile-agnostic: STT/TTS/LLM construction,
 * SIP audio handling, VAD, AEC, turn-detection, voicemail detection,
 * auto-hangup, and transcript persistence live here. The agent's
 * personality, prompt, tools, and welcome message are imported from the
 * per-call profile's `agent.js` module (e.g. profiles/ai-voice-agent/agent.js).
 *
 * The profile id arrives as a LiveKit participant attribute (`profile`)
 * set by trigger-livekit-call.js. Falls back to `ai-voice-agent` for legacy
 * dispatches that pre-date the multi-profile rollout.
 *
 * Architecture + idioms follow @livekit/agents v1.2.x Node.js patterns.
 * Upstream: https://github.com/livekit/agents-js — @livekit/agents@1.2.6.
 */

// Load .env HERE, explicitly, rather than relying on an accident of import
// order. server.js "worked" without ever importing dotenv because it pulls
// in @prisma/client for the scheduler, and Prisma's client runtime loads
// .env as an internal side effect of its own initialization — nothing to
// do with this app's own config. This file never touches Prisma, so it
// got none of that: LIVEKIT_API_KEY/SECRET/URL were genuinely undefined in
// this process, causing MissingCredentialsError on every startup attempt
// (found 2026-09-15 via a real test call that rang and stayed silent —
// this worker had never successfully started before that call). Must be
// the first import: everything below reads process.env at module load.
import 'dotenv/config';

import { fileURLToPath, pathToFileURL } from 'node:url';
import {
  cli,
  defineAgent,
  ServerOptions,
  voice,
} from '@livekit/agents';
import * as silero from '@livekit/agents-plugin-silero';
import * as livekit from '@livekit/agents-plugin-livekit';
import * as sarvam from '@livekit/agents-plugin-sarvam';
import * as elevenlabs from '@livekit/agents-plugin-elevenlabs';
import * as openai from '@livekit/agents-plugin-openai';
import { RoomServiceClient } from 'livekit-server-sdk';
import { getProfile } from './lib/profiles.js';
import { detectVoicemail } from './lib/voicemail-detection.js';

const WEBHOOK_BASE = process.env.COD_CONFIRM_WEBHOOK_BASE
  || 'https://your-domain.com/ai-voice-agent';
const TOOL_SECRET = process.env.LIVEKIT_TOOL_SECRET || '';
const DEFAULT_PROFILE_ID = process.env.DEFAULT_AGENT_PROFILE || 'ai-voice-agent';

// STT provider factory. Sarvam Saaras v3 is best-in-class for Hindi /
// Hinglish but is hi-IN-only — it transliterates pure-English audio into
// Devanagari nonsense (real test call to a +1 number: customer said
// "hello", STT produced "हेलो"-like garbage and the LLM had nothing
// transcribable to respond to).
//
// For en-IN we use OpenAI's Whisper-based STT (gpt-4o-transcribe — the
// successor to whisper-1 — handles Indian-accented English and code-mix
// well enough for our domain). OpenAI plugin is already a dependency for
// the LLM, so no new install.
// Languages this deployment actually supports end to end (STT + TTS + chat
// LLM). Not every Sarvam-supported language belongs here — this list is
// "we have verified all three legs", not "Sarvam's full catalogue".
const SUPPORTED_LANGS = ['en-IN', 'hi-IN', 'pa-IN'];

// Single source of truth for turning a possibly-missing/unsupported language
// attribute into one of SUPPORTED_LANGS. Used at both places a call's
// language gets decided (dispatch metadata and participant attributes) so
// the two can't drift into checking different sets — see the entrypoint.
function normalizeLang(raw) {
  return SUPPORTED_LANGS.includes(raw) ? raw : 'hi-IN';
}

function buildSTT(lang) {
  if (lang === 'en-IN') {
    console.log(`[stt] provider=openai model=gpt-4o-transcribe lang=${lang}`);
    return new openai.STT({
      model: 'gpt-4o-transcribe',
      language: 'en',
      detectLanguage: false,
    });
  }
  // Saaras v3 supports pa-IN directly (23-language set) — passing the real
  // code through rather than hardcoding hi-IN, which silently ran every
  // Punjabi call through Hindi transcription before this fix.
  console.log(`[stt] provider=sarvam model=saaras:v3 lang=${lang}`);
  return new sarvam.STT({
    model: 'saaras:v3',
    languageCode: lang,
  });
}

// TTS provider factory. Provider choice is engine-level (cost/quality/outage
// is a deployment concern, not a profile concern). Per-profile voice tuning
// (voiceId, model) will move into profile.json in a later phase; for now,
// env vars stay the lever.
function buildTTS(lang) {
  const provider = (process.env.TTS_PROVIDER || 'elevenlabs').toLowerCase();
  if (provider === 'elevenlabs') {
    const voiceId = process.env.ELEVENLABS_VOICE_ID;
    if (!voiceId) {
      throw new Error('TTS_PROVIDER=elevenlabs but ELEVENLABS_VOICE_ID is not set');
    }
    console.log(`[tts] provider=elevenlabs voice=${voiceId} lang=${lang}`);
    return new elevenlabs.TTS({
      voiceId,
      model: process.env.ELEVENLABS_MODEL || 'eleven_turbo_v2_5',
      language: lang,
      encoding: 'pcm_8000',
    });
  }
  console.log(`[tts] provider=sarvam speaker=neha lang=${lang}`);
  return new sarvam.TTS({
    model: 'bulbul:v3',
    speaker: 'neha',
    targetLanguageCode: lang,
    pace: 1.0,
    sampleRate: 8000,
  });
}

// LLM provider factory.
//
// The openai plugin is an OpenAI-PROTOCOL client, not a vendor: baseURL and
// apiKey point it at anything speaking the same API. That keeps this stack at
// LiveKit + Sarvam + Plivo plus an LLM the business already pays for, instead
// of opening an OpenAI account for the sake of one class. The Sarvam plugin
// cannot fill this slot -- at 1.2.6 it exports STT and TTS only, and the whole
// @livekit/agents-* family is pinned there.
//
// maxCompletionTokens, NOT maxTokens. The previous code passed maxTokens: 60,
// which is not in LLMOptions and was therefore silently dropped -- the model
// had no ceiling at all on a live phone call.
function buildLLM() {
  const provider = (process.env.VOICE_LLM_PROVIDER || 'sarvam').toLowerCase();
  const temperature = 0.6;
  const maxCompletionTokens = 60;   // one conversational turn, not an essay

  if (provider === 'sarvam') {
    // Sarvam serves STT, TTS and the LLM, so the whole stack is
    // LiveKit + Sarvam + Plivo with one key. Measured, not assumed:
    //
    //   sarvam-105b-conversations  2.1s  finish=stop    no reasoning leak
    //   sarvam-105b                2.3s  finish=length  reasoning, empty content
    //   nvidia nemotron-3.5        3.1s  finish=length  reasoning, empty content
    //
    // The '-conversations' suffix is load-bearing. The plain model is a
    // reasoning variant that spends the whole token budget thinking and
    // returns nothing -- on a phone call that is silence, or worse, the
    // model's monologue read aloud to a prospect.
    console.log('[llm] provider=sarvam model=sarvam-105b-conversations');
    return new openai.LLM({
      model: process.env.SARVAM_LLM_MODEL || 'sarvam-105b-conversations',
      apiKey: process.env.SARVAM_API_KEY,
      baseURL: 'https://api.sarvam.ai/v1',
      temperature,
      maxCompletionTokens,
    });
  }

  if (provider === 'cerebras') {
    console.log('[llm] provider=cerebras model=llama-3.3-70b');
    return openai.LLM.withCerebras({
      model: 'llama-3.3-70b',
      apiKey: process.env.CEREBRAS_API_KEY,
      temperature,
    });
  }

  if (provider === 'openai') {
    console.log('[llm] provider=openai model=gpt-4o-mini');
    return new openai.LLM({
      model: 'gpt-4o-mini',
      apiKey: process.env.OPENAI_API_KEY,
      temperature,
      maxCompletionTokens,
    });
  }

  // NVIDIA NIM. Kept as an option, but NOT the default for voice: the only
  // model this account can reach is nemotron-3.5-lightning, a reasoning model
  // that emits its thinking as content. Fine for the backend's batch work,
  // unusable on a live call.
  const model = process.env.NVIDIA_MODEL || 'nvidia/nemotron-3.5-lightning-30b-a3b';
  console.log(`[llm] provider=nvidia model=${model}`);
  return new openai.LLM({
    model,
    apiKey: process.env.NVIDIA_API_KEY,
    baseURL: process.env.NVIDIA_BASE_URL || 'https://integrate.api.nvidia.com/v1',
    temperature,
    maxCompletionTokens,
  });
}

// Profile-module cache: import once per (profileId, worker process).
const profileModuleCache = new Map();
async function loadProfileModule(profileId) {
  if (profileModuleCache.has(profileId)) return profileModuleCache.get(profileId);
  const profile = getProfile(profileId);
  if (!profile) {
    throw new Error(`[profile] unknown profile "${profileId}" — check profiles/<id>/profile.json`);
  }
  const agentPath = `${profile._dir}/agent.js`;
  const mod = await import(pathToFileURL(agentPath).href);
  for (const fn of ['renderContext', 'buildSystemPrompt', 'buildWelcome', 'buildTools', 'turnPersistKey']) {
    if (typeof mod[fn] !== 'function') {
      throw new Error(`[profile] ${agentPath} is missing required export "${fn}"`);
    }
  }
  if (!(mod.TERMINAL_TOOLS instanceof Set)) {
    throw new Error(`[profile] ${agentPath} must export TERMINAL_TOOLS as a Set<string>`);
  }
  console.log(`[profile] loaded "${profileId}" from ${agentPath} (tools: ${[...mod.TERMINAL_TOOLS].join(', ')})`);
  const cached = { profile, ...mod };
  profileModuleCache.set(profileId, cached);
  return cached;
}

// Voicemail detection — engine-level, applies to every profile. Real call
// #2953 (Storico) burned 2 minutes pitching to a voicemail recording before
// we added this. Patterns are broad: partial match anywhere in any of the
// first 4 user transcripts triggers an immediate hangup. Detection logic
// lives in lib/voicemail-detection.js so it can be unit-tested without
// pulling in the LiveKit CLI (importing this file runs cli.runApp()).

export default defineAgent({
  prewarm: async (proc) => {
    proc.userData.vad = await silero.VAD.load({
      sampleRate: 8000,
      minSilenceDuration: 400,
    });
    const warmupHosts = [
      'https://api.sarvam.ai/',
      'https://api.elevenlabs.io/v1/models',
      'https://api.openai.com/v1/models',
    ];
    await Promise.all(warmupHosts.map(async (url) => {
      try {
        const controller = new AbortController();
        const t = setTimeout(() => controller.abort(), 4000);
        await fetch(url, { method: 'HEAD', signal: controller.signal }).catch(() => {});
        clearTimeout(t);
      } catch { /* best-effort */ }
    }));
    console.log('[prewarm] TLS/DNS warmed for Sarvam + ElevenLabs + OpenAI');
  },

  entry: async (ctx) => {
    await ctx.connect();

    // Dispatch metadata is set by trigger-livekit-call.js (createDispatch
    // metadata field) and includes the call's language hint. Reading it
    // BEFORE the AgentSession is constructed lets us pick the right STT
    // provider (Sarvam for Hindi, OpenAI for English) without a mid-call
    // STT swap. Falls back to hi-IN if metadata is missing or unparseable
    // (the production-default and 98% of ai-voice-agent traffic).
    let dispatchMeta = {};
    try {
      const raw = ctx.job?.metadata || ctx.room?.metadata || '';
      if (raw && typeof raw === 'string' && raw.trim().startsWith('{')) {
        dispatchMeta = JSON.parse(raw);
      }
    } catch (err) {
      console.warn('[dispatch-meta] could not parse:', err.message);
    }
    const initialLang = normalizeLang(dispatchMeta.lang);
    console.log(`[entry] dispatch lang=${initialLang} profile=${dispatchMeta.profile || '?'}`);

    const ctxMut = {
      v: {},
      lang: initialLang,
      turnIndex: 0,
      sipCallId: null,
      profileId: DEFAULT_PROFILE_ID,
      profileMod: null,
    };
    const roomName = ctx.room?.name || '';

    async function postTurn({ role, text, tool_name, tool_args, tool_result, stt_confidence }) {
      if (!ctxMut.profileMod || !roomName) return;
      const key = ctxMut.profileMod.turnPersistKey(ctxMut.v);
      if (!key) return; // sandbox / demo call without entity context — skip persistence
      const payload = {
        ...key,
        room_name:  roomName,
        sip_call_id: ctxMut.sipCallId,
        turn_index: ctxMut.turnIndex++,
        role,
        text:       text || '',
        lang:       ctxMut.lang,
        tool_name, tool_args, tool_result, stt_confidence,
        started_at: new Date().toISOString(),
      };
      try {
        const res = await fetch(`${WEBHOOK_BASE}/webhook/livekit/turn`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            ...(TOOL_SECRET ? { 'X-COD-Tool-Secret': TOOL_SECRET } : {}),
          },
          body: JSON.stringify(payload),
        });
        if (!res.ok) {
          console.warn(`[turn-persist] HTTP ${res.status} for ${role} turn #${payload.turn_index}`);
        }
      } catch (err) {
        console.warn(`[turn-persist] fire-and-forget error on ${role} turn #${payload.turn_index}:`, err.message);
      }
    }

    const session = new voice.AgentSession({
      vad: ctx.proc.userData.vad,
      stt: buildSTT(initialLang),
      llm: buildLLM(),
      tts: buildTTS(initialLang),
      // Re-enabled 2026-09-15: removed earlier the same day after "File not
      // found in cache" for revision v0.4.1-intl crashed the worker on every
      // startup. Root cause wasn't a permanent upstream bug — the crash-loop
      // itself was racing download-files' cache-ref write (hf_utils.js writes
      // storageFolder/refs/<tag> -> <commit-hash> after fetching), so restarts
      // kept catching that mapping half-written. Confirmed fixed: refs/v0.4.1-intl
      // and refs/v1.2.2-en both now resolve to snapshots that exist on disk
      // with the model file present (396MB / 65MB respectively). Better
      // end-of-turn detection than VAD alone matters most exactly where this
      // was reported missing: distinguishing a mid-sentence pause from an
      // actual turn end in Hindi/Hinglish conversational cadence.
      turnDetection: new livekit.turnDetector.MultilingualModel(),
      preemptiveGeneration: true,
      aecWarmupDuration: 500,
      minInterruptionWords: 3,
      minInterruptionDuration: 600,
    });

    let terminalToolFired = false;
    let hangupTimer = null;
    let voicemailDetected = false;
    let userTurnCount = 0;
    // Set the moment ANY outcome — record_call_outcome or the engine's own
    // VOICEMAIL report — is on its way to Python. Checked on session Close
    // so a dropped call, a crash, or the caller just hanging up mid-sentence
    // still produces a FAILED record instead of vanishing with no trace,
    // same reasoning as the voicemail fix just above it.
    let outcomeReported = false;
    const autoHangupMs = parseInt(process.env.AUTO_HANGUP_MS || '10000', 10);

    function hangupNow(reason) {
      const rn = ctx.room?.name;
      if (!rn) return;
      const lkUrl = process.env.LIVEKIT_URL;
      const lkKey = process.env.LIVEKIT_API_KEY;
      const lkSecret = process.env.LIVEKIT_API_SECRET;
      if (!lkUrl || !lkKey || !lkSecret) {
        console.warn(`[hangup] LIVEKIT_* env missing — cannot terminate room (${reason})`);
        return;
      }
      console.log(`[hangup] deleteRoom ${rn} — ${reason}`);
      const rs = new RoomServiceClient(lkUrl, lkKey, lkSecret);
      rs.deleteRoom(rn).catch(err =>
        console.log(`[hangup] deleteRoom (likely already closed): ${err.message}`)
      );
    }

    session.on(voice.AgentSessionEventTypes.UserInputTranscribed, (ev) => {
      if (!ev.isFinal) return;
      const transcript = ev.transcript || '';
      console.log(`[user] ${transcript}`);
      userTurnCount++;

      if (!voicemailDetected && userTurnCount <= 4) {
        const matched = detectVoicemail(transcript);
        if (matched) {
          voicemailDetected = true;
          console.log(`[voicemail] detected on user turn #${userTurnCount}: "${transcript}" (${matched.kind}: ${matched.matched})`);
          postTurn({
            role: 'tool',
            text: 'voicemail_detected',
            tool_name: 'voicemail_detected',
            tool_args: { transcript },
            tool_result: 'hangup',
          });
          // Before this, a voicemail hit hung up with ZERO record on the
          // Python side: ai_call_count never incremented, no CallHistory
          // row, nothing in founder_call_pipeline at all for a call that
          // did in fact happen. reportOutcome() is the same function the
          // LLM's record_call_outcome tool calls — this is the engine
          // calling it directly for an outcome the model never gets a
          // meaningful turn to decide.
          outcomeReported = true;
          if (ctxMut.profileMod && typeof ctxMut.profileMod.reportOutcome === 'function') {
            ctxMut.profileMod.reportOutcome(ctxMut.v, 'VOICEMAIL', `voicemail greeting matched: "${transcript.slice(0, 200)}"`)
              .catch(err => console.warn('[voicemail] reportOutcome failed:', err.message));
          }
          hangupNow('voicemail detected');
          return;
        }
      }

      postTurn({
        role: 'user',
        text: transcript,
        stt_confidence: typeof ev.confidence === 'number' ? ev.confidence : undefined,
      });
    });

    // Real per-stage latency, from the SDK's own instrumentation rather than
    // a guess. Before this listener existed, nothing in this codebase ever
    // read AgentSessionEventTypes.MetricsCollected -- "the call feels slow"
    // had no numbers behind it because nothing was logging the numbers the
    // framework was already computing on every turn. eou = time from the
    // caller going silent to the turn being considered over; llm.ttftMs =
    // time to the model's first token (not durationMs, which includes the
    // whole completion and streams to TTS incrementally); tts.ttfbMs = time
    // to the first audio byte the caller actually hears.
    session.on(voice.AgentSessionEventTypes.MetricsCollected, (ev) => {
      const m = ev.metrics;
      if (m.type === 'eou_metrics') {
        console.log(`[latency] eou endOfUtteranceDelayMs=${m.endOfUtteranceDelayMs} transcriptionDelayMs=${m.transcriptionDelayMs} onUserTurnCompletedDelayMs=${m.onUserTurnCompletedDelayMs}`);
      } else if (m.type === 'llm_metrics') {
        console.log(`[latency] llm ttftMs=${m.ttftMs} durationMs=${m.durationMs} completionTokens=${m.completionTokens} tokensPerSecond=${m.tokensPerSecond.toFixed(1)}`);
      } else if (m.type === 'tts_metrics') {
        console.log(`[latency] tts ttfbMs=${m.ttfbMs} durationMs=${m.durationMs} audioDurationMs=${m.audioDurationMs} streamed=${m.streamed}`);
      } else if (m.type === 'stt_metrics') {
        console.log(`[latency] stt durationMs=${m.durationMs} audioDurationMs=${m.audioDurationMs} streamed=${m.streamed}`);
      }
    });

    session.on(voice.AgentSessionEventTypes.ConversationItemAdded, (ev) => {
      if (ev.item?.role !== 'assistant') return;
      const text = ev.item.textContent ?? '';
      console.log(`[assistant] ${text.slice(0, 200)}`);
      postTurn({ role: 'assistant', text });

      if (terminalToolFired && !hangupTimer) {
        const rn = ctx.room?.name;
        hangupTimer = setTimeout(async () => {
          try {
            if (!rn) return;
            const lkUrl = process.env.LIVEKIT_URL;
            const lkKey = process.env.LIVEKIT_API_KEY;
            const lkSecret = process.env.LIVEKIT_API_SECRET;
            if (!lkUrl || !lkKey || !lkSecret) {
              console.warn('[auto-hangup] LIVEKIT_* env missing — cannot terminate room');
              return;
            }
            console.log(`[auto-hangup] deleting room ${rn} after farewell — VoIP-minutes guard (${autoHangupMs}ms)`);
            const rs = new RoomServiceClient(lkUrl, lkKey, lkSecret);
            await rs.deleteRoom(rn);
          } catch (err) {
            console.log(`[auto-hangup] deleteRoom (likely already closed): ${err.message}`);
          }
        }, autoHangupMs);
      }
    });

    session.on(voice.AgentSessionEventTypes.FunctionToolsExecuted, (ev) => {
      const calls = ev.functionCalls || [];
      console.log('[tool]', calls.map(c => c.name).join(',') || '?');
      for (const c of calls) {
        postTurn({
          role:        'tool',
          text:        c.name || '',
          tool_name:   c.name,
          tool_args:   c.arguments ?? c.args ?? undefined,
          tool_result: typeof c.result === 'string' ? c.result : (c.result ? JSON.stringify(c.result) : undefined),
        });
        if (ctxMut.profileMod && ctxMut.profileMod.TERMINAL_TOOLS.has(c.name)) {
          terminalToolFired = true;
          // Bug found via the 2026-09-15 controlled test call: this used to
          // set outcomeReported = true just because the TOOL NAME matched,
          // regardless of whether execute() actually reached Python. That
          // call's webhook POST got HTTP 404 (purity-api was running from
          // before the route existed) — the tool "fired" by every log here,
          // but the Close handler's FAILED-fallback below never ran, because
          // this flag had already (wrongly) claimed something succeeded.
          // Checking the tool's own returned {ok} is the only way to know.
          const succeeded = c.result && typeof c.result === 'object' && c.result.ok === true;
          if (succeeded) {
            outcomeReported = true;
          } else {
            console.warn(`[auto-hangup] terminal tool ${c.name} fired but did not report a successful outcome (result: ${JSON.stringify(c.result)}) — leaving outcomeReported=false so Close still reports FAILED`);
          }
          console.log(`[auto-hangup] armed after terminal tool: ${c.name}`);
        }
      }
    });

    session.on(voice.AgentSessionEventTypes.Close, () => {
      if (hangupTimer) {
        clearTimeout(hangupTimer);
        hangupTimer = null;
      }
      console.log(`[livekit-agent] session closed after ${ctxMut.turnIndex} turns`);

      // Every completed conversation should produce an auditable outcome —
      // a dropped call, a crash, or the caller hanging up mid-question
      // shouldn't leave Python with zero trace that a call ever happened.
      // Only fires when nothing else already reported (record_call_outcome
      // or the voicemail path above), and only when there was actually a
      // lead to report against — a sandbox call with no lead_id has nothing
      // for reportOutcome to attach to anyway.
      if (!outcomeReported && ctxMut.v?.lead_id && ctxMut.profileMod
          && typeof ctxMut.profileMod.reportOutcome === 'function') {
        outcomeReported = true;
        ctxMut.profileMod.reportOutcome(
          ctxMut.v, 'FAILED',
          `session closed after ${ctxMut.turnIndex} turn(s) with no outcome recorded`,
        ).catch(err => console.warn('[close] reportOutcome(FAILED) failed:', err.message));
      }
    });

    // Serial start (the previous parallel-start attempt killed AgentActivity
    // before updateAgent could land — see git history for the bug). With TLS
    // prewarmed, serial cold-start is ~3-5s and the welcome reliably plays.
    const participant = await ctx.waitForParticipant();
    const attrs = participant.attributes || {};

    ctxMut.profileId = attrs.profile || DEFAULT_PROFILE_ID;
    ctxMut.lang = normalizeLang(attrs.language);
    ctxMut.sipCallId = attrs.sip_call_id || null;

    let profileMod;
    try {
      profileMod = await loadProfileModule(ctxMut.profileId);
    } catch (err) {
      console.error(`[profile] failed to load "${ctxMut.profileId}":`, err.message);
      hangupNow(`profile load failed: ${err.message}`);
      return;
    }
    ctxMut.profileMod = profileMod;

    ctxMut.v = profileMod.renderContext(attrs, ctxMut.lang, process.env);
    const v = ctxMut.v;
    const lang = ctxMut.lang;
    console.log(`[livekit-agent] profile=${ctxMut.profileId} call for ${v.customer_name || '(no name)'} / ${v.order_number || attrs.entity_ref || '-'} lang=${lang}`);

    const realAgent = new voice.Agent({
      instructions: profileMod.buildSystemPrompt(v, lang),
      tools:        profileMod.buildTools(v, { WEBHOOK_BASE, TOOL_SECRET }),
    });

    const coldStartMs = Date.now();
    await session.start({ agent: realAgent, room: ctx.room });
    console.log(`[cold-start] serial-start: ${Date.now() - coldStartMs}ms (TLS prewarmed)`);

    session.say(profileMod.buildWelcome(v, lang, process.env), { allowInterruptions: false });
  },
});

cli.runApp(
  new ServerOptions({
    agent: fileURLToPath(import.meta.url),
    agentName: process.env.LIVEKIT_AGENT_NAME || 'ai-voice-agent-priya',
    host: process.env.LIVEKIT_AGENT_HTTP_HOST || '127.0.0.1',
  }),
);
