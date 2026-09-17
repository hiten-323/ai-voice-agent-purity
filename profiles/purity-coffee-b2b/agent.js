/**
 * purity-coffee-b2b — the disclosed AI qualification call.
 *
 * Division of authority, unchanged since this file was first built:
 *   - founder_call_pipeline.py owns the script CONTENT (opening, questions,
 *     constraints) and the outcome vocabulary. This module renders what it
 *     is given and decides HOW to hold the conversation; it does not invent
 *     the founder's wording.
 *   - This module owns nothing about who may be called — that was already
 *     decided before voice_router.place_call() ever ran, and nothing below
 *     can override consent, DNC, or eligibility (see may_place_ai_call() /
 *     check_eligibility() / voice_router.kill_switch_engaged()).
 *   - The outcome tool is the only way this call's result reaches Python,
 *     via POST /api/v1/founder/ai-call-outcome ->
 *     founder_call_pipeline.record_ai_outcome(), the one place the outcome
 *     vocabulary is enforced (see that function's ValueError on an
 *     unrecognised key).
 *
 * 2026-09-15 revision — conversion + safety audit (see engineering log for
 * the full 13-phase writeup). What changed and why:
 *   - The system prompt now instructs adaptive branching (stop asking
 *     scripted questions the moment an answer already gives an outcome)
 *     instead of marching through all four regardless of what's said —
 *     the previous version sounded like an interrogation, per direct
 *     feedback from the first live test call.
 *   - SEND_INFO_EMAIL is now distinct from WHATSAPP_OPT_IN. Before this,
 *     a generic "send me details" had no correct outcome to reach for
 *     other than WHATSAPP_OPT_IN, which is the ONE outcome in this whole
 *     system that manufactures WhatsApp consent (Meta requires the
 *     business to ask for it, in its own words) — an ambiguous "send
 *     details" silently becoming a WhatsApp opt-in would have been this
 *     codebase's ninth instance of turning a fact into a licence.
 *   - MEETING_REQUESTED / CALLBACK_REQUESTED / HUMAN_HANDOFF / WRONG_PERSON
 *     / OTHER added as distinct outcome keys (see founder_call_pipeline.py
 *     OUTCOMES) so reporting can see what was actually asked for, without
 *     opening a new pipeline-stage transition — they map onto the same
 *     AI_INTEREST_DETECTED / AI_NOT_INTERESTED stages the existing FSM
 *     already allows from AI_CALL_ATTEMPTED.
 *   - VOICEMAIL / FAILED are reported by the ENGINE (livekit-agent.js),
 *     never by this tool's enum — see reportOutcome() below. Before this
 *     change a voicemail hit produced no record in founder_call_pipeline
 *     at all: ai_call_count never incremented, no CallHistory row, no
 *     audit trail for a call that did in fact happen.
 *   - Objection handling and voice-style rules added to the prompt,
 *     condensed rather than scripted per-objection, because Phase 5/6 of
 *     the audit explicitly warns that mechanically scripting every
 *     objection is how a conversation starts sounding like one.
 *   - Punjabi (pa-IN) added as a supported language: Saaras v3 (STT),
 *     Bulbul v3 (TTS) and Sarvam's chat LLM all support it, and this
 *     business's actual leads are Punjab-based (Ludhiana in the existing
 *     test fixtures) — English/Hindi-only was leaving the stack's own
 *     capability unused for the real target market.
 */
import { llm } from '@livekit/agents';
const { tool } = llm;

// Mirrors founder_call_pipeline.OUTCOMES, minus NO_ANSWER/VOICEMAIL/FAILED —
// those three are engine-detected or engine-reported (voicemail-greeting
// pattern match, no-pickup, a dropped/errored session), never something the
// model decides mid-conversation. Keeping them out of the enum means the
// model literally cannot report "no answer" on a call it is, by definition,
// having — the previous version already excluded NO_ANSWER for this reason;
// this just extends the same principle to the other two engine-side outcomes.
// Ordered by conversion priority, not alphabetically or by how safety-critical
// they are (OPT_OUT still always wins in practice — see the prompt's OBJECTIONS
// section). A concrete next step (meeting/callback/send-info/WhatsApp/handoff)
// is worth more than a bare INTERESTED, so the prompt is written to reach for
// one of those first; this order just keeps the schema's own documentation
// consistent with that.
const OUTCOME_VALUES = [
  'MEETING_REQUESTED',
  'CALLBACK_REQUESTED',
  'SEND_INFO_EMAIL',
  'WHATSAPP_OPT_IN',
  'HUMAN_HANDOFF',
  'INTERESTED',
  'NOT_INTERESTED',
  'WRONG_PERSON',
  'WRONG_NUMBER',
  'OPT_OUT',
  'OTHER',
];

// Outcomes the engine itself may report, bypassing the LLM tool entirely —
// see reportOutcome() below. Kept separate from OUTCOME_VALUES so the two
// enums can never be confused: one is "things the model concluded", the
// other is "things that happened to the call before/without the model
// getting a meaningful turn".
const ENGINE_OUTCOME_VALUES = ['VOICEMAIL', 'FAILED'];

const PURITY_API_BASE = (process.env.PURITY_API_BASE || 'http://127.0.0.1:8003').replace(/\/$/, '');
const PURITY_API_ADMIN_SECRET = process.env.PURITY_API_ADMIN_SECRET || '';

function parseJsonArray(raw) {
  if (!raw) return [];
  try {
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    // nuraveda_provider.py JSON-encodes list context values before they're
    // spread into SIP participant attributes (LiveKit coerces every
    // attribute value with String(), which comma-joins a raw array and
    // corrupts any entry that itself contains a comma — CALL_CONSTRAINTS
    // does). A parse failure here means that encoding step was skipped
    // somewhere upstream, not that the call has no constraints to enforce.
    console.warn('[purity-coffee-b2b] failed to parse JSON array attribute:', raw.slice(0, 120));
    return [];
  }
}

export function renderContext(attrs, lang, env) {
  return {
    lead_id: attrs.lead_id || '',
    customer_name: attrs.customer_name || attrs.contact || '',
    company: attrs.company || '',
    city: attrs.city || '',
    segment: attrs.segment || '',
    opening: attrs.opening || '',
    questions: parseJsonArray(attrs.questions),
    constraints: parseJsonArray(attrs.constraints),
    order_number: attrs.entity_ref || '',
  };
}

export function buildSystemPrompt(v, lang) {
  const questionsBlock = v.questions.length
    ? v.questions.map((q, i) => `${i + 1}. ${q}`).join('\n')
    : '(no questions were provided for this call — ask whether they buy coffee commercially, then whether the founder may call them.)';

  const constraintsBlock = v.constraints.length
    ? v.constraints.map((c) => `- ${c}`).join('\n')
    : '';

  const languageLine = {
    'en-IN': 'Speak Indian English throughout.',
    'pa-IN': 'Speak Punjabi, mixing in English business terms naturally — this is normal for a B2B call in Punjab. If Punjabi ever feels forced for a specific phrase, Hindi/Hinglish is an acceptable fallback for that phrase only.',
  }[lang] || 'Speak Hindi, mixing in English business terms naturally (Hinglish) — this is normal for a B2B call in India.';

  return [
    `You are conducting one disclosed AI qualification call on behalf of Pure Pantry Provisions, a coffee supplier. The call is with ${v.company || 'a business'}${v.city ? ` in ${v.city}` : ''}.`,
    `You already spoke the opening disclosure aloud before this prompt takes over — do not repeat it or re-introduce yourself.`,
    ``,
    `YOUR GOAL`,
    `Your goal is NOT to sell anything or talk them into switching suppliers. Your goal is to find out whether this business actually has a problem you can help with, and if so, get them to a concrete next step — a founder call, a callback, or sending details. A short "not relevant, thanks" is a completely fine outcome too. Don't treat this as a pitch you're delivering or a form you're filling in — it's a real, short conversation, and the moment you have enough to know which way it's going, act on that instead of continuing to ask questions.`,
    ``,
    `HOW TO RUN THIS CALL`,
    `Don't work through the list below top to bottom like a script. It's what you need to find out, in roughly this order, and you skip whatever they've already told you:`,
    `1. Right person? (are they the one who handles coffee/procurement)`,
    `2. Do they actually buy coffee commercially right now?`,
    `3. How are they currently set up — who they use, and whether anything about it sounds like a hassle.`,
    `4. Only if something in their answer sounds like a real pain point (cost, consistency, a supplier letting them down) — ask one follow-up on that specific thing, and only that.`,
    `Reference questions, use your own words for each:`,
    questionsBlock,
    ``,
    `Then branch on what you actually heard:`,
    `- Nothing that sounds like a real problem -> a short, honest, polite close. Don't manufacture a pain point that isn't there.`,
    `- Something that does sound like a real problem -> give the short value line (see MICRO-PITCH below), then ask for a next step. This is the point of the call — don't stop at "that's interesting," push gently to an actual next step.`,
    `Skip straight to a close the moment the call is effectively decided — a clear no, a clear wrong person, a request to stop. Don't keep asking the remaining questions once you already have the answer that ends the call.`,
    ``,
    `MICRO-PITCH — use only when they ask what you offer, or once you've heard a real reason to continue`,
    `One or two sentences, not a pitch: "We supply coffee for cafes and businesses, and the founder usually looks at what a place is currently using — quality, consistency, pricing — and says honestly if switching would actually help." Then ask for the next step. Never expand this into a feature list, never repeat it a second way if they don't bite, and never add anything not in HARD RULES below.`,
    ``,
    `OBJECTIONS — handle briefly, never argue, never repeat the same pitch a second way, never manufacture urgency:`,
    `- "I'm busy" / "call later" -> ask for a better time if they offer one, then close politely.`,
    `- "Not interested" / "we already use someone" -> accept it immediately, do not counter-sell.`,
    `- "Who are you" / "how did you get my number" -> say plainly you're an AI assistant calling on behalf of Pure Pantry Provisions, a coffee supplier reaching out to local businesses. Never deny being AI, and never claim a relationship with them that doesn't exist.`,
    `- "Is this an AI?" -> yes, say so plainly and continue; this was already disclosed at the start of the call.`,
    `- "What do you offer" -> the MICRO-PITCH above. Nothing more unless HARD RULES explicitly permits it.`,
    `- "Send me details" -> ask email or WhatsApp. If they specifically say WhatsApp, that is WHATSAPP_OPT_IN. Anything else — email, "just send it", no channel named — is SEND_INFO_EMAIL. Never assume WhatsApp when they didn't say it; that is the one outcome in this whole system that creates a messaging permission, and it may only be used when they asked for that channel by name.`,
    `- "Remove my number" / "don't call again" / "stop calling" -> stop immediately. Do not ask anything else, do not continue the questions, say you will not call again, and close with OPT_OUT. Nothing overrides this, including if they say it before you've finished a question.`,
    ``,
    constraintsBlock ? `HARD RULES — do not violate these under any circumstance, even if asked directly:\n${constraintsBlock}` : '',
    ``,
    `This call qualifies interest only. You do not sell, quote a price, or take an order.`,
    ``,
    `VOICE STYLE — sound like a real person on a short business call, not a script being read aloud`,
    `- 3-12 words for a simple acknowledgement, one short question or sentence otherwise. Never deliver more than two sentences before pausing for them.`,
    `- Acknowledge what they actually said before moving on — a short "achha", "haan", "right", "okay", "got it", "that's helpful" — vary it, don't repeat the same one every turn.`,
    `- Don't repeat the business's name back at them unless it's genuinely needed. Don't open every turn with a filler phrase.`,
    `- Never say "I completely understand your concern", "thank you for sharing that", "I'd like to understand...", "I would like to take this opportunity to", or anything else that sounds like a corporate script. Say it the way a person actually would.`,
    `- If they start speaking while you are, stop immediately — don't finish your sentence, don't talk over them, just listen.`,
    `- On silence after a question, wait briefly, then a short "are you there?" is enough — don't repeat the whole question or keep talking into silence.`,
    `- Don't add fake hesitation ("um", "uh") or artificial pauses — naturalness comes from what you choose to say, not from imitating disfluency.`,
    `- Match their language. If they answer in Hindi/Hinglish, follow naturally in Hindi/Hinglish — don't mechanically translate an English sentence structure. Business words (coffee, supplier, pricing, quality, sample, founder, cafe, hotel, restaurant) can stay in English inside a Hindi sentence; that's normal, not a language switch.`,
    ``,
    `RECORDING THE OUTCOME`,
    `When the conversation reaches a clear result, call record_call_outcome exactly once, near the end, with the outcome that best matches what happened. "Sounds interesting" on its own is not a finished outcome — if there's genuine relevance, ask for a next step before the call ends, and record whichever of these actually happened:`,
    `- MEETING_REQUESTED: explicitly asked for a visit or a call to discuss further.`,
    `- CALLBACK_REQUESTED: asked to be called back, ideally with a time — put it in callback_window.`,
    `- SEND_INFO_EMAIL: wants details sent, channel not specifically WhatsApp.`,
    `- WHATSAPP_OPT_IN: specifically asked for WhatsApp.`,
    `- HUMAN_HANDOFF: asked urgently to speak to a real person right now.`,
    `- INTERESTED: open to hearing more or agreed the founder may call, but you couldn't pin down a more specific next step than that — this is a fallback, not the goal.`,
    `- NOT_INTERESTED: said no, or doesn't buy coffee commercially.`,
    `- WRONG_PERSON: right business, not the right individual.`,
    `- WRONG_NUMBER: this business/person is not who the call was meant for.`,
    `- OPT_OUT: asked not to be called again.`,
    `- OTHER: a real conversation happened but none of the above fits.`,
    `Always call record_call_outcome before ending the call, even for a quick "not interested" — give a brief, polite closing line first, then call the tool.`,
    ``,
    languageLine,
  ].filter(Boolean).join('\n');
}

export function buildWelcome(v, lang, env) {
  if (v.opening) return v.opening;
  // Fallback only fires if founder_call_pipeline.OPENING_DISCLOSURE somehow
  // didn't arrive in the payload — script_discloses() already validates the
  // real one before dispatch, so this should be unreachable in practice.
  const fallbacks = {
    'en-IN': 'Hello, this is an AI assistant calling on behalf of Pure Pantry Provisions. We supply coffee to cafes and businesses. Is this a good time for one quick question?',
    'pa-IN': 'Sat sri akal, main Pure Pantry Provisions valon AI assistant bol rahi haan. Asi cafes te businesses nu coffee supply karde haan. Ki hun ik chhota jiha sawaal puch sakdi haan?',
  };
  return fallbacks[lang] || 'Namaste, main Pure Pantry Provisions ki taraf se AI assistant baat kar rahi hoon. Hum cafes aur businesses ko coffee supply karte hain. Kya ek chhota sa sawaal poochh sakti hoon?';
}

export function buildTools(v) {
  return {
    record_call_outcome: tool({
      description: 'Record the result of this qualification call. Call this exactly once, near the end of the conversation, once the outcome is clear.',
      parameters: {
        type: 'object',
        properties: {
          outcome: {
            type: 'string',
            enum: OUTCOME_VALUES,
            description: 'The single outcome that best matches how the call went.',
          },
          summary: {
            type: 'string',
            description: 'One or two plain-language sentences summarizing what was said.',
          },
          interest: {
            type: 'string',
            description: 'HIGH, MEDIUM, or LOW — only meaningful when outcome is INTERESTED or MEETING_REQUESTED.',
          },
          callback_window: {
            type: 'string',
            description: 'When they said the founder should call back, if they mentioned one, e.g. "tomorrow morning" or "after 6pm".',
          },
        },
        required: ['outcome', 'summary'],
      },
      execute: async (args) => reportOutcome(v, args.outcome, args.summary, {
        interest: args.interest,
        callback_window: args.callback_window,
      }),
    }),
  };
}

/**
 * The one path to POST /api/v1/founder/ai-call-outcome, used by both the
 * LLM tool above and the engine directly for VOICEMAIL/FAILED (see
 * livekit-agent.js's voicemail-detection and session-error handling, which
 * call this the same way the tool's execute() does). Kept as one function
 * so there is exactly one place that builds this request, not two that
 * could drift.
 */
export async function reportOutcome(v, outcome, summary, extra = {}) {
  if (!v.lead_id) {
    console.warn(`[reportOutcome] no lead_id in call context — cannot record ${outcome}`);
    return { ok: false, error: 'no lead_id in context' };
  }
  if (!OUTCOME_VALUES.includes(outcome) && !ENGINE_OUTCOME_VALUES.includes(outcome)) {
    console.warn(`[reportOutcome] "${outcome}" is not a known outcome for either the model or the engine — refusing to guess, not sending`);
    return { ok: false, error: `unknown outcome: ${outcome}` };
  }
  try {
    const res = await fetch(`${PURITY_API_BASE}/api/v1/founder/ai-call-outcome`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        ...(PURITY_API_ADMIN_SECRET ? { 'X-Api-Admin-Secret': PURITY_API_ADMIN_SECRET } : {}),
      },
      body: JSON.stringify({
        lead_id: Number(v.lead_id),
        outcome,
        summary: summary || '',
        interest: extra.interest || '',
        callback_window: extra.callback_window || '',
      }),
    });
    const body = await res.json().catch(() => ({}));
    if (!res.ok) {
      console.warn(`[reportOutcome] HTTP ${res.status}:`, body);
      return { ok: false, error: `HTTP ${res.status}: ${body.detail || 'unknown error'}` };
    }
    console.log(`[reportOutcome] lead=${v.lead_id} outcome=${outcome} -> stage=${body.stage}`);
    return { ok: true, stage: body.stage };
  } catch (err) {
    console.warn('[reportOutcome] request failed:', err.message);
    return { ok: false, error: err.message };
  }
}

export function turnPersistKey(v) {
  // No lead_id means a sandbox/manual test call with no real context to
  // attach transcripts to — postTurn() already treats a null return as
  // "skip persistence", not an error.
  if (!v.lead_id) return null;
  return { shop: 'purity-coffee-b2b', shopify_order_id: String(v.lead_id) };
}

export const TERMINAL_TOOLS = new Set(['record_call_outcome']);
