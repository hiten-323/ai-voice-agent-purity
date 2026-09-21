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
 *
 * 2026-09-18 revision — AI Calling V3:
 *   - Permission-first opening (founder_call_pipeline.opening_for), then a
 *     decision tree instead of a question list: at most two qualifying
 *     questions, trade leads get the distributor conversation, everyone else
 *     the supply one (isTradeLead).
 *   - Interest and consent are separate steps. The agent offers WhatsApp OR
 *     email; WhatsApp consent binds to the number called unless they read out
 *     another, which travels as whatsapp_number and is validated server-side.
 *   - Pre-call context from the record (provenance, business type, previous
 *     contact), so "how did you get my number?" gets a true answer or an
 *     honest "I don't know", never an invented one.
 *   - Commercial guardrails: no figure of any kind; margins/territory offer a
 *     person (HUMAN_HANDOFF). The rule text itself stays Python-owned.
 *   - Structured outcome fields (preferred_channel, handles_instant_coffee,
 *     decision_maker, objection) in closed vocabularies, for learning which
 *     channel and which objection actually recur.
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
// they are (OPT_OUT still always wins in practice — see the prompt's WHEN THEY
// STEER section). A concrete next step (meeting/callback/send-info/WhatsApp/handoff)
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
    handoff_topics: parseJsonArray(attrs.handoff_topics),
    // founder_call_pipeline.call_context(): taken from the record, each ""
    // when the record does not hold it — never filled in here.
    provenance: attrs.provenance || '',
    business_type: attrs.business_type || '',
    previous_contact: attrs.previous_contact || '',
    order_number: attrs.entity_ref || '',
  };
}

// The closed vocabularies founder_call_pipeline.CALL_DETAIL_VALUES accepts.
// Anything else is dropped server-side; mirrored here so the model is offered
// only values that will be kept.
const CALL_DETAIL_ENUMS = {
  preferred_channel: ['WHATSAPP', 'EMAIL', 'CALL', 'NONE'],
  handles_instant_coffee: ['YES', 'NO', 'UNKNOWN'],
  decision_maker: ['YES', 'NO', 'UNKNOWN'],
  objection: ['NONE', 'EXISTING_SUPPLIER', 'PRICE', 'NO_NEED', 'TIMING', 'OTHER'],
};

// Businesses that RESELL coffee get the distributor conversation ("do you carry
// any instant coffee brands?"); everyone else is a business that USES it and
// gets the supply conversation ("do you use instant coffee at the moment?").
// Asking a cafe whether it distributes instant coffee brands, or a distributor
// whether it serves coffee to guests, is the fastest way to sound like a script
// that was never pointed at them.
const TRADE_SEGMENTS = new Set([
  'distributor', 'wholesaler', 'wholesaler_agglo', 'modern_trade', 'stockist',
  'retail', 'retail_chain', 'retail_kirana', 'kirana_store', 'grocery', 'supermarket',
]);

export function isTradeLead(v) {
  return TRADE_SEGMENTS.has(String(v.segment || '').trim().toLowerCase());
}

export function buildSystemPrompt(v, lang) {
  const questionsBlock = v.questions.length
    ? v.questions.map((q, i) => `${i + 1}. ${q}`).join('\n')
    : '(no questions were provided for this call — ask whether they buy coffee commercially, then whether the founder may call them.)';

  const whereEn = v.city ? ` around ${v.city}` : '';
  const whereHi = v.city ? ` ${v.city} ke aas-paas` : '';
  const trade = isTradeLead(v);
  const hindi = (lang === 'hi-IN' || lang === 'pa-IN');
  const contextLine = hindi
    ? (trade
      ? `"Dhanyavaad. Hum distributors se instant coffee pe baat kar rahe hain${whereHi}. Kya explore karna chahenge?"`
      : `"Dhanyavaad. Hum cafes aur businesses ko instant coffee supply karte hain${whereHi}. Kya yeh useful ho sakta hai?"`)
    : (trade
      ? `"Thank you. We're currently connecting with distributors in the instant coffee category${whereEn}, and I wanted to see if this is something you'd be open to exploring."`
      : `"Thank you. We supply instant coffee to cafes and businesses${whereEn}, and I wanted to see if it could be useful for you."`);
  const qualifyBlock = hindi
    ? (trade
      ? [
          `- Ask: "Kya aap abhi koi instant coffee brand distribute karte hain?"`,
          `- If yes: "Theek hai. Terms theek hon to kya ek aur brand add kar sakte hain?"`,
          `- If no: "Samajh gayi. Instant coffee add karne ka soch rahe hain?" If also no, close politely.`,
        ]
      : [
          `- Ask: "Kya aap abhi instant coffee use karte hain?"`,
          `- If yes: "Theek hai. Usually kahan se lete hain?" Then at most one short follow-up if they name a real problem.`,
          `- If no: "Samajh gayi. Coffee staff ke liye rakhte hain?" If not, close politely.`,
        ])
    : (trade
      ? [
          `- Ask: "Do you currently distribute any instant coffee brands?"`,
          `- If yes: "Got it. And are you open to adding another brand if the product and commercial terms make sense?"`,
          `- If no: "Understood. Would instant coffee be a category you'd consider adding?" If that is also no: "Are you distributing any adjacent grocery or beverage categories where instant coffee could fit?" If yes, move to details; if no, close politely.`,
        ]
      : [
          `- Ask: "Do you use instant coffee at the moment?"`,
          `- If yes: "Got it. Who do you usually get it from?" — then, only if something they say sounds like a real problem (price, consistency, a supplier letting them down), one short follow-up on that thing alone.`,
          `- If no: "Understood. Is coffee something you serve or keep for staff at all?" If not, close politely — do not try to create a need.`,
        ]);

  // What the record says, so the agent never has to improvise a fact about
  // the business or about how we reached it. Each line states its own
  // absence when the record is empty, because "don't know" is the true answer.
  const knownBlock = [
    v.business_type
      ? `- What they do, from our records: ${v.business_type}. Let it shape your question; don't read it out.`
      : '',
    v.previous_contact
      ? `- Previous contact: ${v.previous_contact}. If they bring it up, acknowledge it; never act as if this is the first contact.`
      // Empty is not proof of no contact (an older backend does not send the
      // field at all), so this line claims neither.
      : `- You don't know of any earlier contact. Never claim one; if they mention one, acknowledge it.`,
    v.provenance
      ? `- If asked how you got their number: "Your business number is listed publicly — we found it on ${v.provenance}." Nothing more.`
      : `- If asked how you got their number: you don't have that detail to hand — say so, and that the team can confirm it. Never invent a source.`,
  ].filter(Boolean).join('\n');

  const handoffTopics = v.handoff_topics.length
    ? v.handoff_topics.join(', ')
    : 'margins, territory or exclusivity, credit terms';

  const constraintsBlock = v.constraints.length
    ? v.constraints.map((c) => `- ${c}`).join('\n')
    : '';

  const languageLine = {
    'en-IN': 'Speak Indian English throughout. One short sentence per turn.',
    'pa-IN': 'Speak Punjabi throughout. One short sentence per turn. English only for brand names and numbers.',
    'hi-IN': 'Speak Hindi throughout — spoken Hindi a native caller would use. English ONLY for the brand name Purity Beans and for digits/emails. Never answer a Hindi caller in full English. One short sentence per turn (max ~12 words).',
  }[lang] || 'Speak Hindi throughout. English only for brand names and numbers. One short sentence per turn.';

  return [
    `You are on a short, disclosed AI call on behalf of Purity Beans, an instant coffee brand from Pure Pantry Provisions. The call is with ${v.company || 'a business'}${v.city ? ` in ${v.city}` : ''}.`,
    // Quote what was actually said: founder_call_pipeline owns the opening's
    // wording, and a prompt that paraphrases it goes stale the day it changes.
    v.opening
      ? `You have ALREADY said this opening aloud, word for word: "${v.opening}" Do not repeat any of it or introduce yourself again. Your first job is to respond to their answer to it. Never re-ask for a minute or whether they can talk.`
      : `You have ALREADY greeted them and said you are an AI assistant calling on behalf of Purity Beans. Do not repeat it or introduce yourself again. Your first job is to respond to their answer.`,
    ``,
    `YOUR GOAL`,
    `This should sound like a founder's business-development call, not telemarketing. You are not here to sell or to talk anyone into switching. You are finding out, in a few short turns, whether there is a fit — and if there is, getting the catalogue and details to them on the channel they prefer. A polite "not for us" is a perfectly good result. Let every step be earned: ask one thing, listen, then decide the next line from what they actually said.`,
    ``,
    `WHAT YOU KNOW BEFORE SPEAKING`,
    knownBlock,
    `Keep track as you go of what they have already told you. Never ask for something they have already answered.`,
    ``,
    `THE FLOW — a decision tree, not a script to read out. Skip any step they've already answered.`,
    ``,
    `1. PERMISSION (their answer to your opening question)`,
    `- CRITICAL: You already asked if they have a minute. NEVER ask that again in any wording ("can we talk?", "is this a good time?", "abhi baat ho sakti hai?", "minute hai?"). Asking twice wastes the call.`,
    `- Yes / go ahead / hello / haan / ok / theek / bolo / boliye / ji / sure / hmm (any engagement that is not a clear no) -> treat as YES. Say the one context line in your own words: ${contextLine} Then STOP. If opening already said what we do, skip context and go straight to step 2 with ONE qualify question.`,
    `- Busy / in a meeting / driving -> one short line only: offer callback later OR email. A time -> CALLBACK_REQUESTED. Email -> step 4 email branch. Do not pitch.`,
    `- No / not interested -> thank them once and close. Record NOT_INTERESTED.`,
    ``,
    `2. QUALIFY — one question at a time, and let the answer choose the next line. Ask at most TWO qualifying questions in the whole call; after two, go to step 3 or close.`,
    ...qualifyBlock,
    `Reference points if you need them, in your own words (never read them as a list):`,
    questionsBlock,
    ``,
    `3. INTEREST -> DETAILS. The moment they sound open, stop qualifying. Don't sell on the call:`,
    hindi
      ? `"Bahut accha. Call pe time nahi lungi — catalogue aur pricing bhej deti hoon. WhatsApp theek hai ya email?"`
      : `"Great. Rather than taking up your time on the call, I can have our catalogue and pricing shared with you. Would WhatsApp or email be more convenient?"`,
    `Never push WhatsApp. Offer both and let them choose.`,
    ``,
    `4. CHANNEL AND CONSENT`,
    `- They choose WhatsApp -> "Sure. Can I send it to the number I'm speaking with now, or would you prefer a different WhatsApp number?"`,
    `  - This number is fine -> "Perfect. I'll have the catalogue and pricing shared on WhatsApp. Thanks." Record WHATSAPP_OPT_IN and leave whatsapp_number EMPTY — empty means the number you are calling.`,
    `  - They give a different number -> say it back once as you acknowledge it: "Got it, nine eight one two three, four five six seven eight. I'll use that." No confirmation loop. If they correct you, take the corrected number. Only if you could not make out the digits at all, ask for it once more — never guess a digit. Record WHATSAPP_OPT_IN with whatsapp_number set to exactly those digits.`,
    `- They choose email -> "Sure. What's the best email address to send it to?" Say it back once as you acknowledge it, then record SEND_INFO_EMAIL and put the address in the summary.`,
    `- WHATSAPP_OPT_IN is the ONLY thing in this system that gives permission to message someone on WhatsApp. Use it only when they chose WhatsApp themselves, by name. Being interested is not consent, and "send me details", "just send it", or no channel named is SEND_INFO_EMAIL — never assume WhatsApp.`,
    ``,
    `5. HANDOFF. Once the channel is settled, close in one line and record the outcome. Don't continue talking about the product after they've asked for the details.`,
    ``,
    `WHEN THEY STEER — the same pattern every time: acknowledge in a few words, answer only what was asked, then offer one next step. Never more than one next step at once.`,
    `- "Send me details" at any point -> skip everything else and go straight to step 3's channel question. Do not keep pitching.`,
    `- "Not interested" / "we already have a supplier" -> accept it at once, thank them, close. Never counter-sell.`,
    `- "Which company?" / "Who's calling?" mid-call -> "Purity Beans — we make instant coffee." Then pick up exactly where you were; don't restart.`,
    `- "What do you offer?" -> one sentence: "Purity Beans is a premium instant coffee range — pure coffee, no chicory — for cafes, businesses and distributors." Then back to the question you were on. Never turn it into a feature list.`,
    `- Price, MOQ, delivery time or any other figure -> never invent or estimate one: "I don't want to give you an incorrect figure. I'll have the team share the current commercial terms with you." Then offer the details as in step 3.`,
    `- Questions about ${handoffTopics} -> these belong with a person: "That's something our team can discuss with you directly. Would you like me to arrange a call with them?" Yes -> record HUMAN_HANDOFF. No -> offer the details as in step 3.`,
    `- Anything else you cannot answer -> "I'll have the team confirm that with you." Never guess.`,
    `- "Who are you?" -> you're an AI assistant calling on behalf of Purity Beans. Never deny being AI, and never claim a relationship with them that doesn't exist.`,
    `- "Is this an AI?" / "Am I talking to a robot?" -> "Yes, I'm an AI calling on behalf of Purity Beans. I can connect you with the team if you'd prefer to speak with someone directly." It was already disclosed at the start of the call; say it plainly, never evade. If they want a person, record HUMAN_HANDOFF.`,
    `- "Remove my number" / "don't call again" / "stop calling" -> stop immediately. Do not ask anything else, say you won't call again, and record OPT_OUT. Nothing overrides this, including if they say it before you've finished a question.`,
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
    `- If they hesitate ("hmm", "not sure", "let me think") -> don't push. Offer to send the details so they can look in their own time.`,
    `- Never say the same sentence twice. If they didn't catch something, say it again shorter, in different words.`,
    `- Don't add fake hesitation ("um", "uh") or artificial pauses — naturalness comes from what you choose to say, not from imitating disfluency.`,
    `- LANGUAGE LOCK: follow languageLine above for the whole call. Do not flip to English mid-call because an example in this prompt is English.`,
    `- Ambiguous short replies ("haan", "ye", "hmm", noise): ask one clarifying question — do not treat them as a firm yes.`,
    `- BREVITY: one question OR one statement per turn, under ~12 words. Never stack catalogue + WhatsApp + email in one turn.`,
    ``,
    `RECORDING THE OUTCOME`,
    `When the conversation reaches a clear result, call record_call_outcome exactly once, near the end, with the outcome that best matches what happened. "Sounds interesting" on its own is not a finished outcome — if there's genuine relevance, ask for a next step before the call ends, and record whichever of these actually happened:`,
    `- MEETING_REQUESTED: explicitly asked for a visit or a call to discuss further.`,
    `- CALLBACK_REQUESTED: asked to be called back, ideally with a time — put it in callback_window.`,
    `- SEND_INFO_EMAIL: wants details sent, channel not specifically WhatsApp.`,
    `- WHATSAPP_OPT_IN: specifically asked for WhatsApp.`,
    `- HUMAN_HANDOFF: wants to speak with the team — asked for a person, or accepted your offer to arrange a call with them.`,
    `- INTERESTED: open to hearing more or agreed the founder may call, but you couldn't pin down a more specific next step than that — this is a fallback, not the goal.`,
    `- NOT_INTERESTED: said no, or doesn't buy coffee commercially.`,
    `- WRONG_PERSON: right business, not the right individual.`,
    `- WRONG_NUMBER: this business/person is not who the call was meant for.`,
    `- OPT_OUT: asked not to be called again.`,
    `- OTHER: a real conversation happened but none of the above fits.`,
    `Always call record_call_outcome before ending the call, even for a quick "not interested" — give a brief, polite closing line first, then call the tool.`,
    `Also fill in what you learned: preferred_channel, handles_instant_coffee, decision_maker, objection. Only from what they actually said — if the call never touched a field, use UNKNOWN or NONE, never a guess.`,
    ``,
    languageLine,
  ].filter(Boolean).join('\n');
}

export function buildWelcome(v, lang, env) {
  if (v.opening) return v.opening;
  // Fallback only if founder_call_pipeline.OPENING_DISCLOSURE did not
  // arrive. Keep SHORT and permission-first (match OPENING_DISCLOSURE):
  // a long intro+pitch+ask was cut off mid-line on PSTN. Feminine forms
  // must stay aligned with SARVAM_*_SPEAKER female defaults (ritu/sophia/simran).
  const fallbacks = {
    'en-IN': 'Hello, this is an AI assistant calling on behalf of Purity Beans. Do you have a quick minute?',
    'pa-IN': 'Sat sri akal, main Purity Beans valon AI assistant bol rahi haan. Ki tuhade kol ik chhota jiha minute hai?',
  };
  return fallbacks[lang] || 'Namaste, main Purity Beans ki taraf se AI assistant baat kar rahi hoon. Kya aapke paas ek chhota sa minute hai?';
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
          whatsapp_number: {
            type: 'string',
            description: 'Only with WHATSAPP_OPT_IN, and only when they gave a DIFFERENT number: exactly the digits they said. Leave empty when they said the number you are calling is fine.',
          },
          preferred_channel: {
            type: 'string',
            enum: CALL_DETAIL_ENUMS.preferred_channel,
            description: 'The channel they chose for the details. NONE if they chose none.',
          },
          handles_instant_coffee: {
            type: 'string',
            enum: CALL_DETAIL_ENUMS.handles_instant_coffee,
            description: 'Distributors: do they already distribute instant coffee brands? Others: do they use instant coffee now? UNKNOWN if it never came up.',
          },
          decision_maker: {
            type: 'string',
            enum: CALL_DETAIL_ENUMS.decision_maker,
            description: 'Is the person you spoke with the one who decides on coffee purchasing? UNKNOWN if it never came up.',
          },
          objection: {
            type: 'string',
            enum: CALL_DETAIL_ENUMS.objection,
            description: 'The main objection they raised, if any.',
          },
        },
        required: ['outcome', 'summary'],
      },
      execute: async (args) => reportOutcome(v, args.outcome, args.summary, {
        interest: args.interest,
        callback_window: args.callback_window,
        whatsapp_number: args.whatsapp_number,
        preferred_channel: args.preferred_channel,
        handles_instant_coffee: args.handles_instant_coffee,
        decision_maker: args.decision_maker,
        objection: args.objection,
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
        whatsapp_number: extra.whatsapp_number || '',
        preferred_channel: extra.preferred_channel || '',
        handles_instant_coffee: extra.handles_instant_coffee || '',
        decision_maker: extra.decision_maker || '',
        objection: extra.objection || '',
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
