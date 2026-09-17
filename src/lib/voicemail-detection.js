/**
 * Voicemail / gatekeeper detection, extracted from livekit-agent.js so it can
 * be unit-tested without pulling in the LiveKit CLI (importing
 * livekit-agent.js directly runs cli.runApp() as a side effect).
 *
 * Real call #2953 (Storico) burned 2 minutes pitching to a voicemail
 * recording before this existed. Patterns are broad: partial match anywhere
 * in any of the first 4 user transcripts should trigger an immediate hangup.
 */

const VOICEMAIL_PATTERNS = [
  /व[ोॉ]इस[\s\-]*मे/,
  /फ[ॉो]रवर्डेड\s+टू/,
  /न[ॉो]ट\s+अव[ेै]लेबल/,
  /रिक[ॉो]र्ड\s+य[ोौ]र\s+म[ैे]सेज/,
  /लीव\s+अ\s+म[ैे]सेज/,
  /एट\s+द\s+ट[ोौ]न/,
  /आफ्टर\s+द\s+बीप/,
  /व्हेन\s+य[ोौ]\s+ह[ैै]व\s+फिनिश्ड/,
  /इस\s+समय\s+उपलब्ध\s+नहीं/,
  /कृपया\s+संदेश\s+छोड़/,
  /voicemail|voice\s*mail/i,
  /answering\s*machine/i,
  /please\s+(record|leave)\s+(your\s+)?(message|name)/i,
  /(after|at)\s+the\s+(tone|beep)/i,
];

// Fallback for the pattern list above: an exact-phrase match assumes the
// same English greeting comes back the same way every time, but Sarvam's
// hi-IN STT transliterates English speech into Devanagari inconsistently —
// vowels and consonants drift with pronunciation. Real call: an English
// gatekeeper greeting ("if you record your name and reason for calling, I
// will see if this person is available") came back as "इफ यू रेकॉर्ड योर
// नेम एंड रीजन फॉर कॉलिंग आई विल सी इफ दिस पर्सन इज अवेलेबल" — no
// VOICEMAIL_PATTERNS entry matches that exact wording, and the next
// gatekeeper will phrase it differently again. Chasing each new wording with
// another full-phrase regex never catches up.
//
// So this checks for the underlying signal instead of the phrasing: English
// voicemail/gatekeeper greetings are loanword-dense when transliterated —
// "record", "message", "available" etc. stay recognisable even though the
// sentence around them doesn't. A single such loanword appears in ordinary
// Hinglish too (someone might say "मैसेज कर दो" mid-conversation), so this
// only fires on TWO OR MORE distinct concepts in the same turn, which
// ordinary conversational Hindi/Hinglish is very unlikely to produce by
// coincidence.
const VOICEMAIL_LOANWORD_FRAGMENTS = [
  /रे?क[ॉा]र्ड/,             // record
  /मैसे?ज|मेसे?ज/,           // message
  /नेम/,                     // name
  /रीज[ऩ]|रिज़?न/,           // reason
  /क[ॉा]लिंग/,               // calling
  /अवेलेबल/,                 // available
  /बीप/,                     // beep
  /ट[ोौ]न/,                  // tone
  /मोमेंट/,                  // moment
  /बिज़?ी/,                  // busy
  /करें?टली/,                // currently
  /अनेबल/,                   // unable
  /कनेक्ट/,                  // connect
  /एक्सटें?शन/,              // extension
  /मेलब[ॉा]क्स/,             // mailbox
  /पर्सन/,                   // person
];

export function detectVoicemail(transcript) {
  const exact = VOICEMAIL_PATTERNS.find((p) => p.test(transcript));
  if (exact) return { matched: exact.toString(), kind: 'phrase' };

  const hits = VOICEMAIL_LOANWORD_FRAGMENTS.filter((p) => p.test(transcript));
  if (hits.length >= 2) {
    return { matched: hits.map((p) => p.toString()).join(', '), kind: 'loanword-fragments' };
  }
  return null;
}

export { VOICEMAIL_PATTERNS, VOICEMAIL_LOANWORD_FRAGMENTS };
