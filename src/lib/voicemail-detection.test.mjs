import { test } from 'node:test';
import assert from 'node:assert/strict';
import { detectVoicemail } from './voicemail-detection.js';

test('catches the real gatekeeper greeting Sarvam STT mistransliterated', () => {
  // Actual STT output from a real call, 2026-09-15: an English gatekeeper
  // greeting ("if you record your name and reason for calling, I will see
  // if this person is available") transliterated into Devanagari that no
  // fixed VOICEMAIL_PATTERNS entry matched — the call proceeded as if a
  // human had answered.
  const transcript = 'इफ यू रेकॉर्ड योर नेम एंड रीजन फॉर कॉलिंग आई विल सी इफ दिस पर्सन इज अवेलेबल।';
  const result = detectVoicemail(transcript);
  assert.ok(result, 'expected the loanword-fragment heuristic to catch this');
  assert.equal(result.kind, 'loanword-fragments');
});

test('still catches known native-Hindi voicemail phrasing (existing pattern)', () => {
  const transcript = 'यह नंबर इस समय उपलब्ध नहीं है, कृपया संदेश छोड़ें।';
  const result = detectVoicemail(transcript);
  assert.ok(result);
  assert.equal(result.kind, 'phrase');
});

test('still catches plain English voicemail greetings (existing pattern)', () => {
  const result = detectVoicemail('Please leave your message after the beep.');
  assert.ok(result);
  assert.equal(result.kind, 'phrase');
});

test('does not false-positive on an ordinary Hindi/Hinglish reply', () => {
  const result = detectVoicemail('हां मैं ही देखता हूं कॉफी वाला काम, बोलिए क्या चाहिए।');
  assert.equal(result, null);
});

test('a single loanword mention (e.g. asking to send a WhatsApp message) does not trigger it', () => {
  const result = detectVoicemail('ठीक है आप मुझे मैसेज कर दो डिटेल्स के साथ।');
  assert.equal(result, null);
});

test('two or more distinct loanword fragments together do trigger it', () => {
  const result = detectVoicemail('कृपया अपना नेम बताएं और मैसेज छोड़ें।');
  assert.ok(result);
  assert.equal(result.kind, 'loanword-fragments');
});
