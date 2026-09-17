/**
 * SIP-trunk selection by destination phone.
 *
 *   +91 (India)          → Plivo   — Zentrunk outbound trunk, KYC-approved
 *                                    Indian caller ID (+918031905525),
 *                                    required for outbound voice traffic to
 *                                    Indian numbers.
 *   everything else      → Twilio  — global reach, per-minute billing,
 *                                    standard E.164 international routing.
 *
 * Both trunks are registered in LiveKit Cloud; this module just looks at the
 * destination number and returns which trunk to use for a given call.
 *
 * Renamed 2026-09-15: this read "Vobiz" throughout even though
 * LIVEKIT_SIP_TRUNK_ID has pointed at the Plivo Zentrunk trunk
 * (purity-plivo-outbound, confirmed via SipClient.listSipOutboundTrunk())
 * since scripts/wire_livekit_trunk.mjs was run — Vobiz was the India carrier
 * before Plivo KYC was approved, and nothing here was updated when the
 * trunk itself was. The variable name is what a future reader trusts fastest
 * on a live call issue; a name that lies about the carrier is worse than no
 * comment at all.
 *
 * If a non-India number arrives and Twilio isn't configured, we throw rather
 * than dial through the Plivo trunk — a DLT-registered Indian caller ID
 * dialing a US number would either fail at the carrier or look like spam.
 * Loud failure with a clear error message is the right behaviour.
 */

const PLIVO_TRUNK_ID    = process.env.LIVEKIT_SIP_TRUNK_ID;
const PLIVO_FROM_NUMBER = process.env.PLIVO_FROM_NUMBER || process.env.VOBIZ_FROM_NUMBER || '';
const TWILIO_TRUNK_ID   = process.env.LIVEKIT_SIP_TRUNK_ID_TWILIO || '';
const TWILIO_FROM_NUMBER = process.env.TWILIO_FROM_NUMBER || '';

/**
 * Pick the outbound trunk for a destination phone number.
 *
 * @param {string} phone — E.164 (with leading +)
 * @returns {{ trunkId: string, fromNumber: string, carrier: 'plivo'|'twilio' }}
 * @throws if no configured trunk can serve the destination
 */
export function pickTrunkForPhone(phone) {
  if (!phone || typeof phone !== 'string' || !phone.startsWith('+')) {
    throw new Error(`pickTrunkForPhone: phone must be E.164 (got: ${phone})`);
  }

  if (phone.startsWith('+91')) {
    if (!PLIVO_TRUNK_ID) {
      throw new Error('pickTrunkForPhone: +91 destination but LIVEKIT_SIP_TRUNK_ID (Plivo) not set');
    }
    return { trunkId: PLIVO_TRUNK_ID, fromNumber: PLIVO_FROM_NUMBER, carrier: 'plivo' };
  }

  if (!TWILIO_TRUNK_ID) {
    throw new Error(
      `pickTrunkForPhone: non-India destination (${phone}) but LIVEKIT_SIP_TRUNK_ID_TWILIO not set. ` +
      `Run \`node src/create-twilio-trunk.mjs\` to provision Twilio, then add the trunk id to .env.`,
    );
  }
  return { trunkId: TWILIO_TRUNK_ID, fromNumber: TWILIO_FROM_NUMBER, carrier: 'twilio' };
}
