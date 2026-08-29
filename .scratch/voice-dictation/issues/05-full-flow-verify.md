# 05: Verify full flow into WhatsApp

**What to build:** The complete end-to-end dictation flow is verified on the A50 in airplane mode: record → local transcription → clipboard copy → paste into a WhatsApp chat → manual review → send. This confirms it is truly offline and usable for real messages.

**Blocked by:** 04 (Benchmark transcription on the A50)

**Status:** ready-for-agent

- [ ] dictation → clipboard → paste → send works end-to-end in a real WhatsApp chat
- [ ] Verified working with internet disabled (airplane mode) proving full locality
- [ ] Latency and accuracy are acceptable for real use
- [ ] Any rough edges in the interaction are documented

## Acceptance criteria notes
Done when the user can dictate a real WhatsApp message offline with an acceptable experience, and the Phase 1 prototype is usable.
