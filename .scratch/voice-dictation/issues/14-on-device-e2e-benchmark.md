# 14: On-device E2E acceptance + benchmark

**What to build:** The final acceptance gate: verify the full dictation loop on
the A50 against known English and Roman-Urdu phrases, confirm the resident Model
is reused across repeated dictations, and benchmark the Models on the A50 to
confirm the default and speed/accuracy trade-offs.

**Blocked by:** 13 (Foreground service + launch surface), 10 (Model catalog +
downloader).

**Status:** ready-for-agent

## Acceptance criteria

- [ ] Record a known English phrase and a known Roman-Urdu phrase on the A50;
      transcript on the clipboard matches within a tolerable WER.
- [ ] Roman-Urdu output stays in Roman/Latin script (auto-detect path; do not
      leak Urdu script).
- [ ] Two (or more) dictations in one session reuse the resident Model — no
      per-inference reload observed.
- [ ] Recording/transcribing works when started from the home-screen widget in
      the background.
- [ ] Benchmark wall-clock transcription time vs audio length and qualitative
      accuracy for the six Models on the A50, confirming the default selection
      and whether any Model should be dropped or re-defaulted.
