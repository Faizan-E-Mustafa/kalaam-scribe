# TICKET 23: sherpa-onnx benchmark on phone

## Goal
Compare sherpa‑onnx inference latency and accuracy against the current AAR baseline on the Android phone.

## Acceptance Criteria
- [ ] Identical dictation clips (5‑10 short WhatsApp‑style sentences, +1 longer clip) transcribed with both AAR (baseline) and sherpa‑onnx.
- [ ] RTF (real‑time factor) measured for each clip on both back‑ends.
- [ ] Accuracy comparison: % of clips where output stays in Roman Urdu/Latin script vs. English leakage.
- [ ] Thread config tested: auto‑select 6 vs. user‑override 4 vs. 8.
- [ ] Results documented in a shared table (mean RTF, accuracy %).
- [ ] Recommendation documented: stay with AAR + auto, or migrate to sherpa‑onnx.

## Notes
- Baseline data already exists from earlier laptop tests (threads 4→6→8 RTF: 0.39, 0.43, 0.16 for different models).
- phone on‑device measurement will be the definitive data point.
- Should run at least 3 runs per clip and average RTF.

## Dependencies
- Clips from `/var/tmp/urdu-eval/clips/` or recorded via WSLg `parecord`.
- Access to phone device (or emulator with sherpa‑onnx binary).
- Script to measure elapsed time from audio start to transcript output.

## Labels
benchmark, sherpa-onnx, performance, the phone, threads