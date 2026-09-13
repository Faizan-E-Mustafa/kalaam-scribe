# 06: Deploy Roman-Urdu model to the phone

**What to build:** The validated Roman-Urdu whisper.cpp model
(`ggml-model-q4_0.bin`, small, q4_0, 139 MB) is deployed on the phone and
`dictate` switches to it, giving on-device dictation that transcribes spoken
Roman Urdu into clean Roman/Latin script.

**Blocked by:** nothing (validation complete on dev machine — see below).

**Status:** in progress

## Validation already done (dev machine)

- `cheetos18/whisper-small-roman-urdu` (`vocab_size` 51865, standard
  multilingual) converts to whisper.cpp GGML via `convert-h5-to-ggml.py`.
- Converted to `ggml-model-q4_0.bin` (139 MB). Both f16 and q4_0 verified
  running under whisper.cpp (ticket 16).
- Real voice test (`samples/rec.wav`, user said "aap ka kya haal hai"):
  - **`-l auto` → clean Roman:** `aap ka kya haal hai`  ✓
  - `-l ur` → leaks Urdu script (`aap ka kya haal ہے۔`) — do NOT force `ur`.
  - Original HF model (transformers) ground truth: auto/en → clean Roman.

## Steps

- [ ] Get `ggml-model-q4_0.bin` (139 MB) onto the phone at
      `~/whisper.cpp/models/ggml-model-q4_0.bin`
      (now downloadable from `femustafa/voicedictation-models`, or transfer
      method TBD: USB/adb push, `termux-setup-storage` + file copy, or cloud
      download from a link).
- [ ] `dictate.sh` already updated in-repo: `MODEL` default → `ggml-model-q4_0.bin`,
      `-l en` → `-l auto` (commit the change).
- [ ] Push updated `dictate.sh` to the phone.
- [ ] Verify a full record→transcribe→clipboard→paste loop with a Roman-Urdu
      message on the phone (can reuse ticket 05 flow).
- [ ] On-phone transcript must be Roman/Latin script, not Urdu script.
- [ ] Re-run `tools/bench/bench.sh` to capture small-model timing/RTF on the phone
      (decides whether q4 small is acceptable vs falling back to base).

## Notes / risks

- Small model on Exynos 9610 (CPU-only, 4 GB RAM) will be slower than `base.en`.
  If too slow, options: accept latency, use a smaller quant (q5/q4), or keep
  `base.en` for English and only the RU model for Roman-Urdu mode.
- whisper.cpp must be a recent build to load this model (it was built on the phone
  already). Volatile: the model is multilingual vocab 51865 — fine.
- Keep `-l auto`. Do not set `-l ur`.
