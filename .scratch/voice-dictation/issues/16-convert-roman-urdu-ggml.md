# 16: Convert Roman-Urdu to GGML (f16 + q4_0)

**What to build:** Download the Roman-Urdu Whisper fine-tune
(`cheetos18/whisper-small-roman-urdu`) and convert it to the whisper.cpp GGML
format, producing both an unquantized f16 file and a quantized q4_0 file, so the
app has a standard-quality and a fast on-device model to offer.

**Blocked by:** None (can start immediately).

**Status:** done

## Outputs (verified on dev machine)

- `ggml-model-f16.bin` — 487,601,984 bytes (~465 MiB, f16 small)
- `ggml-model-q4_0.bin` — 145,458,032 bytes (~139 MB, q4_0 small)
- Both transcribe `samples/rec.wav` → **"aap ka kya haal hai?"** with `-l auto`,
  clean Roman/Latin script (never Urdu script).
- Model metadata: `n_vocab = 51865` (Roman-Urdu fine-tune), `n_audio_head = 12`,
  `n_audio_layer = 12` → small (`type = 3`).
- q4_0 much faster than f16 on the same 4.5 s clip (11 s vs 27 s).

- [x] `ggml-model-f16.bin` produced from the fine-tune (f16, small, ~465 MiB)
- [x] `whisper-quantize` applied to produce `ggml-model-q4_0.bin` (~139 MB)
- [x] Both verified under whisper.cpp with `-l auto` → clean Roman/Latin script
- [x] Confirm `type = 3 (small)`; never rely on `-l ur` (leaks Urdu script)
