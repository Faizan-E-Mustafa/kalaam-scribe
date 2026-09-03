# 28: Smartphone ONNX pipeline for Dolphin lang/region (export decoder)

**What to build:** On-device (Android) recognition with Dolphin that honors an explicit
`lang`/`region` selection, without waiting for upstream sherpa-onnx (#3904). Because the
shipped sherpa CTC `model.onnx` contains only the encoder+CTC head, this ticket exports
Dolphin's attention decoder to ONNX from the original PyTorch weights and runs a
standalone ONNX Runtime inference pipeline that prepends `<lang><region>` tokens to the
decoder prompt — the same design the upstream issue proposes.

**Blocked by:** 27 (Laptop validation of Dolphin lang/region)

**Status:** ready-for-agent

- [ ] Export `encoder` and `decoder` (and CTC head) to ONNX from the Dolphin PyTorch
      weights, reusing or replacing the sherpa `model.onnx`; confirm the decoder onnx
      has `encoder_out` + token-id inputs and logits output.
- [ ] Feature extraction reproduced in the inference path: 16 kHz, 80 log-mel,
      STFT `n_fft=512, hop=128` (mirror the Dolphin frontend), with global CMVN
      from `feats_stats.npz` / `mean`+`invstd` metadata.
- [ ] Standalone ONNX Runtime script (Python-first) that: runs encoder → builds the
      `<sos><lang><region>` decoder prefix → greedy/beam-search decodes with KV cache
      → maps token ids to text via the BPE `units.txt`/`tokens.txt`.
- [ ] On the laptop, the standalone ONNX pipeline matches the `dolphin` package output
      (from ticket 27) for `ur`/`PK` on the same clip.
- [ ] Decide on Android integration path: (a) ONNX Runtime Mobile standalone engine
      replacing/paralleling `DolphinCtcEngine`, or (b) contribute the decoder+config
      upstream to sherpa-onnx and consume on the next release. Record the choice and
      trade-offs (per AGENTS.md: note both, recommend one).
- [ ] `lang` field in the app's result JSON is populated from the pinned/decoded
      language when the model provides it.

## Comments
