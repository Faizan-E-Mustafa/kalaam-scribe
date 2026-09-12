# Static int8 Quantization — Findings & Decision (2026-09-12)

**Decision: NOT pursued now.** Dynamic int8 (`quantize_int8.py`, `quantize_dynamic`,
`op_types_to_quantize=["MatMul"]`) remains the shipped quantization tier. Findings
recorded here so the analysis doesn't have to be redone.

## What dynamic int8 actually covers (check the `int8-small/decoder.onnx` initializers)

The `["MatMul"]` pass quantized MORE than just MatMul:

| weight | dtype now | size (small) |
|---|---|---|
| 144 × attention/FFN MatMul weights | int8 | ~70 MB |
| `dolphin.decoder.output_layer.weight` (Gemm output proj) | **int8** (768×40002) | 30.7 MB |
| `dolphin.decoder.embed.0.weight` (input embedding, `Gather`-fed) | **fp32** (40002×768) | **122.9 MB ← only big holdout** |

The output-layer `Gemm` got quantized because ORT rewrites a constant-weight `Gemm`
to `MatMulInteger` internally even when only `MatMul` is requested. The input
embedding is consumed by `Gather`, which dynamic quantization never touches — that
single fp32 table is the entire reason int8-small (748 MB) exceeds fp16-small (699 MB).

## Is static int8 (`quantize_static`, QDQ) possible?

Yes. ORT 1.29.0's QDQ registry (`onnxruntime/quantization/registry.py`) supports every
op the Dolphin graph needs: `MatMul`, `Gemm`, `Gather`/`GatherElements`, `Slice`,
`LayerNormalization`. `Softmax` and `ArgMax` are NOT in the QDQ registry, so they stay float.

## Why we're NOT doing it (costs vs. benefit)

- **Requires calibration data** — realistic decoder feeds (encoder outputs on Urdu
  speech + KV caches at several prefix lengths), buildable from `verify_dakeqq_beam.py`.
- **Quantizes activations too** — including the full-logits Gemm output that the beam
  search scores. `small` already picks the correct hypothesis by a hair (the
  length-normalized-scoring fix); int8 activation noise on the final 40,002-wide logits
  is exactly where it's fragile. Would need full Urdu re-verification.
- **File-size gain ≈ zero beyond weight-only** — activations aren't stored in the file.
  Static QDQ's only real value would be a future NNAPI/accelerator (int8 EP) speed path.

## Size projections (decimal MB; encoder + decoder)

| variant | current int8 | fp16 tier | static QDQ (A) | weight-only embed int8 (B) | embed→fp16 (C) |
|---|---|---|---|---|---|
| small | 366.8 + 381.1 = 747.9 | 377.8 + 321.7 = 699.4 | ~289 dec / ~656 total | ~289 dec / ~656 total | ~320 dec / ~686 total |
| base | 109.9 + 153.3 = 263.1 | 124.2 + 126.3 = 250.4 | ~92 dec / ~202 total | ~92 dec / ~202 total | ~112 dec / ~222 total |

- (A) true static QDQ: matches (B) for file size, adds activation quant + calibration.
- (B) weight-only int8 embedding: custom surgery (`quantize_int8.py` extension) — int8
  `embed.0.weight` + per-row fp32 scale (~40002 × (768 + 4) bytes = ~30.9 MB) +
  `DequantizeLinear` after `Gather`. No calibration, activations stay fp32 (protects the
  fragile final logits). **Recorded recommendation if revisited.**
- (C) convert only the embedding to fp16: trivial, ~zero accuracy risk (matches the
  verified DakeQQ fp16 tier), still under the fp16 tier but barely.

## Numbers referenced

- int8-small decoder initializers: `embed.0.weight` fp32 122.9 MB; `output_layer.weight_quantized`
  int8 (768,40002) 30.7 MB; MatMuls int8 ~2.4 MB each (12 layers × 6).
- fp16-arm small decoder: both 40002×768 tables fp16 61.4 MB; MatMuls fp16 4.7 MB.
- ORT version inspected: 1.29.0 (`onnx`, `onnxruntime` in `tools/validate_stt/.venv`).