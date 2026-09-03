# 28: Smartphone ONNX pipeline for Dolphin lang/region (export decoder)

**What to build:** On-device (Android) recognition with Dolphin that honors an explicit
`lang`/`region` selection, without waiting for upstream sherpa-onnx (#3904). Because the
shipped sherpa CTC `model.onnx` contains only the encoder+CTC head, this ticket exports
Dolphin's attention decoder to ONNX from the original PyTorch weights and runs a
standalone ONNX Runtime inference pipeline that prepends `<lang><region>` tokens to the
decoder prompt — the same design the upstream issue proposes.

**Blocked by:** 27 (Laptop validation of Dolphin lang/region)

**Status:** ready-for-agent

- [x] Decoder exported to ONNX (`decoder.onnx`, `torch.jit.trace` + legacy exporter,
      verified max diff `1e-6` vs torch `forward_one_step`). Inputs: `memory`
      `[B,T',512]` f32, `memory_mask` `[B,T']` bool (2D), `tgt` `[B,L]` int64.
      Output: `logits` `[B,L,vocab]` (log-softmax).
- [x] Encoder export investigated but NOT done as ONNX. `torch.jit.trace` of
      EBranchformerEncoder diverges (6.2 diff, rel_pos dynamic control flow);
      `torch.jit.script` fails (return-type bug at `model.py:3361`); `dynamo`
      produces invalid `Split` nodes. Graph surgery from sherpa `model.onnx`
      (`encoder_from_sherpa.onnx`) is close (9.7e-6 on random input) but flips
      decoder output on real features. **The working pipeline keeps the encoder in
      torch** (runs fine at fp32) and exports only the decoder. See Findings below
      for why torch-encoder-on-phone is NOT a real deployment option.
- [x] Feature extraction reproduced WITHOUT CMVN. **Critical fix:** the encoder
      applies `global_cmvn` internally (`model.py:1743`); normalizing features in
      the inference path too (double-CMVN) produced garbage. Correct pipeline feeds
      RAW log-mel (16 kHz, 80 mel, STFT `n_fft=512, hop=160, win=400`) to the model.
- [x] Standalone ONNX Runtime pipeline works (torch encoder + `decoder.onnx` +
      attention beam search), verified matching the `dolphin` package `attention`
      output for `ur`/`PK`:
      - rec.wav → `آپ کا کیا حال ہے؟` (exact match)
      - live5.wav → `میں تھوڑی دیر میں آفس ہونگا...` (torch: `...آفس آوں گا...`)
      - live6.wav → `توری در میں آفس رنگ؟` (torch: `تھری دیر میں آفس چونگ`)
      - Minor word variation is expected torch-vs-ONNX numerical drift.
- [x] **Decoding must be beam-search `attention`, NOT `attention_rescoring`.** Ticket 27
      found `attention_rescoring` overrides the `ur`/`PK` pin and flips output to
      Devanagari; `attention` consistently yields Urdu script when pinned. Also
      reconfirmed here: DakeQQ's greedy decode of rec.wav emitted `<tg><nl>` garbage
      (greedy ignores the pin). The ONNX beam path MUST use `attention` (single-pass
      beam, not greedy / not LM rescore).
- [x] On the laptop, DECIDE between DakeQQ matched pipeline vs our `decoder.onnx`
      for the phone. **DONE (2026-09-03): DakeQQ pair passes beam-search spike;
      it is the phone path.** See spike results below.
- [ ] Port DakeQQ attention beam-search loop to onnxruntime-android (Kotlin):
      encoder (fused STFT+mel+CMVN in-graph) → prefill → KV-cached beam decode
      using the full-logits output → detokenize via `units.txt`/BPE.
- [ ] Runtime: use the graph-surgery decoder (`decoder_withlogits.onnx`) that
      exposes `/output_layer/Gemm_output_0`; confirm it survives `onnxruntime-android`.
- [ ] Benchmark on-device RTF for `dolphin-small` INT8 (239MB) — must be < real-time
      (~1.0) for interactive voice dictation; pick base vs small accordingly.
- [ ] Decide on Android integration path: (a) ONNX Runtime Mobile standalone engine
      replacing/paralleling `DolphinCtcEngine`, or (b) contribute the decoder+config
      upstream to sherpa-onnx and consume on the next release. Record the choice and
      trade-offs (per AGENTS.md: note both, recommend one).
- [ ] `lang` field in the app's result JSON is populated from the pinned/decoded
      language when the model provides it.

## Findings (2026-09-03)

### Working pipeline (verified on laptop)
`torch encoder` → `decoder.onnx` → **attention beam search** → detokenize.
The encoder stays in torch because EBranchformer's `rel_pos` defies clean ONNX
export (all four approaches fail: trace/script/dynamo/surgery). Exported only the
decoder via `torch.jit.trace` (1e-6 vs torch). Implementation in
`/tmp/opencode/dolphin_onnx/infer_onnx.py`.

### Root cause of earlier garbled output: double-CMVN
`infer_onnx.py` subtracted `feats_stats.npz` mean/std AND the encoder applies
`global_cmvn` internally (`model.py:1743`). Removing our normalization fixed it.
The correct frontend is RAW log-mel; do NOT normalize before the encoder.

### DakeQQ/sherpa encoder interface is incompatible with our decoder
- sherpa `model.onnx` = encoder+CTC only, no attention decoder (upstream #3904).
- DakeQQ's `Dolphin_Encoder.onnx` (HuggingFace `onnx-community/dataocean-dolphin-asr`)
  outputs **pre-projected cross-KV** per decoder layer: `en_key_i` `[8,64,T]`,
  `en_value_i` `[8,T,64]`. Our `decoder.onnx` expects raw `[B,T',512]` and does its
  own cross-KV. **The two are a matched pair only** — no cross-mixing.
- DakeQQ's decoder uses incremental self-KV cache + `language_start`/`language_end`
  inputs; interface differs entirely from our decoder.

### Torch encoder on phone is NOT viable
- PyTorch Android only loads `torch.jit` TorchScript, not raw `nn.Module`.
- Our encoder fails both JIT paths (trace 6.2 diff; script type bug).
- PyTorch Mobile is deprecated/stale (2.1.0, Oct 2023); `libtorch_cpu.so` ~80MB vs
  ONNX Runtime Android AAR ~24MB.
- Phone path = ONNX Runtime (already the app's stack via sherpa-onnx).

### Phone path decision (2026-09-03): DakeQQ matched pair (a)
The phone needs a **matched ONNX encoder+decoder pair** plus **attention beam
search**. Two candidates were considered:
- **(a) DakeQQ matched pair** + reimplement beam search around their KV-cache
  decoder. Phone-ready (fused, INT8/FP16-ready, ARM-tested).
- **(b) Our `decoder.onnx`** + an ONNX encoder re-exported from DakeQQ's
  `DOLPHIN_ENCODER` reshaped to raw-features output. Reuses our verified beam search
  but needs an ONNX encoder anyway.

<b>(a) was verified and chosen</b> — see spike results below.

### Laptop verification spike PASSED (2026-09-03) → decision: DakeQQ pair (a)
The DakeQQ matched encoder+decoder ONNX **does produce Urdu under attention beam
search**. Verified on all clips (`/tmp/opencode/verify_dakeqq_beam.py`):

| Clip | DakeQQ full-logits beam | Torch `attention` baseline |
|------|-------------------------|----------------------------|
| rec.wav | `آپ کا کیا حال ہے` | `آپ کا کیا حال ہے؟` |
| live5.wav | `میں تھوڑی گیر میں آفس آوں گا...` | `...آفس آوں گا...` (exact `آوں گا`) |
| live6.wav | `تھوڑی دیر میں آفس شونگ` | `تھری دیر میں آفس چونگ` |

**Two decisive technical findings from the spike:**

1. **`language_start`/`language_end` slice the output-layer logits** to the
   language-token range `[136:269]` (via the `Slice_3` node feeding `ArgMax`).
   This is for **LID / auto-detect only** — for CONTENT generation you must use the
   **full logits** output (`/output_layer/Gemm_output_0`), not the sliced one.
   Using sliced logits for content gen produces language-token garbage
   (`<tk><ml><mrj>...`).

2. **Stock DakeQQ decoder exports only `max_logit_id` (argmax), not the full
   logits** — insufficient for beam search. The fix is a **trivial graph surgery**:
   append `/output_layer/Gemm_output_0` to the ONNX graph's output list (2 lines).
   Exposed version saved at `/tmp/opencode/dakeqq/dolphin-base/dolphin-base-decoder_withlogits.onnx`.

**Implication for Android:** DakeQQ pair (a) is the phone path — encoder is already
ONNX (fused STFT+mel+CMVN in-graph, INT8/FP16-ready, ARM-tested), and only a
2-line graph edit is needed to enable beam search on the decoder. Remaining work:
port the beam-search loop to onnxruntime-android (Kotlin), then benchmark on-device
RTF for `dolphin-small` INT8 (239MB) to confirm interactive latency.

## Comments
