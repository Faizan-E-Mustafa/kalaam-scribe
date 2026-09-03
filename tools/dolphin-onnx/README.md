# Dolphin ONNX Export & Verification (Ticket 28 — Urdu Dictation)

## Summary

This repo houses the scripts, scripts, and documentation for the ONNX-based
Urdu dictation pipeline. The phone target is **DakeQQ's Dolphin encoder+decoder
matched pair** (a standalone ONNX export of Dolphin's attention-decoder pipeline),
supplemented by the `infer_onnx.py` torch-onnx hybrid pipeline as a reference
and fallback.

## How it works

- **Encoder**: The Dolphin EBranchformer encoder with STFT+mel+global_cmvn baked
  in. It accepts raw 16 kHz `int16` audio `[1,1,L]` and produces 6 cross-KV
  tensors `[8,64,T]` per layer, plus after_norm output `[B,T,512]`.
- **Decoder**: A standard attention-decoder ONNX model that consumes:
  - `memory` `[B,T,512]` encoder output
  - `memory_mask` `[B,T]` boolean padding mask
  - `tgt` `[B,L]` target token ids
  - outputs `logits` `[B,L,vocab]` (log-softmax over the full 40002-vocab)
- **Decoding**: Plain `attention` beam search over the full logits.
  This is the only method that correctly respects a pinned `ur/PK` language
  region token. Other decoding methods (`attention_rescoring`, greedy) override
  the pin and flip output to Devanagari/garbage.

## The "double-CMVN" bug (root cause of garbled output)

Dolphin's encoder applies `global_cmvn` internally (`model.py:1743`). The
pipeline **must feed RAW log-mel features** — **no manual mean/std normalization**.

- **Wrong**: normalize features yourself → output becomes garbled/Devanagari.
- **Correct**: feed log-mel (as produced by the code in `infer_onnx.py` stft/mel
  block). The encoder's internal CMVN handles normalization.

This was discovered during Ticket 27 where attention vs rescoring decode was
investigated; removing manual CMVN fixed all garbled outputs.

## Two decisive DakeQQ findings (Sep 3 2026 spike)

### 1. `language_start`/`language_end` inputs slice the logits for LID only

The DakeQQ decoder accepts `language_start` and `language_end` int64 inputs
(ids 136 and 269 = `<ur>` and `<PK>`). These are used by the `/Slice_3`
node to restrict logits to the language-token range before `ArgMax`.

- **For LID / auto-detection**: these inputs correctly steer decoding.
- **For content generation**: you **must read the full logits** (the
  `/output_layer/Gemm_output_0` output). Using the sliced logits → the beam
  search produces language-token garbage like `<tk><ml><mrj>...`.

### 2. Stock DakeQQ decoder only exports `max_logit_id` (argmax)

The exported `decoder.onnx` from DakeQQ only has `max_logit_id` as output.
That's sufficient for greedy decoding, **but insufficient for beam search**,
which needs the full probability distribution over the vocabulary.

**Fix**: a **trivial 2-line graph surgery** appends two tensors to the output
list:

- `/output_layer/Gemm_output_0` — full logits `[B, seq, 40002]` (the raw
  distribution before argmax). Pass this to beam search.
- `/Slice_3_output_0` — the language-sliced logits (for LID inspection).

**Script**: `tools/dolphin-onnx/scripts/add_logits_output.py` performs this
surgery. After the edit, the model exposes both tensors and works correctly
with onnxruntime.

The regenerated file: `decoder_withlogits.onnx` (≈241MB, same as original spike).

## File layout (repo, tools/dolphin-onnx/)

```
tools/dolphin-onnx/
  README.md                this file
  .gitignore              ignore *.onnx, etc.
  scripts/
    export_decoder.py      export Dolphin's attention decoder to ONNX
                         (torch.jit.trace; produces decoder.onnx)
    infer_onnx.py          standalone torch-encoder + onnx decoder + beam search
                         (the working laptop pipeline; CLI: uv run infer_onnx.py [wav])
    add_logits_output.py   graph-surgery script: append full+sliced logits
                         to DakeQQ decoder.onnx -> decoder_withlogits.onnx
    verify_dakeqq_beam.py    verification spike (DakeQQ encoder+decoder + beam)
                         CLI: python verify_dakeqq_beam.py clip.wav
    export_dolphin_onnx.py original bulk export script (encoder+decoder, reference)
    dolphin_lang_test.py   ticket 27 validation (attention vs rescoring; pinned ur/PK)
    debug_decoder.py       diagnostic script (old)
    verify_extracted_encoder.py  diagnostic script (old)
    extract_encoder.py     diagnostic script (old)
    verify_onnx.py         diagnostic script (old)
    inspect_sherpa_model.py diagnostic script (old)
```

## Path conventions & how to reproduce

All script paths reference model files at:

- `~/.cache/dolphin/base/` — Dolphin model cache (units.txt, bpe.model, feats_stats.npz, base.pt)
- `tools/validate_stt/.venv` — Python 3.11 with dependencies (torch, onnx, onnxruntime, sentencepiece, dolphin, soundfile, numpy)
- `/home/femustafa/projects/learning_ws` — repo root, where the above `tools/dolphin-onnx/` lives.

### Step 1 — Export a decoder (once)

```sh
cd tools/dolphin-onnx/scripts
uv run export_decoder.py --model base
```

produces `artifacts/decoder.onnx` (≈275MB). The encoder cannot be exported
via torch.jit (rel_pos/cgmlp ops mis-trace); it stays in torch at runtime.

### Step 2 — (If using DakeQQ) Run graph surgery to expose full logits

```sh
uv run add_logits_output.py decoder.onnx decoder_withlogits.onnx
```

produces `decoder_withlogits.onnx` with both `/output_layer/Gemm_output_0`
and `/Slice_3_output_0` appended to the output list.

### Step 3 — Run inference

```sh
uv run infer_onnx.py rec.wav            # pinned ur/PK → Urdu text
python verify_dakeqq_beam.py rec.wav    # DakeQQ spike + beam comparison
```

### Step 4 — Android port

The **phone path** is DakeQQ's ONNX pair + the graph-surgeried decoder +
plain attention beam search. The encoder runs in-graph (fused STFT+mel+CMVN),
feeding raw int16 `[1,1,L]` audio to the ONNX session. The decoder consumes
`memory` + `memory_mask` + `tgt`; beam-search loop runs in Kotlin/ONNX
Runtime Mobile, appending `/output_layer/Gemm_output_0` full logits to the
decode distribution.

See ticket 28 (`.scratch/voice-dictation/issues/28-smartphone-dolphin-onnx.md`)
for the Android checklist and runtime decisions (ORT Mobile vs sherpa-onnx
contribution).

## Model constants (from units.txt)

- Vocabulary size: 40002
- `<sos>` = 39999, `<eos>` = 40000, `<notimestamp>` = 324
- `<ur>` = 136, `<PK>` = 269
- Task token: `<asr>` = 6
- `<blank>` = 0

The `LANG_IDS` and `REGION_IDS` dicts in `infer_onnx.py` map string codes
("ur", "PK") → token ids used as prefix inputs to the decoder.