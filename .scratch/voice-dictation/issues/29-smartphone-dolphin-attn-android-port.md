# 29: Smartphone ONNX pipeline for Dolphin lang/region — Android port (DakeQQ pair)

**What to build:** The Android port of ticket 28's verified laptop pipeline. The phone
will run the **DakeQQ matched ONNX encoder+decoder pair** (`onnx-community/dataocean-dolphin-asr`)
through **onnxruntime-android**, decoding with attention beam search so an explicit
`lang`/`region` pin (ur/PK) is honored — the thing Dolphin CTC (via sherpa-onnx) cannot
do. This ticket covers: pre-baking the graph-surgeried decoder on the laptop, publishing
the artifacts, adding the onnxruntime-android dependency, a new Kotlin ORT engine with the
beam-search decode loop, catalog/downloader wiring, and on-device RTF benchmark.

**Blocked by:** 28 (laptop validation of Dolphin lang/region; DakeQQ pair chosen as the
phone path). Not blocked by 26 (upstream sherpa #3904) — this is the standalone workaround
that does not wait for upstream.

**Status:** ready-for-agent

## Context (from ticket 28, verified 2026-09-03)

- DakeQQ's `Dolphin_Encoder.onnx` + `Dolphin_decoder.onnx` are a **matched pair**: the
  encoder is fused (STFT+mel+CMVN in-graph, accepts raw int16 `[1,1,L]`), the decoder is
  an incremental self-KV-cache decoder. No cross-mixing with ticket 28's own `decoder.onnx`.
- **Stock DakeQQ decoder only outputs `max_logit_id` (argmax)** — insufficient for beam
  search. The 2-line graph surgery (`add_logits_output.py`) appends the **full logits**
  (`/output_layer/Gemm_output_0`) so beam search can read the real distribution.
- Decoding **must be attention beam search over the FULL logits**, NOT the language-sliced
  logits (`/Slice_3_output_0`, used only for LID) and NOT `attention_rescoring`/greedy —
  both flip pinned Urdu to Devanagari.
- Language pin = prepend `<sos> <ur> <PK> <asr> <notimestamp>` and pass
  `language_start=136`/`language_end=269` to the decoder.

## Phase A — Laptop: generate + publish pre-baked artifacts

**Target variants: base and small** (user decision), each built for on-device use.
- [x] Run the graph surgery on the stock decoders to produce
      `<variant>-decoder_withlogits.onnx`:
      - `base`: reuses the `dolphin-base-decoder_withlogits.onnx` from ticket 28's spike
        (now at `~/.cache/dolphin/dakeqq/dolphin-base/`).
      - `small`: ran `add_logits_output.py` → `dolphin-small-decoder_withlogits.onnx`
        (2026-09-04, now at `~/.cache/dolphin/dakeqq/dolphin-small/`).
- [x] Verify each with `verify_dakeqq_beam.py` (FULL-logits path) → clean Urdu
      `آپ کا کیا حال ہے` for BOTH base and small on rec.wav; live5/live6 also Urdu for
      both. LANG-SLICED path correctly yields language-token garbage (confirms the
      full-logits requirement). Results recorded below under Phase D / Verification.
- [x] Publish encoder + decoder_withlogits + vocab to the project's HuggingFace repo
      `femustafa/voicedictation-models` under `dolphin-attn/<variant>/` (same host the
      Roman-Urdu GGML models already use).
      **DONE (2026-09-04)**: commit `a6f29c30`. Published layout:
      ```
      dolphin-attn/
        units.txt, bpe.model            (shared vocab; bpe.model per Phase B "what works")
        base/{encoder.onnx, decoder.onnx}     (fp16 arm, decoder + full-logits)
        small/{encoder.onnx, decoder.onnx}
      ```
      Base decoder round-trip download verified byte-identical (126,272,330 B).

### Precision-tier decision (2026-09-04) — recommend **fp16/arm**

DakeQQ's HF repo offers three tiers. Evaluated for the beam-search surgery requirement:

| tier | decoder file | surgurable? | decoder size (base / small) | verdict |
|------|--------------|-------------|------------------------------|---------|
| fp32 | `.onnx` | ✅ | 252 / 643 MB | works, large |
| **fp16/arm** | `.onnx` | ✅ | 126 / 322 MB | **recommended** (verified) |
| int8 | `.ort` | ❌ | — / ~239 MB (ticket 28) | **disqualified** |

- **int8 is `.ort`** (onnxruntime-optimized flatbuffer), NOT `.onnx`. Our graph surgery
  (`onnx.load` on the ONNX proto) cannot modify `.ort`, so the full-logits output beam
  search requires is unreachable without extra ONNX→ORT reconversion tooling we don't have.
  The ticket-28 "~239MB int8" benchmark target is therefore **not viable** with this approach.
- **fp16/arm `.onnx` is verified working**: I downloaded both base and small fp16-arm
  decoders, ran `add_logits_output.py`, and beam-searched them — both give
  `آپ کا کیا حال ہے` on rec.wav. fp16 is roughly half the fp32 size and is the
  on-device-sized option (and the ONNX `<->` ORT support surface confirms onnxruntime-android
  loads `.onnx`; `.ort` also loads but isn't usable here).
- **fp16 requires dtype-aware beam tensors** (KV cache in float16; `verify_dakeqq_beam.py`
  now derives the cache dtype from the encoder output). The Android engine must do the same.

**Recommendation: publish fp16/arm for both base and small** (fp32 kept as the verified
fallback if fp16 accuracy matters more than size later). RTF benchmark (Phase D) will pick
between base and small.

### File relocation (2026-09-04)

All DakeQQ working files moved from `/tmp/opencode/dakeqq/` (a 3.9 GB tmpfs that ran out
of space) to **`~/.cache/dolphin/dakeqq/`** on the home disk. Verify script default
`--dakeqq-dir` should point there. Artifacts now at:

- `~/.cache/dolphin/dakeqq/dolphin-base/`   (fp32 base pair, surgured)
- `~/.cache/dolphin/dakeqq/dolphin-small/`  (fp32 small pair, surgured)
- `~/.cache/dolphin/dakeqq/base-fp16-arm/`  (fp16 arm pair, surgured)
- `~/.cache/dolphin/dakeqq/small-fp16-arm/` (fp16 arm pair, surgured)

### Phase A findings (2026-09-04)

- **small ≠ base in architecture**: small has **12 decoder/encoder layers and 12 attention
  heads** (vs base's 6 layers / 8 heads). `verify_dakeqq_beam.py` had `NL=6` and head-dim 8
  hardcoded; made both **derived at runtime** (layer count from `en_key_*` outputs; head/dim
  from the encoder output shape) so the same script handles both models.
- **Beam search needs length-normalized scoring** (a real, reusable finding): small's decoder
  gives `<eos>` an abnormally high probability right after the prefix, so the empty
  (prefix-only) hypothesis and short one-token-truncations **out-score the full sentence**
  under plain cumulative-logprob beam selection — beam search returned empty output while
  greedy was correct. Fix in `beam_search`:
  1. separate completed (EOS) hypotheses from active beams, and
  2. select by **length-normalized score** (cumulative ÷ #content tokens) among
     content-bearing hypotheses (drop prefix-only empties).
  With this, both base AND small beam-decode to correct Urdu. **The Android port MUST use
  the same beam logic** — see Phase B requirement.

## Phase B — Android: dependency + engine

- [ ] Add **onnxruntime-android** to `android/app/build.gradle.kts`. NOTE the app today has
      NO direct onnxruntime dependency — only sherpa-onnx's bundled native lib. The DakeQQ
      pair runs through ORT directly, so this AAR is required (this is the whole reason the
      new engine is separate from `DolphinCtcEngine`).
- [ ] New `DolphinAttnEngine.kt` implementing `WhisperEngine`:
      - **load**: open ORT sessions for encoder + decoder; load vocab.
      - **transcribe**: encoder (raw int16 in-graph) → prefill `<sos><ur><PK><asr><nots>`
        → KV-cached attention beam search over the FULL-logits output → detokenize. Port
        the Python logic from `verify_dakeqq_beam.py`/`infer_onnx.py`.
      - **beam search MUST replicate the length-normalized selection** (Phase A finding):
        derive layer count + head/dim from the model (NOT hardcode), keep completed EOS
        hypotheses separate from active beams, drop prefix-only empties, and pick by
        length-normalized score. Otherwise small returns an empty/truncated transcript.
      - **language handling**: unlike `DolphinCtcEngine`, this engine **honors** the passed
        `language`/`languageMode` — it uses them for the ur/PK pin.
- [ ] Decide vocab source on-device. The DakeQQ HF repo ships only `tokens.txt`; the
      verified laptop path uses `bpe.model` + `sentencepiece.DecodePieces`. Pick what works:
      either publish/ship `bpe.model` too, or switch to sherpa-style token decode from
      `tokens.txt` only. Record which.
      **Note (2026-09-04)**: published BOTH `units.txt` (tokens list) and `bpe.model` in
      `dolphin-attn/`, so either decode route is possible from the artifacts already up.

## Phase C — Wire into app

- [ ] `ModelCatalog.kt`: add a `dolphinAttnModels` list (both sizes) with URLs to
      `femustafa/voicedictation-models/dolphin-attn/`; add `isDolphinAttn(entry)` /
      `isDolphinAttnFileName(...)` helpers; include them in the ONNX tab
      (`ModelPickerViewModel.catalogFor`).
- [ ] `ModelDownloader.kt`: new download path fetching the 3–4 sibling files
      (encoder, decoder_withlogits, tokens, [bpe]) with per-file min-size guards and a
      `dolphinAttnComplete()` check.
- [ ] `DualFormatWhisperEngine.kt`: route `dolphinAttnModels` entries to `DolphinAttnEngine`
      via a format discriminator, mirroring the Dolphin CTC routing.

## Phase D — Verify + decide integration posture

- [ ] On-device RTF benchmark for the shipped variant(s) — must be < ~1.0 for interactive
      dictation; pick base vs small / precision tier accordingly.
- [ ] Record the Android integration decision (ticket 28 open item): standalone ORT engine
      (this ticket) vs contribute decoder upstream to sherpa-onnx (ticket 26). Note
      trade-offs, recommend one.
- [ ] `lang` field in the app result JSON is populated from the pinned/decoded language.

## Confirmed verification (laptop, 2026-09-04) — surgured pair, FULL-logits beam

FULL-logits attention beam search on the **graph-surgeried** decoder (both variants):

| model | clip | FULL-logits beam result |
|-------|------|-------------------------|
| base  | rec.wav  | آپ کا کیا حال ہے |
| base  | live5.wav | میں تھوڑی گیر میں آفس آوں گا اسے پہلے میں ناشتہ کروں گا اور ناشتے میں چائے کیوں گا پھر اس کے بعد میں آفس آوں گا |
| base  | live6.wav | تھوڑی دیر میں آفس چونگا تھوڑی دیر میں آفس چونگا تھوڑی دیر میں آفس چونگا |
| small | rec.wav  | آپ کا کیا حال ہے |
| small | live5.wav | اُسے پہلے میں ناشتہ کروں گا اور ناشتے میں چائے پیوں گا پھر اُس کے بعد میں آفس جاوں گا |
| small | live6.wav | تھوڑی دیر میں آفس جاوں گا |

LANG-SLICED path on rec.wav yields language-token garbage (`<tk><ml><mrj>…` / `<ab><ky><nl>…`)
for both variants — reconfirms the decoder MUST use the FULL-logits output for content.

## Notes
- App model files live in app-private `filesDir` (scoped storage); downloads already stream
  to `.part` and rename. Extend that pattern rather than introducing a new one.
- onnxruntime-android AAR adds native size similar to the existing sherpa-onnx lib; include
  in the app-size picture (ticket 25).

## Comments
