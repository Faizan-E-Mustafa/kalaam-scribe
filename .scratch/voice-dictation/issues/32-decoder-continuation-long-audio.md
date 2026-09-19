# 32: Decoder continuation for long Dolphin audio (encode-once, KV-windowed decode)

**What to build:** Replace the audio-side chunking (split into ≤14 s pieces, decode each
independently, join) with **decoder continuation**: encode the *whole* clip once, then
decode in windows bounded only by the decoder's KV-history kernel wall, carrying the last
generated token forward into a fresh decoder KV window that still cross-attends to the
**full** encoder memory. The decode continues seamlessly past the 66-token wall instead of
starting each piece cold.

**Why:** The fused `SkipLayerNormalization` kernel rejects a decoder KV history beyond 72
tokens (66 content tokens after the 5-token prefix) with `ORT_INVALID_ARGUMENT`
(`SkipLayerNorm_0`); verified identical on ORT 1.27 (device) and 1.30 (laptop) for base and
small. The landed mitigation (`629da83`, `81ba7aa`) chunks audio into ≤14 s pieces
preferring quiet pause boundaries — it fixed truncation/crash but keeps two artifacts: (a)
word *repeats/garbles at piece joins* (the model resumes cold at a mid-sentence boundary),
and (b) per-chunk decode cost (a 41.7 s clip re-encodes per chunk; ~40 s wall). Continuation
targets the root cause: the constraint is on *decoded tokens*, not *audio length* — so
window on the decoder axis and keep the whole recording in cross-attention.

**Blocked by:** none (follows 29/31; batches and the streaming session both already decode
through `beamSearch`, which is the single seam to change).

**Status:** won'tfix

## Background established (2026-09-19, verified)

- Encoder has **no** length limit (probe: laptop ORT, base + small). Only the **decoder
  self-attention KV** (history = generated token count) hits the 72 wall. Cross-attention to
  encoder memory is a separate, unbounded axis.
- `beamSearch` (`DolphinAttnEngine.kt`) is called by both the batch path (per chunk) and the
  streaming session (`DolphinAttnSession`), and already implements `runDecoder(ref, enKeys,
  enValues, inputIds, historyLen, deKeys, deValues)` with reusable encoder cross-KV — the
  exact primitives a continuation pass needs.
- Real on-device Urdu runs ~2-4.1 tok/s (dense speaker ~4.1 → 16 s ≈ 66 tokens, zero
  margin). A 14 s cap ≈ 57 tokens leaves ~9 tokens headroom.
- On-device throughput is ~1× realtime (41.7 s clip ≈ 40 s transcribe), so true
  token-synchronous streaming is out of scope; continuation is a *batch-quality* improvement,
  not a streaming one.

## Proposed mechanics (to validate in prototype/task)

1. Encode the whole clip: `encoder.run(audio)` → `enKeys`/`enValues` (kept open, as today).
2. Decode greedily/beam until a window is about to exceed the wall (history_len → 72).
3. Reset decoder KV to empty, seed the new window with the **last generated token** (option:
   last K tokens) as the only self-attention context, **reuse the same `enKeys`/`enValues`**,
   and continue from there.
4. Join windows trivially (tokens are already one stream); EOS anywhere ends the clip.

Open questions to resolve before/during build:
- Does seeding a fresh window with just the last token (as whisper.cpp long-audio does) keep
  quality on these Dolphin models? Prototype on laptop with base + small.
- Beam-vs-greedy state across windows: current code runs a length-normalized beam. Decide
  whether continuation windows continue the beam or reset to a fixed seed beam.
- Interactions with `tokenBudget` (keep as a degenerate-loop guard), `maxH` log line, and
  `decodeTokens`.

## Prototype verdict (2026-09-19; laptop ORT 1.30 fp16 base, real Urdu clips)

Decode continuation is **not viable on Dolphin** in this app's constraints. Every
form of KV-windowed continuation collapses; the decoder is a strict single-pass
transcriber. Verified negative across four decompositions (greedy argmax, CPU):

1. **Encode-once + decoder-KV reset + re-seed** (the ticket's proposed mechanics;
   seed = last 1/4/8/16 generated tokens, same full-audio `enKeys/enValues`):
   window 1 reads flawlessly (66 tokens, verbatim match with the on-device chunk
   transcript), but **every window 2+ degrades to a single repeating token** — with
   17.8 s→only-window-left, and also with 22 s / full 41.7 s clips where ~25 s of
   untranscribed speech still awaited it. `windows=[66,71,63]` etc.; loops like
   `ٹ`, `ائم`, `اس`, never EOS. Seed length irrelevant (1/4/8/16 identical).
2. **Chunked encoder + carried prior text (without prefix)** in decoder KV: also
   collapses by window 2 (`ens`, `a`, `ar`, `at`, `ho at` loops).
3. **whisper.cpp-style prompt-conditioning** — `PREFIX + rolled prior` in the KV
   seed, fresh 9-11 s encoder per window: collapses identically (`ہاں` loop) from
   window 2 on.
4. **Clean-foreign-prior test** (prior = the *perfect* complete-sentence transcript
   of another clip): decoder emits **EOS on the very first token** — it refuses to
   transcribe new audio at all once prior text is in context.

Root cause: this vocabulary has **no timestamp / `<prev>`-style tokens**
(sherpa-onnx Dolphin-attn is a speech-only seq2seq; units contains only
`<sos>/<eos>/<asr>/<ur>/<PK>/<notimestamp>`). whisper.cpp long-audio works because
Whisper's decoder is *trained* to prompt-condition on prior text and word
timestamps; this decoder never saw prior-text context, so it either loops (weak
prior) or EOSes (strong prior). Carrying more is also structurally impossible:
even the best case leaves ≤66 content tokens of context, and 8 s chunks measurably
hurt cold-decode quality vs. 22 s single-context reads.

**Implication:** the prefer-silence, VAD/pause-aligned audio-side splitter
(`629da83`/`81ba7aa`) is the *only* long-audio mechanism available here; ticket 32's
"no repeated word at the ~66-token boundary" criterion (AC-1) cannot be met by any
decoder-axis trick. Residual join artifacts are the intrinsic cost of audio-side
chunking. Recommend disposition: **won't-fix as decoder continuation**; keep the
splitter (possibly tune `DOLPHIN_MAX_MS` / `QUIET_RMS`), don't shrink chunks (hurts
quality).

Prototype scratch: `/tmp/opencode/proto_continuation{,[234]}.py` (throwaway, not
committed).

## Acceptance Criteria
- [ ] A >14 s unbroken utterance yields a transcript with **no repeated/filler words at the
      ~66-token boundary** (compare 41.7 s clip: today "...نہیں جانا اس لئے نہیں جانا").
- [ ] 17.8 s / 41.7 s / 84.6 s corpus keeps complete tails; zero `SkipLayerNorm_0` errors;
      every decoder window `maxH ≤ 72`.
- [ ] Same improvement applies to the streaming session (`DolphinAttnSession`) segments
      that currently get chopped.
- [ ] Wall-clock for a 41.7 s clip ≤ today (~40 s), ideally lower (one encoder pass).
- [ ] JVM tests + on-device verification (Load file: natural41, short17, double; streaming
      mode long recording).

## Labels
dolphin-attn, kernel-wall, long-audio, decode-continuation, quality