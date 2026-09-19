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

**Status:** needs-triage

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