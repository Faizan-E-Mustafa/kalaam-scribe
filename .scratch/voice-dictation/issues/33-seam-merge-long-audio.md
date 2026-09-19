# 33: Seam word-merge for long Dolphin audio

**What to build:** Pure text post-processing that removes chunk/segment seam
word-duplication via word-level longest-common-substring, wired into both batch
`transcribe()` and `DolphinAttnSession` flush joins. Two strings `a` and `b` merged:
the tail of `a`'s last 12 words and the head of `b`'s first 12 words are checked for
a contiguous common word-run; if ≥4 matched words, the matched run is dropped from `b`
(left wins), otherwise both are kept unchanged.

**Why:** The chunking guards (`629da83`, `81ba7aa`) fix truncation/crash but each
chunk ends cold at a mid-sentence boundary → the first N words of the next chunk re-decode
as a repeat of the chunk's tail. This artifact duplicates words across every seam (the
device test showed "بہت عرصہ ہوا آپ سے کوئی **عرصہ ہوا آپ سے کوئی** بات نہیں ہوئی...").
The encode-once continuation fix (ticket 32) is **won'tfix** (proven infeasible — every KV
window continuation collapses). The industry-pattern fix (overlap+LCS merge) is applied
as pure text post-processing with no model/KV changes, so it works today.

**Blocked by:** none (depends on prior fixes `629da83`, `81ba7aa`; ticket 32 verdict settled).

## Background established (2026-09-19, verified)

- **Kernel wall**: fused decoder `SkipLayerNormalization` fails at history_len=72 → 66
  content tokens after 5-token prefix `ur/PK`. Only decoder KV count matters; encoder has no
  length limit.
- **Batch path** (`transcribe`, `DolphinAttnEngine.kt:251-295`): clips >14 s are split into
  ≤14 s chunks via VAD, decoded independently, then `joinToString(" ")` concatenates. The
  seam at every chunk boundary repeats the prior chunk's tail words.
- **Streaming path** (`DolphinAttnSession`, `decodeSegment`, `flush`): simulated streaming
  splits over-cap segments into ≤14 s pieces (preferring quiet pauses) and joins pieces.
  The guard fires "streaming segment 14.3s -> 2 chunks" — the seam also duplicates words
  across the boundary. Two live verification rounds on-device confirmed the guard prevents
  crash but the seam duplication artifact remains visible to the user.
- **On-device streaming verification**: round 1 single-window clean; round 2 guard fires,
  seam duplicated "...بہت عرصہ ہوا آپ سے کوئی **عرصہ ہوا آپ سے کوئی** Bart" (mid-clause
  split). The guard code (`81ba7aa`) is landed and verified.
- **Web research**: industry (FluidAudio LongTranscription; faster-whisper/Whisper) fixes
  seams by **overlapping windows slightly and merging transcripts** — LCS word matching,
  splice on word boundary, rule "may produce a glued word but never delete real content".
  This project applies the text-level merge only (no model/KV change).
- **Prototype** (`proto_seam_merge2.py`): pure text `mergeTranscripts(a,b,tail_n=12,head_n=12,min_match_words=4)` verified:
  - Device artifact (6-word dup): merged (6-word overlap collapsed) ✓
  - Genuine 3-word repeat preserved ✓
  - 1-word seam untouched ✓
  - 6-word overlap collapsed ✓
- **Seam-merge design decisions** (based on prototype):
  - Word-level longest common *contiguous* substring via `SequenceMatcher` (difflib)
  - Tail = last 12 words of `a`, head = first 12 words of `b`
  - Require ≥4 matched words to trigger collapse
  - When collapsed: left's copy kept, matched run dropped from right
  - Trade-off: a genuine repeated ≥4-word phrase exactly at a seam would be mis-collapsed
    (acceptable given these are mid-sentence repeats from cold restarts)
  - No audio overlap needed to ship (device artifact occurred with no overlap); add ~1 s overlap
    only if seams still leak after merge
  - Minimal windows (12 words) keep runtime O(1) per merge and unnoticeable on live audio
  - Batch path: sequential merge accumulator across chunks; streaming: fold across decoded
    texts at flush; per-segment piece split inside decodeSegment also merged

## Proposed mechanics

### Pure helper `seamMerge(a: String, b: String, windowWords: Int = 12, minMatchWords: Int = 4): String`

```
var bestSize = 0; var bestAi = -1; var bestBj = -1
val aw = a.trim().split(" ").filter { it.isNotEmpty() }
val bw = b.trim().split(" ").filter { it.isNotEmpty() }
val tail = aw.takeLast(windowWords)
val head = bw.take(windowWords)
for (i in tail.indices) for (j in head.indices) {
    var k = 0
    while (i+k < tail.size && j+k < head.size && tail[i+k] == head[j+k]) k++
    if (k > bestSize) { bestSize = k; bestAi = i; bestBj = j }
}
if (bestSize >= minMatchWords) {
    // keep a unchanged, drop matched run from b
    val keepB = bw.take(bestBj) + bw.drop(bestBj + bestSize)
    return a + " " + keepB.joinToString(" ")
}
return a + " " + b
```

When `bestBj == 0` (dup starts at right edge): we keep all of `a`, and `b` loses its
leading matched words. When `bestBj > 0` some of `b`'s leading non-dup words survive
before the matched run.

### Wiring points

1. **Batch `transcribe()`** — replace `chunks.asSequence().map { decodeTokens(...) }.joinToString(" ")`
   with a fold accumulator:
   ```
   var acc: String? = null
   for (chunk in chunks) {
       val t = decodeTokens(ref, beamSearch(ref, chunk)).trim()
       if (t.isNotEmpty()) acc = if (acc == null) t else seamMerge(acc!!, t)
   }
   acc ?: ""
   ```

2. **Streaming `flush()`** — replace `decodedTexts.joinToString(" ")` (line 166) with:
   ```
   decodedTexts.fold("") { acc, t -> if (acc.isEmpty()) t else seamMerge(acc, t) }
   ```

3. **Streaming `decodeSegment()`** — the intra-piece join `.map { decodeSegmentOnce(it).trim() }.filter {}.joinToString(" ")` → fold with `seamMerge` so pieces within one over-cap segment are also merged:
   ```
   var acc: String? = null
   for (piece in pieces) {
       val t = decodeSegmentOnce(piece).trim()
       if (t.isNotEmpty()) acc = if (acc == null) t else seamMerge(acc!!, t)
   }
   acc ?: ""
   ```

4. **Partial emissions** (`_partials.tryEmit(text)` at line 125) keep emitting raw per-segment text
   — live UI stays truthful; authoritative final transcript merges via `flush()`.

## Prototype verdict (2026-09-19; laptop ORT 1.30 fp16 base, real Urdu clips)

- Device artifact (6-word dup): merged (6-word overlap collapsed) ✓
- Genuine 3-word repeat preserved ✓
- 1-word seam untouched ✓
- 6-word overlap collapsed ✓
- Trade-off: a repeated ≥4-word phrase exactly at a seam would be mis-collapsed
  (rare; acceptable for this mitigation)

## Commit message on acceptance

`docs: ticket 33 verdict — seam-merge merge-text post-processing (won'tfix continuation)`.