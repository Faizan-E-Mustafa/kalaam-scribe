# 03: Live (streaming) merging in LiveVadDrainer

**What to build:** A merge mode for the live streaming path: when VAD closes an
utterance, hold it back instead of publishing immediately, and if another
utterance starts within a settle window, append it — publishing one merged chunk
(spanned with real source audio, up to `MERGE_MAX_SAMPLES`) instead of several
short per-utterance decodes. See spec: `/home/femustafa/projects/learning_ws/.scratch/vad-accuracy/spec.md`
(Decision 2, "live streaming ... left as a follow-up tuning item").

**Blocked by:** nothing (reuses the `LiveVadDrainer` rolling `source` buffer from
ticket 01 and the source-backed span idea from ticket 02's `mergeSegments`).

**Status:** ready-for-human

## Decisions

- **Where:** a merge mode *inside* `LiveVadDrainer` (opt-in via a ctor param, e.g.
  `merge: Boolean = false` or `mergeHoldMs: Int = 0`), so all three streaming
  backends get it from the shared seam. When off, the drainer behaves exactly as
  today (padding + per-utterance publish).
- **Settle window:** when the drainer holds a closed utterance, it waits for new
  speech for a short window (e.g. ~600–900 ms — long enough to span a natural
  sentence pause, short enough to bound the extra latency on single-phrase
  dictations) before emitting the accumulated chunk. Configurable; a hard cap
  (`MERGE_MAX_SAMPLES`) force-emits mid-settle.
- **Source-backed span:** the drainer already keeps `source` with absolute
  indexing; closed segments are merged against it into `source[start .. lastEnd)`
  (gap audio included), then padded once at the merged edges — same composition as
  ticket 02's batch path.
- **Padding composes post-merge:** a merged chunk is padded once; padding never
  counted against the merge cap; individual held segments are not padded on the way
  in.
- **flush():** the held (un-emitted) chunk is the tail — `flushAndDrain()` must
  emit it (merged + padded) so the worker decodes it before the join, mirroring
  today's tail handling.
- **UX:** because merging delays/shrinks partials, this is a **toggle** ("merge
  phrases" in streaming), not the default. Reuses the BatchVad constants
  (`MERGE_MAX_MS`, `PAD_SAMPLES`).

## Acceptance Criteria

- [x] With merging on, the drainer holds a closed utterance and merges the next
      utterance if it starts within the settle window; emitted chunks are
      source-backed spans (gap included), padded once at the edges, ordered.
- [x] The merge cap force-emits; a single long utterance passes through whole.
- [x] `flushAndDrain()` emits the held chunk so no audio is dropped.
- [x] Merge off ⇒ identical behavior to today (existing `LiveVadDrainer` +
      `StreamingSessionNoDrop` + session tests unchanged).
- [x] JVM tests (fake `VadLike`): hold-and-merge across the settle window,
      force-emit at cap, flush tail, merge-off pass-through.
- [x] `./gradlew :app:testDebugUnitTest` green (97 tests); `:app:assembleDebug`
      builds; `lintDebug` adds no new errors.
- [ ] On-device (the phone): dictating with streaming + merge on shows fewer,
      longer partials and no fewer edge words. **(pending — human step)**

## Comments

- **2026-09-15**: Created as the paper map; ticket 02 (batch merge + Batch+VAD
  mode) is where the shared `mergeSegments`/span primitive was landed and proven.
- **2026-09-15**: Implemented. `LiveVadDrainer` got `mergeSettleWindows`
  (consecutive silent pushes; constant `MERGE_SETTLE_WINDOWS` = 900 ms / 32 ms =
  28) and `maxChunkSamples` (= `MERGE_MAX_SAMPLES`). Release builds one
  source-backed `[pending.first.start .. last.end]` chunk padded once; `pad()` is
  called before the post-emit prune so its pre/post margins read the live source
  window. Settlement is push-count based (deterministic, no wall clock): a speech
  window resets the counter, and `silentWindows >= settleWindows` releases the
  held chunk. Cap force-emits during `drainQueue`; `flushAndDrain()` drains the
  VAD tail then releases whatever is held.
  Wired into the production streaming sessions via a new `mergeSettleWindows`
  ctor param (default 0) on `SherpaOfflineSession`/`DolphinAttnSession`; the four
  engines pass `VoiceDictationApp.vadMergeSettleWindows()` (0 unless enabled).
  `VoiceDictationApp` gained `streamingMerge()`/`setStreamingMerge()`, and the
  settings sheet a "Merge phrases" toggle shown only in SimulatedStreaming.
  New tests `LiveMergingTest` (6): merge across settle, lonely-phrase settle
  emit, flush tail, cap force-emit, merge-off pass-through, pre-pad survival
  across an earlier emit. All 97 tests green; assemble + lint clean.
  Remaining: on-device A/B of streaming with/without merge on.