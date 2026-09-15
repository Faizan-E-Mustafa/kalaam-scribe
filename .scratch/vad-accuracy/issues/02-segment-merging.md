# 02: Segment merging + Batch+VAD mode

**What to build:** Post-process the closed VAD segments from the whole-clip
`segmentAudioWithVad` path: merge consecutive segments into chunks while the
running duration stays under a cap (default **28 s**, matching Whisper/Dolphin's
~30 s training context with headroom), so short utterances are not decoded alone.
Chunk boundaries therefore fall at real silence gaps (WhisperX "cut & merge").
Then wire a production **Batch + VAD** transcription mode that actually runs this:
split the recorded clip with the VAD, merge/pad, decode each chunk with the
resident engine. See spec: `/home/femustafa/projects/learning_ws/.scratch/vad-accuracy/spec.md`
(Decision 2).

**Blocked by:** nothing (reuses the `SpeechSegment`/`VadSegmentation` seam from
tickets 30/31 and the `padSegments`/`segmentAudioWithVad` plumbing from ticket 01).

**Status:** ready-for-human (implementation + JVM tests done; needs an on-device
A/B of Batch vs Batch+VAD — new mode in settings → "Batch + VAD")

## Decisions

- **Merge cap:** a merged chunk's total span must stay under **28 s = 448,000
  samples at 16 kHz** (`MERGE_MAX_MS = 28_000`, `MERGE_MAX_SAMPLES`), leaving
  headroom under Whisper's 30 s design limit (spec Decision 2). The span includes
  the inter-segment silence gaps (the chunk is one continuous region of source
  audio from the first segment's start to the last segment's end).
- **Pure batch pass:** merging is a pure function over the drained raw segments +
  the source buffer: `mergeSegments(segments, source, maxSamples)`. It runs in
  `segmentAudioWithVad` between the drain and the (optional) padding pass, so the
  pipeline is **drain raw -> merge -> pad**. Opt-in via a new
  `maxChunkSamples: Int = 0` param (0 = no merging), mirroring how `padding`
  stays opt-in with a safe default.
- **No splitting:** a segment that alone exceeds the cap is emitted as-is,
  never split. No backward merge: greedy forward accumulation.
- **Ordering preserved:** merged chunks keep the source order and carry `start` =
  index of the chunk's first sample in `source`; `samples` holds the full span
  `source[start .. lastEnd)`, including the silence between constituent segments.
- **Padding composes post-merge:** a merged chunk gets a single pre/post pad at
  its (merged) edges via the existing `padSegments`; no double-padding inside the
  chunk and no padding counted against the merge cap.
- **Production mode:** new `TranscriptionMode.BatchVad` ("Batch + VAD"). Batch and
  BatchVad never open a live [TranscriptionSession]; on Stop, `DictationService`
  (dictation) and `DictationViewModel` (imported files) route the clip through
  `WhisperManager.transcribeVad` instead of `transcribe`. `transcribeVad` reads the
  clip to 16 kHz mono floats, runs `segmentAudioWithVad(padding = PAD_SAMPLES,
  maxChunkSamples = MERGE_MAX_SAMPLES)`, writes each chunk to a temp WAV, decodes
  it with the resident engine (so every backend — including GGML's file-only AAR —
  is supported unchanged), and joins the non-blank texts with spaces.
- **Fallbacks:** when the Silero VAD model is missing (`vadFactory == null`),
  the clip cannot be read, or no speech is found, `transcribeVad` falls back to the
  plain whole-clip `transcribe` — Batch + VAD can never be *worse* than Batch.
- **VAD instance:** built by a `vadFactory` injected into `WhisperManager`
  (constructed once per BatchVad call; mirrors the streaming sessions' config from
  `VoiceDictationApp` settings), so the manager itself stays native-free/JVM-testable.
- **Live path out of scope:** `LiveVadDrainer` keeps per-utterance publish —
  merging there would add decode latency and change the streaming contract; it is
  a separate ticket (03).
- **`SpeechSegment` semantics unchanged:** `start`/`samples` still describe the
  region handed to decode; callers treat it as "the audio to decode".

## Acceptance Criteria

- [x] `mergeSegments(segments, source, maxSamples)` merges consecutive segments
      into continuous source-backed spans while the running span stays under
      `maxSamples`, clamping the slice to the source.
- [x] Chunk boundaries fall at real silence gaps; oversized single segments pass
      through whole; greedy forward (no backward merge).
- [x] A merged chunk's samples are source audio from `start` to the last
      constituent segment's end (intermediate silence included), with correct
      `start` index.
- [x] Merging composes with padding: `segmentAudioWithVad(..., padding, maxChunkSamples)`
      merges first, then pads each merged chunk at its outer edges; no
      double-padding, padding not counted in the cap.
- [x] Default behavior unchanged: `maxChunkSamples = 0` reproduces today's output
      exactly; `LiveVadDrainer` untouched.
- [x] `TranscriptionMode.BatchVad` is offered in settings (third radio) and routes
      dictation + imported files through `WhisperManager.transcribeVad`; falls back
      to whole-clip `transcribe` when the VAD is missing, audio is unreadable, or
      no speech is found.
- [x] JVM unit tests: merge math (cap boundary, over-cap single segment, gap
      inclusion, merged-span clamp), merge+padding composition, no-merge
      pass-through (`SegmentMergingTest`); `transcribeVad` chunking/join + both
      fallbacks (`WhisperManagerTest`). Existing suites stay green.
- [x] `./gradlew :app:testDebugUnitTest` green (**90 tests**); `:app:assembleDebug`
      builds; `lintDebug` adds no new errors.
- [ ] On-device (the phone): settings → "Batch + VAD" vs "Batch" A/B on a long
      clip / imported file shows the merge/pad longer-context decode is not worse
      and helps edge words. **(pending — human step)**

## Implementation outline

1. **`VadSegmentation.kt`**: `MERGE_MAX_MS`/`MERGE_MAX_SAMPLES` constants; pure
   `mergeSegments(segments, source, maxSamples)`; `segmentAudioWithVad` gained
   `maxChunkSamples: Int = 0` (drain -> merge -> pad).
2. **`Model.kt`**: `TranscriptionMode.BatchVad`.
3. **`AudioDecoder.kt`**: exposed `decodeToMono16kSamples(path)` and
   `writePcm16Wav(file, samples, rate)` (deduped the old private writer).
4. **`WhisperManager.kt`**: optional `vadFactory` ctor param (internal ctor, since
   `VadLike` is internal); `transcribeVad(path)` = read->segment->merge->pad->
   decode chunks (temp WAVs)->join, with whole-clip fallbacks.
5. **`VoiceDictationApp.kt`**: `sherpaVadOrNull()` VAD factory (shares the
   streaming VAD settings).
6. **`DictationService.kt`** / **`DictationViewModel.kt`**: route the clip through
   `transcribeVad` in BatchVad mode.
7. **`AdvancedSettingsSheet.kt`**: third "Batch + VAD" radio; VAD settings dim only
   in plain Batch.
8. **Tests**: `SegmentMergingTest` (6 merge cases + 2 integration);
   `WhisperManagerTest` (chunked decode/join + 2 fallbacks).