# 01: Segment padding for VAD-derived speech segments

**What to build:** Add pre/post padding around VAD speech segment boundaries before
the segment reaches the ASR decoders, so words at the onset/offset of each
utterance are not clipped by the VAD's exact silence-boundary cuts. Applies to
the whole-clip `segmentAudioWithVad` path and the live `LiveVadDrainer` path
(shared by sherpa Whisper ONNX, Dolphin CTC, and Dolphin attention). See spec:
`/home/femustafa/projects/learning_ws/.scratch/vad-accuracy/spec.md`.

**Blocked by:** nothing (reuses the existing `VadLike` seam and
`SpeechSegment`/`VadSegmentation` conventions from tickets 30/31).

**Status:** ready-for-human (implementation + JVM tests done; final on-device the phone
verification below is a human step)

## Decisions

- **Padding width:** **200 ms each side** = **3200 samples at 16 kHz**
  (`PAD_MS = 200`, `PAD_SAMPLES = PAD_MS * SAMPLE_RATE / 1000`), a named constant
  in `VadSegmentation.kt`. Rationale: large enough to cover a word's onset/offset
  transients and typical VAD look-ahead error; small enough not to bridge distant
  utterances (VAD min-silence already gates that) or blow the 30 s decode cap.
- **Clamping:** pre-padding clamps at source start (index 0); post-padding clamps
  at source end. Padding never extends past the underlying audio buffer.
- **Where padding happens — batch (`segmentAudioWithVad`):** padding is applied as
  a final pass over the drained segments using the original full `samples` buffer
  (cheap indexed slicing, no copies), so both live and tail segments get padded
  symmetrically.
- **Where padding happens — live (`LiveVadDrainer`):** the drainer is upgraded to
  carry:
  - a rolling **history buffer** of the last `PAD_SAMPLES` samples (pre-padding),
  - a **pending future buffer** collecting samples after VAD declares speech until
    `PAD_SAMPLES` more have arrived, so the segment is emitted with post-padding
    *already available* (the speech region is bytes the VAD itself drained within
    `acceptWaveform`, so "future" here = samples already seen by us).
- **No double padding across adjacent segments:** a segment's post-pad may
  overlap the next segment's pre-pad. That is fine for accuracy (repeated audio is
  expected in overlap-style decoding) and is not merged away in this ticket.
- **`SpeechSegment` semantics unchanged:** `start`/`samples` still describe the
  padded region handed to decode; callers (sessions) keep treating it as "the
  audio to decode". No change to decode loops, KV caches, or the `ur`/`PK` pin.

## Acceptance Criteria

- [x] A segment emitted by `segmentAudioWithVad` carries `PAD_SAMPLES` extra
      samples before and after the raw VAD region, clamped at the buffer edges.
      (Opt-in via the new `padding` param; `padSegments` + test locked.)
- [x] A segment emitted by `LiveVadDrainer.push` carries the same padding using
      history + already-seen samples; a mid-buffer segment has full 200 ms/200 ms
      padding; a segment near the start flushes the pre-pad only up to what
      history holds; the tail segment after `flushAndDrain` has post-pad only up
      to what audio arrived. (`SegmentPaddingTest` 5 live cases.)
- [x] Padding is applied consistently across all three DSP backends (sherpa
      Whisper, Dolphin CTC, Dolphin attention) because it lives in the shared
      `LiveVadDrainer`/`VadSegmentation.kt` seam — engaged by `SherpaOfflineSession`,
      `DolphinAttnSession`, and `EngineTranscriptionSession` (ctor param,
      default `PAD_SAMPLES`).
- [x] JVM unit tests (fake `VadLike`, no native lib): padding math (exact sample
      counts, clamping at both edges), live-drainer history/pending behavior,
      empty-input unchanged. Existing `VadSegmentationTest` + `StreamingSessionNoDropTest`
      stay green (and `EngineTranscriptionSessionTest` — ordering tests pass
      `padding = 0` so they keep testing ordering; padding has its own suite).
- [x] `./gradlew :app:testDebugUnitTest` green (**81 tests**); `:app:assembleDebug`
      builds; `lintDebug` adds no new errors (only the pre-existing
      `AudioRecorder.kt:45` MissingPermission).
- [ ] On-device (the phone): dictating a clip with words at utterance edges shows
      fewer dropped/chopped edge words than pre-padding. **(pending — human step)**

## Implementation outline

1. **`VadSegmentation.kt`**:
   - Added `PAD_MS = 200` / `PAD_SAMPLES = 3200` constants.
   - Added pure batch helper `padSegments(segments, source, padding)` that expands
     each segment by `padding` samples on both sides, clamped to `source`.
   - `segmentAudioWithVad(samples, vad, window, padding = 0)` applies `padSegments`
     as a post-drain pass when `padding > 0` (default 0 preserves historical
     behavior; no production caller today).
   - `LiveVadDrainer(vad, padding = 0)` gained a `GrowingFloatArray` rolling source
     window (absolute-indexed, pruned after each drain to `prevEnd - padding`) so
     every closed segment is padded from real pushed audio — pre from history
     before the boundary, post from samples that arrived after it (best-effort).
2. **Sessions** — `SherpaOfflineSession` and `DolphinAttnSession` construct
   `LiveVadDrainer(vadLike, padding = PAD_SAMPLES)`; `EngineTranscriptionSession`
   gained a `padding: Int = PAD_SAMPLES` ctor param forwarded to the drainer.
3. **Tests** — new `SegmentPaddingTest` (9 cases): live mid-buffer full padding,
   pre-clamp at stream start, post-clamp at arrived samples, history retention
   across idle traffic, no-padding pass-through, batch `padSegments` clamps, batch
   `segmentAudioWithVad` with/without padding.

## Verification

- **JVM unit tests** (no device): new padding cases + all existing suites green.
- **Build:** `./gradlew :app:assembleDebug`, `:app:testDebugUnitTest`, `lintDebug`.
- **On-device:** utterance-edge word clipping regression check on the phone.

## Comments

- **2026-09-15**: Created from the VAD-accuracy spec (padding = ticket 01 in the
  new `vad-accuracy` feature folder; merging + cross-segment context are follow-up
  tickets under the same spec).