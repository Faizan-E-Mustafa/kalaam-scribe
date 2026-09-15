# Spec: Improve ASR accuracy with VAD-segmented audio

## Problem Statement

Simulated streaming and VAD-based segmentation currently degrade transcription
accuracy for Whisper, Dolphin CTC, and Dolphin attention models. Each VAD
`SpeechSegment` is decoded independently with no cross-segment context, no
padding around detected boundaries, and no merging of short utterances into
chunks that match the models' training distribution (Whisper/Dolphin are trained
on ~30s segments).

Three concrete accuracy-destroying gaps:

1. **No segment overlap/padding** — VAD cuts at exact silence boundaries; words at
   the onset/offset of a segment get clipped. (ADR 0006 flags this as a future
   optimization.)
2. **No cross-segment context** — each segment is decoded with a fresh decoder
   state; the model cannot use prior text to disambiguate homophones, proper
   nouns, or style.
3. **No segment merging** — short 1–3s utterances are decoded alone, outside the
   models' training distribution, instead of being batched into ~30s chunks with
   boundaries placed at minimally active speech regions (the WhisperX "VAD Cut &
   Merge" strategy).

## Solution

Improve transcription accuracy by:

1. **Segment padding** — add pre/post padding around each VAD segment boundary
   before decode so words near the edges are not clipped.
2. **Segment merging** — merge consecutive VAD segments whose combined duration
   stays under ~30s (matching Whisper/Dolphin training context), emitting merged
   chunks for decode.
3. **Cross-segment context** — feed prior decoded text back into the decoder as an
   initial prompt, where the backend supports it (Whisper `initial_prompt`;
   Dolphin attention has a controllable decoder prefix).

## User Stories

1. As a user dictating on the Phone, I want words spoken at the start and end of
   each utterance to be transcribed correctly, so that VAD boundary clipping does
   not eat letters or whole words.
2. As a user dictating short phrases with pauses between them, I want related
   phrases merged into a single decode, so that the model has enough acoustic
   context to transcribe accurately.
3. As a user dictating a multi-sentence clip, I want the model to carry text
   context from earlier segments into later ones, so that vocabulary and style
   stay consistent across the transcript.
4. As a developer, I want the padding/merging logic shared between the batch and
   live (simulated streaming) paths, so that accuracy behavior is consistent
   regardless of which path decoded the audio.
5. As a developer, I want padding behavior configurable and unit-tested against a
   fake VAD, so that native libraries are not required on the JVM.

## Implementation Decisions

### Decision 1: Segment padding (Ticket 32)

- Add pre/post padding to VAD-derived speech segments before passing them to the
  ASR models.
- Padding amount is a named constant, default **200 ms before + 200 ms after**
  (3200 samples at 16 kHz each side), clamped to the source audio bounds.
- Padding happens at the segmentation seam: both the batch `segmentAudioWithVad`
  and the live `LiveVadDrainer` need access to the full source buffer (or a
  rolling tail buffer in the live case) to grab samples before/after the segment
  range.
- The live drainer must carry a small rolling history buffer of the last
  `padSamples` samples plus a pending future buffer, since at the moment VAD
  closes a segment the future audio may not have arrived yet. Implementation
  choice: pre-padding is applied "late" — when the segment is drained, use what is
  already in history; post-padding is best-effort for the last segment (delay
  surfacing the segment until `padSamples` more samples arrive, or apply at flush).

### Decision 2: Segment merging

- Merging is implemented as a post-processing pass over closed segments before
  decode: consecutive segments are merged while the running total stays under a
  cap (default **28 s** to leave Whisper headroom under its 30 s design limit).
- Boundaries of merged chunks therefore fall at real silence gaps, matching
  WhisperX's cut-and-merge philosophy.
- Merging applies to the batch path; for live streaming a segment is emitted when
  VAD closes it (per-utterance publish is the point of simulated streaming) so the
  merging decision is left as a follow-up tuning item.

### Decision 3: Cross-segment context

- **Out of scope for ticket 32.** Whisper's `initial_prompt` is a workaround in
  sherpa-onnx (GitHub #2295, `SherpaWhisperEngine.kt:21`); upstream exposes no
  field for it. Dolphin CTC has no text-conditioning mechanism at all. Dolphin
  attention can prepend prior text tokens before beam search but was trained
  without previous-text conditioning (per the Dolphin paper), so it is
  experimental. Rounds 2+ if data shows margin worth chasing.

### Out of scope (this spec phase)

- Cross-segment context / initial prompt conditioning (Decision 3 — research +
  gap analysis only).
- Segment merging for the live streaming path.
- VAD parameter re-tuning beyond what ADR 0006 already set.
- Fine-tuning any model.

## Testing Decisions

- **Seam:** `VadLike` (existing) plus pure functions over `FloatArray`/segment
  lists. JVM-only tests, no native sherpa libs, no `android.content.Context`.
- **What makes a good test:** observed external behavior of the segmentation
  input/output — given a source buffer and a fake `VadLike` emitting segments,
  the returned segments carry correct padding samples and clamped bounds; live
  drainer applies pre/post padding correctly as segments close.
- **Modules tested:** `VadSegmentation.kt` (padding helpers, live-drainer padding)
  via new/extended tests; existing `VadSegmentationTest` stays green.
- **Prior art:** existing `VadSegmentationTest` (5 cases, fake `VadLike`),
  `StreamingSessionNoDropTest` (4 cases).

## Further Notes

- ADR 0006 line 61: "Adding padding is a potential future optimization but is not
  used by upstream." This spec is that optimization, landed under the ADR's own
  note.
- WhisperX's VAD cut-and-merge research is the canonical reference for why
  boundary placement and context length matter for Whisper-family models
  (Bain et al. 2023).
- Applies to all three ASR backends uniformly: sherpa Whisper ONNX, Dolphin CTC,
  Dolphin attention.