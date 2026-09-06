# VAD parameter alignment for sherpa-onnx and Dolphin models

VAD parameters were misaligned with upstream references, causing perceived accuracy
degradation after VAD integration. This ADR documents the fix.

## Problem

The app uses three ASR engines backed by sherpa-onnx's Silero VAD for
voice-activity detection:

- `SherpaWhisperEngine` — Whisper ONNX models
- `DolphinCtcEngine` — Dolphin CTC multilingual models
- `DolphinAttnEngine` — Dolphin attention ASR models (via onnxruntime-android)

Each engine independently configured `SileroVadModelConfig` with values that
diverged from upstream recommendations.

## References consulted

- **sherpa-onnx C API example** (`c-api-examples/vad-whisper-c-api.c`):
  canonical VAD+Whisper integration shipped with the library.
- **Dolphin repository** (`DataoceanAI/Dolphin`): `transcribe.py` uses FunASR VAD
  for long audio segmentation with `max_single_segment_time = 30` seconds.
- **sherpa-onnx Android examples**: `SherpaOnnxVadAsr`,
  `SherpaOnnxSimulateStreamingAsr`.

## Parameters changed

| Parameter | Before | After | Source |
|---|---|---|---|
| `threshold` | 0.5 | **0.25** | sherpa-onnx C API reference |
| `maxSpeechDuration` | 5.0–10.0 | **30.0** | Dolphin repo `SPEECH_LENGTH = 30` |
| `minSilenceDuration` | 0.25 (DolphinAttn) | **0.5** | sherpa-onnx C API reference |
| `tailPaddings` | not set | **0** | sherpa-onnx C API reference (`whisper.tail_paddings = 0`) |

### Why threshold 0.5 → 0.25 matters

A threshold of 0.5 requires higher speech confidence to trigger detection.
Silero VAD's internal look-ahead produces segments that cut abruptly at
utterance boundaries. Lowering to 0.25 captures quieter speech and fuller
word boundaries, reducing clipping at segment onset/offset.

### Why maxSpeechDuration 10 → 30

The Dolphin repository sets `max_single_segment_time = 30` seconds for VAD
segmentation. Shorter max durations cause more splits, each decoded independently
with no cross-segment context. Fewer splits = more acoustic context per decode.

### Why tailPaddings = 0

The sherpa-onnx C API explicitly sets `whisper.tail_paddings = 0`. Without
this, the Java API default (`tail_paddings = -1`, auto-pad to 30s) was active,
changing how short VAD segments hit the Whisper encoder.

## Notes for future tuning

- These values are now in one place per engine: `createSession()` / `load()`.
- If FunASR VAD (`iic/speech_fsmn_vad_zh-cn-16k-common-pytorch`) is evaluated
  as a replacement for Silero VAD, the max segment duration is the primary knob
  to align with the Dolphin reference.
- The VAD does not add padding around segment boundaries. The sherpa-onnx
  reference examples decode each VAD segment as-is without explicit pre/post padding.
  Adding padding is a potential future optimization but is not used by upstream.
