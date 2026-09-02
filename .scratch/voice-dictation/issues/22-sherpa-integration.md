# TICKET 22: sherpa-onnx integration with initial_prompt

## Goal
Integrate sherpa‑onnx into the Android app, enabling `initial_prompt` support to fix the English‑leak issue on short dictation clips.

## Acceptance Criteria
- [x] sherpa‑onnx dependency added (Maven or prebuilt .so) to the app's Gradle file
- [x] `WhisperEngine` (or equivalent) rewritten to use sherpa‑onnx `OfflineRecognizer` API
- [ ] `initial_prompt` parameter passed to sherpa‑onnx — not available in Java API (GitHub #2295); workaround: `language="en"` in `OfflineWhisperModelConfig` keeps output in Latin/Roman script
- [x] VAD (voice activity detection) enabled via separate `Vad` class with `SileroVadModelConfig`
- [x] Thread count configured auto‑select `min(6, Runtime.getRuntime().availableProcessors())` on first launch, with user override in Settings
- [x] App builds and runs on A50 emulator/device
- [ ] Transcription output stays in Roman Urdu/Latin script without English leakage on typical short clips — requires full PCM frame pipeline + ONNX model conversion (ticket #21)

## Notes
- sherpa‑onnx Java API 1.13.5 does not have `setVadEnable()` or `setInitialPrompt()` methods
- VAD handled by separate `Vad` class with `SileroVadModelConfig` (not via `OfflineRecognizerConfig`)
- `initial_prompt` not supported in Java API; maintainer confirms "no plan to add it" (issue #2295)
- Workaround: `language="en"` in `OfflineWhisperModelConfig` keeps Roman Urdu output in Latin script
- ONNX model format required (GGML `.bin` not supported); ticket #21 converts GGML→ONNX
- Builder pattern confirmed working: `OfflineRecognizerConfig.builder()`, `OfflineModelConfig.builder()`
- `OfflineModelConfig` uses no-arg constructor + setters (`setNumThreads()`, `setDebug()`, `setWhisper()`)
- Whisper language passthrough: `SherpaWhisperEngine.transcribe()` calls `recognizer.setConfig()`
  before each decode (official runtime language-change mechanism, k2-fsa/sherpa-onnx#1116).
  This pins a user-picked code (e.g. `"ur"`) at decode time, fixing Urdu→Hindi misrecognition
  on multilingual Whisper models. The C++ whisper impl's `DecodeStream()` re-applies the
  whisper sub-config to the decoder via `decoder_->SetConfig()` on every decode, so the
  change takes effect. Dolphin CTC has no language field (issue #2587, feature #3904).

## Dependencies
- sherpa‑onnx 1.13.5 (Java API + native libs via JitPack)
- Android NDK toolchain
- Updated `WhisperEngine` interface — `SherpaWhisperEngine` implements `WhisperEngine`
- Model conversion GGML→ONNX (ticket #21)

## Labels
sherpa-onnx, prompt, VAD, NDK, thread-count, Roman-Urdu

## Implementation Summary
 sherpa-onnx Java API 1.13.5 compiles successfully. Key API patterns confirmed:
- `OfflineRecognizerConfig.builder().setOfflineModelConfig().setDecodingMethod().build()`
- `OfflineModelConfig.builder().setNumThreads().setDebug().setWhisper().build()`
- VAD via separate `Vad` + `SileroVadModelConfig` 
- `initial_prompt` unavailable; `language="en"` workaround for Roman Urdu
- Full PCM frame pipeline needed for complete transcription
- ONNX model format (not GGML) required for sherpa-onnx
<tool_call>