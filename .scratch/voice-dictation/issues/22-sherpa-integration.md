# TICKET 22: sherpa-onnx integration with initial_prompt

## Goal
Integrate sherpa‑onnx into the Android app, enabling `initial_prompt` support to fix the English‑leak issue on short dictation clips.

## Acceptance Criteria
- [ ] sherpa‑onnx dependency added (Maven or prebuilt .so) to the app's Gradle file
- [ ] `WhisperEngine` (or equivalent) rewritten to use sherpa‑onnx `Generate` API
- [ ] `initial_prompt` parameter passed to sherpa‑onnx `Generate` call (e.g. `"Assalam o alaikum. Mujue ek message bhejna hai..."`)
- [ ] VAD (voice activity detection) enabled for silence‑aware transcription
- [ ] Thread count configured auto‑select `min(6, Runtime.getRuntime().availableProcessors())` on first launch, with user override in Settings
- [ ] App builds and runs on A50 emulator/device
- [ ] Transcription output stays in Roman Urdu/Latin script without English leakage on typical short clips

## Notes
- The current AAR `WhisperConfig` surface (`language`, `translate`, `threads`) is replaced by sherpa‑onnx `Generate` params.
- NDK is required for the native `.so` libraries; acceptance confirmed in ticket #3.
- English Tiny/Small ONNX models can be downloaded directly; Roman Urge requires local GGML→ONNX conversion (ticket #21).

## Dependencies
- sherpa‑onnx Java API (or via JNI wrapper)
- Android NDK toolchain
- Updated WhisperEngine interface

## Labels
sherpa-onnx, prompt, VAD, NDK, thread-count, Roman-Urdu