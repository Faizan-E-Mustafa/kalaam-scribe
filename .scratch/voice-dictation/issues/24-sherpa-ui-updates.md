# TICKET 24: sherpa-onnx UI updates (prompt + VAD)

## Goal
Update the Android app UI to expose prompt text input and VAD toggle for sherpa‑onnx transcription.

## Acceptance Criteria
- [ ] "Prompt" field added to `DictationScreen` or a dedicated `Settings` screen, accepting free‑text seed prompt.
- [ ] "VAD" toggle (on/off) added to the same UI, controlling voice‑activity detection in sherpa‑onnx.
- [ ] "Threads" spinner retained (already implemented) with values 2/4/6/8, default auto‑select 6.
- [ ] Save button persists prompt text and VAD preference via `SharedPreferences` (key `whisper_prompt` and `whisper_vad`).
- [ ] UI updates wired to the new `WhisperEngine` (sherpa‑onnx wrapper).
- [ ] No regression in existing dictation flow (model picker, recording, history).

## Notes
- UI can reuse the existing `SettingsScreen` composable that was drafted on the `threads` branch; only new fields need adding.
- Prompt text should be sent to sherpa‑onnx `Generate` as `initial_prompt`; if empty, sherpa‑onnx transcribes without prompt (fallback to current behavior).
- VAD toggle should enable/disable the `VAD` flag in sherpa‑onnx `Generate` params.

## Dependencies
- UI: `DictationScreen.kt`, `SettingsScreen.kt`, `RootApp.kt` routing.
- Backend: sherpa‑onnx `Generate` API params (`initial_prompt`, `vad`).

## Labels
UI, sherpa-onnx, prompt, VAD, settings, Compose