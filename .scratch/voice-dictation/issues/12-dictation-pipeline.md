# 12: Dictation pipeline (clipboard + notification)

**What to build:** Wire record → transcribe → clipboard + "Copied" notification
into the end-to-end dictation flow (the port of `dictate.sh`). Transcription
reuses the WhisperManager's resident Model, so no model is loaded per
transcription.

**Blocked by:** 09 (WhisperManager), 10 (Model catalog + downloader), 11 (Mic
recording).

**Status:** done (device-verified)

## Implementation

- `DictationViewModel` (now takes `WhisperManager` via a `Factory`) orchestrates
  record -> transcribe -> clipboard. On stop: `recorder.stop()`, then
  `whisper.transcribe(audioPath)` using the **resident** model, writes the trimmed
  result to the clipboard (no silent empty clipboard), surfaces a "Copied"
  notification. Guards the "no model loaded" case.
- `NotificationHelper` — idempotent channel creation + "Recording… tap Stop" and
  "Copied" notifications (plain notifications; no foreground service, ADR 0001:
  manual send only). Best-effort on API 33+ without `POST_NOTIFICATIONS`.
- `DictationScreen` — shows Record/Stop/"Transcribing…", the transcript, or the
  error; requests `RECORD_AUDIO` (and `POST_NOTIFICATIONS` on 33+).
- Resident reuse is code-guaranteed: `switchTo` is only called from the picker;
  `transcribe` reuses the held `resident` handle and never reloads (ADR 0004).

## On-device verification (2026-08-31)

Downloaded the public English quantized model (`ggml-base.en-q8_0.bin`) via the
picker, ran two dictations in a row — both transcribed successfully. Empty/no-speech
Stop produced the "No speech detected — please try again" failure without filling
the clipboard.

## Acceptance criteria

- [x] Tapping record captures a dictation clip; stopping it transcribes using the
      resident Model.
- [x] The transcript is written to the Android clipboard.
- [x] A "Copied" notification shows the transcript (and a "Recording… tap Stop"
      notification is shown while recording, per the notify convention in
      `dictate.sh`).
- [x] Empty/no-speech audio is handled with a clear failure (no silent empty
      clipboard).
- [x] Manual send only: no automessaging or auto-insert (ADR 0001 boundary).
- [x] A repeated dictation in the same session does not reload the model
      (resident Model reuse, verified end-to-end with 09).
