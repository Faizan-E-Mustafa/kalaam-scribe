# 12: Dictation pipeline (clipboard + notification)

**What to build:** Wire record → transcribe → clipboard + "Copied" notification
into the end-to-end dictation flow (the port of `dictate.sh`). Transcription
reuses the WhisperManager's resident Model, so no model is loaded per
transcription.

**Blocked by:** 09 (WhisperManager), 10 (Model catalog + downloader), 11 (Mic
recording).

**Status:** ready-for-agent

## Acceptance criteria

- [ ] Tapping record captures a dictation clip; stopping it transcribes using the
      resident Model.
- [ ] The transcript is written to the Android clipboard.
- [ ] A "Copied" notification shows the transcript (and a "Recording… tap Stop"
      notification is shown while recording, per the notify convention in
      `dictate.sh`).
- [ ] Empty/no-speech audio is handled with a clear failure (no silent empty
      clipboard).
- [ ] Manual send only: no automessaging or auto-insert (ADR 0001 boundary).
- [ ] A repeated dictation in the same session does not reload the model
      (resident Model reuse, verified end-to-end with 09).
