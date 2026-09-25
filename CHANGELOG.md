# Changelog

All notable changes to Kalaam Scribe. Newest first.

Release notes here are the source of truth; each GitHub release body is copied
from the matching section below. See [`android/SETUP.md`](android/SETUP.md) §4.5
for how to cut a release.

Requirements have not changed since v1.0.0 and are documented once in the
[README](README.md#requirements).

## v1.0.2

Adds a scrollbar and character count to the transcript card, and makes the copied
notification show the end of a dictation. No model, engine, or accuracy changes.

### Improvements

- **Visible transcript scrollbar** — the transcript card is a fixed-height window
  that auto-scrolls to the newest text. That is right for proving the tail landed,
  but nothing told you the text was long or that the card scrolls at all. An
  always-visible track-and-thumb scrollbar now runs down the right edge whenever
  the text overflows, and disappears when it fits. The thumb's length shows how
  much is visible; its position shows where you are, sitting at the bottom when
  auto-scrolled to the end.
- **Character count** — the transcript footer reads
  `Transcribed in 3.4 s · 2,399 chars`, and the copy confirmation says how many
  characters it copied.
- **Copied notification shows the tail** — the collapsed notification only has room
  for about two lines, and it used to show the *start* of the transcript, which is
  already visible on screen. It now shows the last 140 characters, so a glance at
  the notification confirms the dictation finished with real words. The expanded
  notification still carries the full text.

### Regression guards

The scrollbar is covered by JVM layout tests (Robolectric + Compose): present for
a 2,399-character transcript, absent for a short one, across small-, medium-, and
large-phone sizes. The existing Copy and Full-text layout regressions stay green.

## v1.0.1

Patch release fixing small-phone layout bugs from v1.0.0. No feature changes.

### Fixes

- **Welcome screen overflow** — after picking a dictation language, the model list
  could push the Continue button below the fold on small phones. The screen now
  scrolls, so Continue is always reachable.
- **Transcript clipping** — a very long transcript could grow past the screen edge
  and push the Copy button out of view. The transcript body is now capped and
  scrolls inside its card, keeping Copy and history on-screen.

### Regression guards

Both bugs are now covered by JVM layout tests (Robolectric + Compose) that render
the screens at small-, medium-, and large-phone sizes, so the fixes can't silently
regress.

## v1.0.0

First release. A native Android app for fully **on-device voice dictation** — no
cloud, no API keys. Tap record, speak, stop, and the transcript is copied to your
clipboard, ready to paste anywhere. Audio never leaves the phone; the only internet
use is the one-time download of your chosen ASR model.

### What's inside

- **100% on-device transcription** — audio never leaves the phone
- **Dual ASR engines** — ONNX Whisper (sherpa-onnx) for English / multilingual /
  Roman-Urdu, and Dolphin attention ASR (onnxruntime) for Urdu-script dictation
- **Language-aware model picker** — English, Roman-Urdu, Urdu, and auto-detect
- **Silero VAD segmentation** — live speech split into utterances
- **Two transcription modes** — Batch and simulated streaming (settings)
- **Long-audio chunking** — clips >14 s split by VAD into ≤14 s chunks with
  word-seam merge to remove boundary duplication
- **Resident model** — loaded once and reused across dictations
- **In-app model download** — fetched from HuggingFace with progress
- **File transcription** — transcribe a WAV/MP3 from your picker
- **Manual send only** — transcript copied to the clipboard; you send it yourself
- **Offline after setup** — works with no internet once the model is downloaded
- **Material 3 UI** (Jetpack Compose)

### Default model

The picker recommends **Roman-Urdu · int8** by default; it suggests
Whisper · Balanced for English, Roman-Urdu for the Roman-Urdu language, and
Dolphin · Balanced for Urdu.

### Known limitations

- Self-signed: users must allow unknown sources
- First release; expect rough edges
