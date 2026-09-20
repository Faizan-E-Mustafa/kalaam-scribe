# Kalaam Scribe v1.0.0

First release. A native Android app for fully **on-device voice dictation** — no cloud, no API keys. Tap record, speak, stop, and the transcript is copied to your clipboard, ready to paste anywhere. Audio never leaves the phone; the only internet use is the one-time download of your chosen ASR model.

## Install

1. Download `app-release.apk` from the assets below.
2. Open it on your phone and allow "Install from unknown sources" when prompted.
   - The APK is self-signed, so Android shows an unknown-publisher warning — that's expected.

## What's inside

- **100% on-device transcription** — audio never leaves the phone
- **Dual ASR engines** — ONNX Whisper (sherpa-onnx) for English / multilingual / Roman-Urdu, and Dolphin attention ASR (onnxruntime) for Urdu-script dictation
- **Language-aware model picker** — English, Roman-Urdu, Urdu, and auto-detect
- **Silero VAD segmentation** — live speech split into utterances
- **Two transcription modes** — Batch and simulated streaming (settings)
- **Long-audio chunking** — clips >14 s split by VAD into ≤14 s chunks with word-seam merge to remove boundary duplication
- **Resident model** — loaded once and reused across dictations
- **In-app model download** — fetched from HuggingFace with progress
- **File transcription** — transcribe a WAV/MP3 from your picker
- **Manual send only** — transcript copied to the clipboard; you send it yourself
- **Offline after setup** — works with no internet once the model is downloaded
- **Material 3 UI** (Jetpack Compose)

## Requirements

- **Android 8.0+** (API 26+), arm64-v8a device
- **~300 MB+ free space** for the default model
- **Internet once** — to download your chosen model from HuggingFace on first launch

## Default model

The picker recommends **Roman-Urdu · int8** by default; it suggests Whisper · Balanced for English, Roman-Urdu for the Roman-Urdu language, and Dolphin · Balanced for Urdu.

## Known limitations

- Self-signed: users must allow unknown sources
- First release; expect rough edges

## Development

Build instructions and the signed-APK release process: see `android/SETUP.md` §4.5.