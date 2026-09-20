# Kalaam Scribe v1.0.0

First release. A fully local, on-device voice dictation app for Android. No cloud, no API keys — audio is transcribed on your phone using whisper.cpp.

## Install

1. Download `app-release.apk` from the assets below.
2. Open it on your phone and allow "Install from unknown sources" when prompted.
   - This APK is self-signed, so Android will show an unknown-publisher warning — that's expected.

## What's inside

- On-device speech transcription via whisper.cpp (no internet required)
- Compose UI (Material 3)
- Supports arm64-v8a, armv7a, x86_64 devices
- File-upload transcription (WAV + MP3)

## Requirements

- Android 11 (API 26) or later
- Arm64 (arm64-v8a) recommended

## Known limitations

- Self-signed: users must allow unknown sources
- First release; expect rough edges

## Development

Build instructions: see `android/SETUP.md` §4.5 (Releasing a signed APK).