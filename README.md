# 🎙️ Voice Dictation (Local)

> Fully on-device dictation for WhatsApp. No cloud, no WhatsApp API — your voice never leaves the phone.

[![Platform](https://img.shields.io/badge/platform-Android%208.0%2B-blue)]()
[![Engine](https://img.shields.io/badge/engine-ONNX%20Whisper%20%2B%20Dolphin%20ASR-purple)]()
[![Privacy](https://img.shields.io/badge/privacy-100%25%20on--device-brightgreen)]()

---

## Overview

A native **Android app** (Kotlin + Jetpack Compose) that turns spoken messages into text entirely on-device. Tap record, speak, stop — the transcript is copied to the clipboard, ready to paste into WhatsApp. Everything runs on the phone; the only internet use is the one-time download of your chosen ASR model.

The app is driven by ONNX ASR engines: **sherpa-onnx Whisper** for English / multilingual / Roman-Urdu transcription, and **Dolphin attention ASR** (onnxruntime) for Urdu-script dictation.

## Features

- ✅ **100% on-device** — transcription runs locally, no audio ever leaves the phone
- ✅ **Dual ASR engines** — ONNX Whisper (sherpa-onnx) and Dolphin attention (onnxruntime)
- ✅ **Language-aware model picker** — English, Roman-Urdu, Urdu, and auto-detect
- ✅ **Silero VAD segmentation** — speech split into utterances with padding + merging for accurate boundaries
- ✅ **Three transcription modes** — Batch, Batch + VAD, and live simulated streaming
- ✅ **Resident model** — loaded once, reused across dictations (no reload per clip)
- ✅ **In-app model download** — models fetched from HuggingFace with progress
- ✅ **File transcription** — transcribe a WAV/MP3 from your picker
- ✅ **Manual send only** — transcript is copied to the clipboard; you send it yourself
- ✅ **Offline after setup** — works with no internet once the model is downloaded

## Requirements

- **Android 8.0+** (API 26+), arm64-v8a device
- **~300 MB+ free space** for the default model (int8 tiers are the smallest useful option)
- **Internet once** — to download your chosen model from HuggingFace on first launch

## Getting Started

### Build the app

First create `android/local.properties` with your Android SDK path (this file is gitignored):

```bash
echo "sdk.dir=$ANDROID_HOME" > android/local.properties
```

If `$ANDROID_HOME` is unset, use the absolute path, e.g. `sdk.dir=/home/you/Android/Sdk`. Then build and install:

```bash
cd android
./gradlew :app:assembleDebug
adb install -r app/build/outputs/apk/debug/app-debug.apk
```

The APK lands at `app/build/outputs/apk/debug/app-debug.apk`. For the full toolchain setup (JDK, SDK, device connection, wireless debugging) see [`android/SETUP.md`](android/SETUP.md).

### First run

1. Pick your dictation language (English, Roman-Urdu, Urdu, or auto-detect).
2. Download the recommended model (or any model from the picker).
3. Tap the mic, speak, tap stop — the transcript is copied to the clipboard and a notification shows your dictation history.

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                    Voice Dictation                  │
│                      (Android app)                  │
│                                                     │
│  ┌───────────────┐   ┌───────────────────────────┐  │
│  │  Recorder     │   │  WhisperManager           │  │
│  │  (foreground  │   │  (resident model,         │  │
│  │  mic service) │──►│  VAD + transcription)     │  │
│  └───────────────┘   └────────┬──────────────────┘  │
│                               │                     │
│               ┌───────────────┼───────────────┐     │
│               ▼               ▼               ▼     │
│        SherpaWhisper   DolphinAttn      (GGML relay)│
│        ONNX Whisper    encoder+decoder   whisper.cpp│
│        (sherpa-onnx)   (onnxruntime)    AAR — kept  │
│        eng/multi/      Urdu (ur/PK)     but disabled│
│        Roman-Urdu      pin, beam search              │
│                                                     │
│  ModelDownloader ──► HuggingFace (one-time fetch)   │
└─────────────────────────────────────────────────────┘
```

**Transcription modes** (pick in Settings):

| Mode | Behaviour |
|------|-----------|
| **Batch** | One decode of the whole clip after Stop — simplest, works on every backend |
| **Batch + VAD** | Silero VAD splits the clip, merges utterances into ≤28 s chunks, pads each, decodes in order |
| **Simulated streaming** | Live VAD decodes each utterance as you speak, growing the transcript in real time |

The default is **Roman-Urdu · int8** (`roman-urdu-int8`), and the picker recommends Whisper · Balanced for English, Roman-Urdu for the Roman-Urdu language, and Dolphin · Balanced for Urdu.

## Models

Models are downloaded in-app from HuggingFace. Whisper models ship in `int8` (active) and `fp32` (retained) tiers; Dolphin ships `fp16/arm` and `int8`.

| Family | Tiers | Used for |
|--------|-------|----------|
| Whisper `tiny` · Fast | int8 | Fast English/multilingual dictation |
| Whisper `base` · Balanced | int8 | Default-balanced dictation (default for English / auto-detect) |
| Whisper `small` · High Accuracy | int8 | Best quality English/multilingual |
| Roman-Urdu fine‑tune | fp32, int8 | Urdu in Roman/Latin script (default model) |
| Dolphin attention (base/small) · Balanced / High Accuracy | fp16, int8 | Urdu-script dictation (respects the `ur/PK` language pin) |

Sources:
- Whisper ONNX: [`csukuangfj/sherpa-onnx-whisper-*`](https://huggingface.co/csukuangfj)
- Roman-Urdu + Dolphin attention: [`femustafa/voicedictation-models`](https://huggingface.co/femustafa/voicedictation-models)
- Silero VAD: [sherpa-onnx releases](https://github.com/k2-fsa/sherpa-onnx/releases)

## Project Structure

```
.
├── android/                  # Native Android app (Kotlin + Compose)
│   ├── app/                  # App module (sources + unit tests)
│   └── SETUP.md              # Android toolchain + on-device runbook
├── tools/
│   ├── dolphin-onnx/         # Dolphin ASR ONNX export & verification
│   └── roman-urdu/           # Roman-Urdu model conversion & deployment
├── docs/
│   └── agents/               # Agent workflow docs
├── .scratch/                 # Feature specs and implementation tickets
├── CONTEXT.md                # Domain glossary
├── AGENTS.md                 # Agent workflow conventions
└── README.md                 # This file
```

## Dependencies

**Android**:
- `sherpa-onnx` — ONNX ASR runtime (Whisper + Silero VAD)
- `onnxruntime-android` — Dolphin attention inference
- `whisper-android` — whisper.cpp AAR (GGML; retained, disabled in the picker)
- `jlayer` — MP3 decoding for file transcription
- Jetpack Compose (Material 3), AndroidX, Kotlin coroutines
- `kotlinx-coroutines-test` + JUnit (unit tests)

## Documentation

| Doc | Description |
|-----|-------------|
| [`android/SETUP.md`](android/SETUP.md) | Android build and on-device runbook |
| [`CONTEXT.md`](CONTEXT.md) | Domain glossary (terms to use and avoid) |
| [`AGENTS.md`](AGENTS.md) | Agent workflow and repository conventions |
| [`.scratch/`](.scratch) | Feature specs and implementation tickets |

## License

Apache License 2.0. See [`LICENSE`](LICENSE) for details.