# 🎙️ Voice Dictation (Local)

> Fully on-device dictation for WhatsApp. No cloud, no WhatsApp API — your voice never leaves the phone.

[![Platform](https://img.shields.io/badge/platform-Android%208.0%2B%20%7C%20Linux%20harness-blue)]()
[![Engine](https://img.shields.io/badge/engine-whisper.cpp%20%2B%20faster--whisper-purple)]()
[![Privacy](https://img.shields.io/badge/privacy-100%25%20on--device-brightgreen)]()

---

## Overview

A fully on-device voice dictation flow. The user taps a recorder, speaks a message, and the transcript is copied to the clipboard — ready to paste into WhatsApp. **Everything runs on the device.** No internet required after the initial model download.

- **Phase 1 (Termux):** A prototype running on an Android phone via Termux, with the Linux desktop as a development/validation harness.
- **Phase 2 (Native Android):** A native Android app (Jetpack Compose, Kotlin) replacing the Termux prototype with a reliable launch surface, a resident model for fast repeated dictations, and a model picker for switching between models and language modes.

## Architecture

```
┌─────────────────────┐     ┌──────────────────────┐
│   Android Phone     │     │   Linux Development   │
│                     │     │   Harness             │
│  ┌───────────────┐  │     │  ┌────────────────┐  │
│  │ whisper.cpp   │◄────────►│  │ faster-whisper │  │
│  │ (on-device)   │  │     │  │ (validation)   │  │
│  └───────────────┘  │     │  └────────────────┘  │
│  ┌───────────────┐  │     │                       │
│  │ resident      │  │     │  Model catalog:       │
│  │ model (hot)   │  │     │  - Roman-Urdu q4_0    │
│  └───────────────┘  │     │  - English base/en    │
│  ┌───────────────┐  │     │  - Multilingual       │
│  │ VAD + ASR     │  │     │  - 10 models total    │
│  └───────────────┘  │     │                       │
└─────────────────────┘     └──────────────────────┘
```

## Features

- ✅ **100% on-device** — transcription happens locally, no audio uploads
- ✅ **Multiple models** — 10 models from tiny to small, English and Roman-Urdu
- ✅ **Resident model** — loaded once, reused across dictations for speed
- ✅ **VAD integration** — Silero VAD for voice activity detection
- ✅ **Manual send only** — no automessaging, no WhatsApp API
- ✅ **Offline after setup** — models downloaded once, work without internet

## Installation

### Prerequisites

- **Git**
- **Linux harness:** Python 3.11–3.13 (or [`uv`](https://github.com/astral-sh/uv))
- **Android app:** Android Studio or Android SDK + JDK 17 + an Android device (or emulator)

### 1. Clone the repo

```bash
git clone <your-repo-url>
cd voicedictation
```

### 2. Linux development harness (validate faster-whisper)

```bash
# one-time: install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

cd tools/validate_stt
uv sync
uv run validate_stt.py
```

This creates a Python 3.11 virtualenv and runs the `base.en` validation sample so you know the transcription pipeline works before touching the phone.

### 3. Build and run the Android app

First create `android/local.properties` with your Android SDK path (this file is gitignored):

```bash
echo "sdk.dir=$ANDROID_HOME" > android/local.properties
```

If `$ANDROID_HOME` is not set, use the absolute SDK path instead, e.g. `sdk.dir=/home/you/Android/Sdk`.

Then build the app:

```bash
cd android
./gradlew :app:assembleDebug
```

The APK is at `app/build/outputs/apk/debug/app-debug.apk`. Install it to a connected device:

```bash
adb install -r app/build/outputs/apk/debug/app-debug.apk
```

For full toolchain setup (JDK, SDK, device connection, wireless debugging), see [`android/SETUP.md`](android/SETUP.md).

### 4. Models

The app downloads the default model (Roman-Urdu q4_0) from HuggingFace on first launch and works offline afterward.

For local spike/QA assets:

```bash
mkdir -p android/spike
curl -fsSL -o android/spike/ggml-base-q8_0.bin \
  "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base-q8_0.bin"
curl -fsSL -o android/spike/jfk.wav \
  "https://github.com/ggerganov/whisper.cpp/raw/master/samples/jfk.wav"
```

## Key Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Phone engine | **whisper.cpp** (GGML/C++) | Builds natively on Termux, no NDK/CMake needed |
| Linux harness | **faster-whisper** | Faster validation on desktop before phone testing |
| Android packaging | **Prebuilt AAR** (`whisper-android`) | Single Gradle dependency, no native toolchain |
| Model lifecycle | **Resident model** | Avoid reloading tens-to-hundreds of MB per dictation |
| Language | **Roman-Urdu + English** | Primary use case; multilingual deferred |
| Send flow | **Manual** | Clipboard paste into WhatsApp; no auto-Enter |

## Quick Start

### On the phone (Termux prototype)
```bash
# Follow the setup runbook
cat android/SETUP.md
```

### On the Linux development machine
```bash
# Validate transcription faster-whisper before touching the phone
cd tools/validate_stt
uv sync && uv run validate_stt.py
```

> **Note for anyone cloning the repo:** Before building the Android app, you must create `android/local.properties` yourself (it is gitignored, so it is never committed) with your own Android SDK path, e.g. `sdk.dir=/path/to/your/Android/Sdk`. Without this file, Gradle will fail with "SDK location not found". See [`android/SETUP.md`](android/SETUP.md) for the full setup runbook.

## Documentation

| Doc | Description |
|-----|-------------|
| [`CONTEXT.md`](CONTEXT.md) | Domain glossary (terms to use and avoid) |
| [`docs/adr/`](docs/adr/) | Architecture decision records |
| [`android/SETUP.md`](android/SETUP.md) | Android build and on-device runbook |
| [`AGENTS.md`](AGENTS.md) | Agent workflow and repository conventions |
| `.scratch/voice-dictation/spec.md` | Full feature specification and tickets |

## ADRs

| # | Decision |
|---|----------|
| [0001](docs/adr/0001-fully-local-dictation-no-whatsapp-api.md) | Fully local dictation, no WhatsApp API |
| [0002](docs/adr/0002-whisper-cpp-for-phone-engine.md) | whisper.cpp as the on-phone engine |
| [0003](docs/adr/0003-prebuilt-whisper-cpp-aar-on-android.md) | Prebuilt whisper.cpp AAR, no NDK build |
| [0004](docs/adr/0004-resident-model-lifecycle.md) | Resident model lifecycle in the native app |
| [0005](docs/adr/0005-shared-libonnxruntime-version-lockstep.md) | sherpa-onnx / onnxruntime version lockstep |
| [0006](docs/adr/0006-vad-parameter-alignment.md) | VAD parameter alignment for sherpa-onnx and Dolphin |

## Project Structure

```
.
├── android/                  # Native Android app (Kotlin + Compose)
├── tools/
│   ├── validate_stt/         # Linux faster-whisper validation harness
│   ├── dolphin-onnx/         # Dolphin ONNX export & verification
│   ├── roman-urdu/           # Roman-Urdu Whisper model converters
│   └── urdu-eval/            # Urdu-script model evaluation harness
├── docs/
│   ├── adr/                  # Architecture decision records
│   └── agents/               # Agent workflow docs
├── .scratch/voice-dictation/ # Feature spec and implementation tickets
├── AGENTS.md                 # Agent workflow conventions
└── README.md                 # This file
```

## Dependencies

**Python (Linux development harness)**:
- `av` — audio/video processing
- `faster-whisper` — Whisper inference engine
- `librosa` — audio analysis
- `soundfile` — audio file I/O
- `torch` — tensor computation
- `transformers` — HuggingFace transformers

**Android**:
- `whisper-android` — whisper.cpp AAR
- `onnxruntime-android` — ONNX inference
- `sherpa-onnx` — Serger ASR engine
- `jlayer` — MP3 decoder

## Model Sources

Models are hosted on HuggingFace:

| Model | Repository |
|-------|------------|
| Roman-Urdu (default) | [`femustafa/voicedictation-models`](https://huggingface.co/femustafa/voicedictation-models) |
| Dolphin ONNX (Urdu) | [`onnx-community/dataocean-dolphin-asr`](https://huggingface.co/onnx-community/dataocean-dolphin-asr) |
| Whisper base models | [`ggerganov/whisper.cpp`](https://huggingface.co/ggerganov/whisper.cpp) |
| Urdu fine-tunes | [`cheetos18/whisper-small-roman-urdu`](https://huggingface.co/cheetos18/whisper-small-roman-urdu), [`kingabzpro/whisper-base-urdu-full`](https://huggingface.co/kingabzpro/whisper-base-urdu-full), [`Qasimhassan65/whisper-small-urdu`](https://huggingface.co/Qasimhassan65/whisper-small-urdu) |

## License

Apache License 2.0. See [`LICENSE`](LICENSE) for details.
