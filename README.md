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

## License

This is a learning project. No license file yet.
