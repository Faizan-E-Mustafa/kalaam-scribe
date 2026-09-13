# Spec: Local Voice Dictation for WhatsApp

> This file is the full feature spec. Phase 1 (Termux prototype) is below as
> written; the native Android app that replaces it is captured in the
> "Phase 2: native Android app" section nearer the bottom. Phase 1 content is
> retained for history and because the Linux harness stays in use.

Status: ready-for-agent

## Problem Statement

The user wants to send WhatsApp messages by speaking instead of typing. Chatting on
a phone is slow, and existing dictation either uploads audio to the cloud or is
bundled into third-party keyboards. They want dictation that runs entirely on their
Android phone with no internet, no cloud, and no WhatsApp API, so their voice never
leaves the device.

## Solution

A fully on-device dictation flow. The user taps a recorder, speaks a message, and
the transcript is copied to the Android clipboard ready to paste into WhatsApp. The
first version runs as a Termux prototype on the phone to validate speed and accuracy;
the eventual product is a native keyboard (stretch goal).

## User Stories

1. As a WhatsApp user, I want to tap a button and speak a message, so that I don't have to type it.
2. As a WhatsApp user, I want my spoken message to become text on my phone, so that it can be pasted into a chat.
3. As a privacy-conscious user, I want transcription to happen on-device with no internet, so that my voice never leaves my phone.
4. As a user, I want a notification when my message is ready, so that I know when to paste it into WhatsApp.
5. As a user, I want the transcript copied to the clipboard automatically, so that pasting into WhatsApp is one step.
6. As a user, I want to record in English, so that my English WhatsApp messages are transcribed accurately.
7. As a user, I want the recording to stop easily without typing, so that I can dictate hands-free.
8. As a user, I want to review the transcript before sending, so that I don't send a mistake (manual send only).
9. As a user, I want the whole flow to work offline after the initial model download, so that I can dictate anywhere.
10. As a developer, I want to benchmark transcription speed/accuracy on the phone, so that I can pick the best model size.
11. As a developer, I want to validate the model on a fast Linux machine first, so that I de-risk the phone setup.

## Implementation Decisions

- Use **whisper.cpp** (GGML/C++ port) as the on-phone transcription engine, running on CPU only — it builds natively in Termux without the ctranslate2 source-compile that faster-whisper requires (ADR 0002). The Linux validation harness keeps **faster-whisper**.
- Phase 1 uses the **`base.en`** model in a quantized form with voice-activity detection, chosen for the phone's CPU-only constraints. Other sizes (`small`, `tiny`) are compared in the benchmark ticket.
- Runtime target is the **phone only**; the Linux desktop is a development/validation harness only.
- Recording uses **Termux:API** (`termux-microphone-record`); the transcript is written to the Android clipboard via `termux-clipboard-set` and signalled via `termux-notification`.
- The interaction is: tap recorder widget → speak → tap/volume to stop → wait for local transcription → "Copied" notification → user long-press-pastes into WhatsApp → user sends manually.
- **Manual send only.** No auto-Enter/automessaging in Phase 1.
- **Phase 2 (stretch goal):** a native Android IME keyboard app that inserts text directly into the focused field (no paste), reusing the validated model. Not specced/ticketed now. The native app is where **hold-to-record** (press = record, release = stop) belongs — Termux:Widget cannot distinguish press-hold from release, so the Phase 1 Termux prototype uses tap-to-start / tap-Stop instead. The native app is also **required for a reliable launch surface**: on Android (esp. the phone's 4GB RAM) the Termux widget only fires while the Termux process is alive and is killed in the background, so Phase 2 needs a foreground service / proper IME to start recording reliably from the home screen.
- **Multilingual support** is deferred; Phase 1 is English-only.

## Testing Decisions

- The key acceptance test is behavioral and end-to-end: record a known English phrase on the phone, transcribe, and confirm the text in the clipboard matches the phrase (within a tolerable WER).
- Benchmark ticket measures transcription time vs. audio length and qualitative accuracy on the phone, comparing `base`/`small`/`tiny` to choose the default.
- A Linux harness test confirms the `base.en` model loads and transcribes a known audio sample, catching environment/wheel problems before touching the phone.
- No unit-test framework is imposed on the Termux shell layer; verification is via the end-to-end behaviour above plus the benchmark.

## Out of Scope

- WhatsApp API integration and automessaging (rejected by ADR 0001).
- Native Android IME keyboard app (Phase 2, stretch goal, unticketed).
- Multilingual transcription.
- Live/streaming transcription while speaking (the phone CPU cannot keep up).
- Auto-insert into a focused field in Phase 1 (the paste step is accepted).

## Further Notes

- Primary risk: `faster-whisper`/`ctranslate2` wheels on the developer's Python 3.14 Linux environment may not exist (worked around in the linux-dev-harness ticket with uv-managed Python 3.11). On the phone, ctranslate2 has no aarch64 wheel, which is why the phone uses whisper.cpp (ADR 0002) instead of faster-whisper.
- Secondary risk: phone transcription speed; the benchmark ticket exists to de-risk and choose the model size.

---

# Phase 2: native Android app

Replaces the Termux prototype (Phase 1) with a native Android app installed from a
sideloaded APK. Runs entirely on-device; needs internet only once to download the
selected Model. The deliverable is an installable APK for the developer's own Android phone (no Play Store publishing required).

## Problem Statement

The Termux prototype works but is unreliable as a launch surface: the Termux
widget only fires while the Termux process is alive and is killed in the
background on the phone's 4 GB RAM, and it cannot distinguish hold from release for
hold-to-record. The user wants the same on-device dictation without depending on
Termux.

## Solution

A native Android app that keeps the current interaction (tap to record → speak →
stop → local transcript on the clipboard → "Copied" notification → user pastes
into WhatsApp and sends manually), but launches reliably from the home screen in
the background and reuses a resident Model so repeated dictations do not reload
the model. Android 8.0+ (API 26+), arm64-v8a.

## User Stories

1. As a user, I want to record a WhatsApp dictation clip by tapping a button or
   widget from the home screen, so that I don't rely on Termux.
2. As a user, I want the recording to keep running in the background while I use
   other apps, so that a background launch is reliable on my 4 GB phone.
3. As a user, I want a visible "Recording… tap Stop" notification while dictating,
   so that I can stop hands-free without typing.
4. As a user, I want the transcript copied to the clipboard with a "Copied"
   notification, so that I can paste it into WhatsApp (manual send only).
5. As a user, I want repeated dictations in a session to reuse the same loaded
   Model instead of reloading each time, so that dictation stays fast.
6. As a user, I want to switch Model (e.g. Roman-Urdu vs English vs multilingual,
   full vs quantized) and its language mode from a model picker, so that I can
   choose speed/accuracy per need.
7. As a user, I want the app to download a Model from HuggingFace on first use
   with progress, so that I don't have to copy files manually.
8. As a user, I want everything to work offline after the chosen Model is
   downloaded, so that my voice never leaves the phone.
9. As a developer, I want the on-device benchmark (speed/accuracy per Model) to
   guide the default, so that the app is usable on the phone.
10. As a developer, I want the app to transcribe in both English and Roman-Urdu,
    so that the primary use case is covered.

Phase 2 keeps the Phase 1 boundary decisions: manual send only (no automessaging,
ADR 0001); whisper.cpp engine (ADR 0002); English + Roman-Urdu in scope, other
languages only via the multilingual Model.

## Implementation Decisions

- **Project layout**: a new top-level `android/` directory in the repo, a Kotlin
  Android app using Jetpack Compose.
- **Engine packaging**: whisper.cpp delivered as a prebuilt AAR
  (`dev.ffmpegkit-maintained:whisper-android`) with no NDK/CMake source build
  (ADR 0003). Models are GGML/GGUF files loaded from app storage at runtime,
  including quantized forms (q8_0 / q5_1).
- **Resident model**: one Model loaded in memory and reused across dictations
  while the app process lives; reload only on Switch or cold start (ADR 0004).
  Serialize the audio pipeline so a second dictation cannot start during a
  transcription.
- **Model catalog**: ten Models, each a model file plus its language mode (see
  Notes for the table). In-app download from HuggingFace with progress; no size
  gating. The default on first launch is Roman-Urdu q4_0, matching the current
  on-device setup (see tickets 16–18).
- **Recording**: mic capture to 16 kHz mono PCM (matching the Termux 16 kHz step);
  the whisper library decodes/resamples as needed (no ffmpeg on-device).
- **Foreground service**: starts only while recording/transcribing, then stops.
  No permanent always-on service. The persistent notification is the
  record/stop/copied surface.
- **Launch surface**: app icon plus a home-screen widget/shortcut to start
  recording.
- **File transcription**: out of scope for v1 of this phase (mic dictation only),
  matching the primary WhatsApp use case.

## Testing Decisions

- **Unit seam — WhisperManager**: the model lifecycle is testable Kotlin without
  a device. Tests assert that repeated transcribe calls reuse the resident Model
  (load count stays 1), that Switch unloads and reloads, and that the language
  mode is applied. This is the primary guard for "no reload per inference".
- **Unit seam — ModelCatalog**: maps each of the ten Models to its file and
  language mode.
- **Integration seam — on-device E2E** (the final acceptance gate): record a known
  English phrase and a known Roman-Urdu phrase, transcribe, and assert the
  clipboard transcript matches within a tolerable WER — the same behavioural test
  as Phase 1.
- **Device-verified, not unit-tested**: mic capture (`AudioRecord`) and the actual
  whisper.cpp inference via the AAR, verified on the phone.

## Out of Scope

- Native IME keyboard (later stretch goal; hold-to-record belongs there).
- Media-file transcription in v1.
- Automessaging / voice-send (ADR 0001; manual send only).
- Play Store publishing.
- Live/streaming transcription while speaking (phone CPU cannot keep up).

## Further Notes

### Model catalog (ten entries)

| # | Model | File (GGML) | Approx size | Language mode |
|---|-------|-------------|-------------|---------------|
| 1 | Roman-Urdu q4_0 (default) | `ggml-model-q4_0.bin` (small, q4_0) | 139 MB | auto |
| 2 | Roman-Urdu full | `ggml-model-f16.bin` (small, f16) | ~550 MB | auto |
| 3 | English full | `ggml-base.en.bin` | ~148 MB | en |
| 4 | English quantized | `ggml-base.en-q8_0.bin` | ~82 MB | en |
| 5 | Multilingual small q8_0 | `ggml-small-q8_0.bin` | ~264 MB | auto |
| 6 | Multilingual tiny full | `ggml-tiny.bin` | ~77 MB | auto |
| 7 | English tiny | `ggml-tiny.en.bin` | ~75 MB | en |
| 8 | English tiny q8_0 | `ggml-tiny.en-q8_0.bin` | ~42 MB | en |
| 9 | Tiny q5_1 (multilingual) | `ggml-tiny-q5_1.bin` | ~32 MB | auto |
| 10 | English tiny q5_1 | `ggml-tiny.en-q5_1.bin` | ~32 MB | en |

- Roman-Urdu = `cheetos18/whisper-small-roman-urdu` fine-tune converted to
  whisper.cpp GGML; must run with language auto-detect to stay in Roman/Latin
  script (do not force `ur`). See tickets 06 and 16–18, and CONTEXT.md language.
- Sources: 1–2 are conversions of the HF fine-tune, downloadable from
  `femustafa/voicedictation-models` (tickets 16–18); 3–10 are downloadable from
  the `ggerganov/whisper.cpp` HuggingFace repo.
- The default is Roman-Urdu q4_0. f16 is a higher-quality optional download, not
  the default (larger + slower on the phone).
- The default Model downloads on first launch; full offline thereafter.

### Tail calls to Termux Phase 1
- The native app replaces the Termux widget/`dictate.sh` flow. The Linux harness
  and benchmark tooling (`tools/bench/`, `tools/validate_stt/`) stay in use for
  model selection.

