# Spec: Local Voice Dictation for WhatsApp

Status: ready-for-agent

## Problem Statement

The user wants to send WhatsApp messages by speaking instead of typing. Chatting on
a phone is slow, and existing dictation either uploads audio to the cloud or is
bundled into third-party keyboards. They want dictation that runs entirely on their
Samsung A50 with no internet, no cloud, and no WhatsApp API, so their voice never
leaves the device.

## Solution

A fully on-device dictation flow. The user taps a recorder, speaks a message, and
the transcript is copied to the Android clipboard ready to paste into WhatsApp. The
first version runs as a Termux prototype on the A50 to validate speed and accuracy;
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
10. As a developer, I want to benchmark transcription speed/accuracy on the A50, so that I can pick the best model size.
11. As a developer, I want to validate the model on a fast Linux machine first, so that I de-risk the phone setup.

## Implementation Decisions

- Use **whisper.cpp** (GGML/C++ port) as the on-phone transcription engine, running on CPU only — it builds natively in Termux without the ctranslate2 source-compile that faster-whisper requires (ADR 0002). The Linux validation harness keeps **faster-whisper**.
- Phase 1 uses the **`base.en`** model in a quantized form with voice-activity detection, chosen for the A50's CPU-only constraints. Other sizes (`small`, `tiny`) are compared in the benchmark ticket.
- Runtime target is the **phone only**; the Linux desktop is a development/validation harness only.
- Recording uses **Termux:API** (`termux-microphone-record`); the transcript is written to the Android clipboard via `termux-clipboard-set` and signalled via `termux-notification`.
- The interaction is: tap recorder widget → speak → tap/volume to stop → wait for local transcription → "Copied" notification → user long-press-pastes into WhatsApp → user sends manually.
- **Manual send only.** No auto-Enter/automessaging in Phase 1.
- **Phase 2 (stretch goal):** a native Android IME keyboard app that inserts text directly into the focused field (no paste), reusing the validated model. Not specced/ticketed now. The native app is where **hold-to-record** (press = record, release = stop) belongs — Termux:Widget cannot distinguish press-hold from release, so the Phase 1 Termux prototype uses tap-to-start / tap-Stop instead. The native app is also **required for a reliable launch surface**: on Android (esp. the A50's 4GB RAM) the Termux widget only fires while the Termux process is alive and is killed in the background, so Phase 2 needs a foreground service / proper IME to start recording reliably from the home screen.
- **Multilingual support** is deferred; Phase 1 is English-only.

## Testing Decisions

- The key acceptance test is behavioral and end-to-end: record a known English phrase on the A50, transcribe, and confirm the text in the clipboard matches the phrase (within a tolerable WER).
- Benchmark ticket measures transcription time vs. audio length and qualitative accuracy on the A50, comparing `base`/`small`/`tiny` to choose the default.
- A Linux harness test confirms the `base.en` model loads and transcribes a known audio sample, catching environment/wheel problems before touching the phone.
- No unit-test framework is imposed on the Termux shell layer; verification is via the end-to-end behaviour above plus the benchmark.

## Out of Scope

- WhatsApp API integration and automessaging (rejected by ADR 0001).
- Native Android IME keyboard app (Phase 2, stretch goal, unticketed).
- Multilingual transcription.
- Live/streaming transcription while speaking (the A50 CPU cannot keep up).
- Auto-insert into a focused field in Phase 1 (the paste step is accepted).

## Further Notes

- Primary risk: `faster-whisper`/`ctranslate2` wheels on the developer's Python 3.14 Linux environment may not exist (worked around in the linux-dev-harness ticket with uv-managed Python 3.11). On the phone, ctranslate2 has no aarch64 wheel, which is why the phone uses whisper.cpp (ADR 0002) instead of faster-whisper.
- Secondary risk: A50 transcription speed; the benchmark ticket exists to de-risk and choose the model size.
