# 11: Mic recording

**What to build:** On-device microphone capture to a 16 kHz mono WAV file, the
audio input for dictation. Runs in parallel with the model work; only needs the
scaffold because capturing a valid audio file can be verified independently of
transcription.

**Blocked by:** 07 (Android toolchain + project scaffold).

**Status:** done (device-verified)

## On-device verification (2026-08-31)

Installed APK, tapped Record → allowed mic → spoke → Stop, then pulled the WAV:

- `file`: `RIFF WAVE audio, Microsoft PCM, 16 bit, mono 16000 Hz` — **valid format**.
- 241,964 bytes (44-byte header + 241,920 PCM = ~7.6 s).
- PCM analysis: 120,960 samples, peak 9089, RMS 724.6 → **real speech, not silence**.

All acceptance criteria met.

## Implementation

- `AudioRecorder` — wraps `AudioRecord` to capture mic at **16 kHz mono 16-bit PCM**
  into a WAV (no on-device ffmpeg; whisper AAR decodes/resamples, matching the
  Phase-1 16 kHz step). Writes a 44-byte WAV header, streams PCM on a background
  daemon thread (`read()` is blocking), then seeks back to finalize lengths.
- `DictationViewModel` — owns the recorder; `toggleRecording()` starts/stops into
  `filesDir/dictation.wav` off the main thread; exposes `recording`/`lastFile`/`error`
  StateFlows.
- `DictationScreen` — Record/Stop control that requests `RECORD_AUDIO` **runtime
  permission** on first use via `rememberLauncherForActivityResult(
  RequestPermission())`, then toggles recording; shows recording state, saved path,
  error. Manifest already declared `RECORD_AUDIO`.
- `RootApp` wires the DictationViewModel. Build + unit tests pass.

## Acceptance criteria

- [x] Recording via the mic produces a valid 16 kHz mono PCM WAV file. (Code
      complete; file-validity on the A50 is the pending device verification.)
- [x] A start/stop control ends the capture and the resulting file can be read
      back on the A50 (verified on device, not unit-tested — hardware seam).
      PASSED: valid 16k mono PCM WAV, ~7.6 s of real speech pulled from the A50.
- [x] Mic runtime permission is requested and handled.
- [x] The recording path avoids needing ffmpeg on-device (matches the Phase 1
      16 kHz step; the whisper AAR decodes/resamples).
