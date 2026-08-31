# 11: Mic recording

**What to build:** On-device microphone capture to a 16 kHz mono WAV file, the
audio input for dictation. Runs in parallel with the model work; only needs the
scaffold because capturing a valid audio file can be verified independently of
transcription.

**Blocked by:** 07 (Android toolchain + project scaffold).

**Status:** ready-for-agent

## Acceptance criteria

- [ ] Recording via the mic produces a valid 16 kHz mono PCM WAV file.
- [ ] A start/stop control ends the capture and the resulting file can be read
      back on the A50 (verified on device, not unit-tested — hardware seam).
- [ ] Mic runtime permission is requested and handled.
- [ ] The recording path avoids needing ffmpeg on-device (matches the Phase 1
      16 kHz step; the whisper AAR decodes/resamples).
