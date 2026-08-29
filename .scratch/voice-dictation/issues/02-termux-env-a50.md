# 02: Termux environment on Samsung A50

**What to build:** The on-phone runtime environment is set up: Termux (from F-Droid) with the Termux:API and Termux:Widget add-ons, the Python packages and ffmpeg needed for faster-whisper, the `base.en` model downloaded and cached locally, and microphone permission granted. The phone is now capable of local transcription.

**Blocked by:** 01 (Linux dev harness) — confirms the model/packages work before installing on the phone

**Status:** ready-for-agent

- [ ] Termux + Termux:API + Termux:Widget installed from F-Droid on the A50
- [ ] Python/pip/faster-whisper/ffmpeg installed inside Termux
- [ ] `base.en` model downloaded and cached on the phone (usable offline)
- [ ] Microphone permission granted to Termux

## Acceptance criteria notes
Done when Termux can load faster-whisper and the `base.en` model on the A50 without internet.
