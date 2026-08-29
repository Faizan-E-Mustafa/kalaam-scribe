# 02: Termux environment (whisper.cpp) on Samsung A50

**What to build:** The on-phone runtime environment is set up: Termux (from F-Droid) with the Termux:API and Termux:Widget add-ons, build tools (`git cmake clang make`) plus ffmpeg, whisper.cpp built natively, the `base.en` GGML model downloaded (`ggml-base.en.bin`), and microphone permission granted. The phone is now capable of local transcription.

**Blocked by:** 01 (Linux dev harness) — confirms the model/settings before installing on the phone

**Status:** ready-for-agent

- [ ] Termux + Termux:API + Termux:Widget installed from F-Droid on the A50
- [ ] Build tools + ffmpeg installed; whisper.cpp built with `GGML_NO_OPENMP=ON` (stable on Android)
- [ ] `base.en` GGML model downloaded and cached on the phone (usable offline)
- [ ] Microphone permission granted to Termux

## Acceptance criteria notes
Done when Termux can run `whisper-cli` with `ggml-base.en.bin` on the A50 without internet.
