# 02: Termux environment (whisper.cpp) on Android phone

**What to build:** The on-phone runtime environment is set up: Termux (from F-Droid) with the Termux:API and Termux:Widget add-ons, build tools (`git cmake clang make`) plus ffmpeg, whisper.cpp built natively, the `base.en` GGML model downloaded (`ggml-base.en.bin`), and microphone permission granted. The phone is now capable of local transcription.

**Blocked by:** 01 (Linux dev harness) — confirms the model/settings before installing on the phone

**Status:** ready-for-agent

- [x] Termux + Termux:API + Termux:Widget installed from F-Droid on the phone
- [x] Build tools + ffmpeg installed; whisper.cpp built with `GGML_NO_OPENMP=ON` (stable on Android)
- [x] `base.en` GGML model downloaded and cached on the phone (usable offline)
- [x] Microphone permission granted to Termux

## Acceptance criteria notes
Done when Termux can run `whisper-cli` with `ggml-base.en.bin` on the phone without internet.

## Outcome (implemented)
- Termux, Termux:API, Termux:Widget installed from F-Droid.
- Packages installed: git, cmake, clang, make, ffmpeg, curl, termux-api.
- whisper.cpp cloned and built natively with `GGML_NO_OPENMP=ON`; `whisper-cli` present at `./build/bin/whisper-cli`.
- `ggml-base.en.bin` downloaded; transcription verified working offline on the phone.
- Mic permission granted to Termux:API (Termux itself has no mic entry — expected; recording delegates to the API app).
