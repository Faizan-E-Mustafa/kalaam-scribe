# 03: Record-and-clip dictation script

**What to build:** A single command (and home-screen widget shortcut) on the A50 that records the microphone via Termux:API, transcribes locally with whisper.cpp `base.en` (`whisper-cli` on the GGML model), writes the transcript to the Android clipboard, and shows a "Copied" notification. The user stops recording without typing (on-screen tap or volume button).

**Blocked by:** 02 (Termux environment on A50)

**Status:** ready-for-agent (in progress; file-transcription path verified on A50)

- [ ] Command records mic via termux-microphone-record with a hands-free stop
- [x] Recording/content is transcribed locally with `base.en` (file path verified on A50)
- [x] Transcript is written to the clipboard via termux-clipboard-set (verified)
- [x] A termux-notification tells the user it's ready to paste (verified)
- [ ] A home-screen widget/tile launches the flow in one tap

## Acceptance criteria notes
Done when a spoken English test phrase ends up on the A50 clipboard as text.

## Delivered
- `tools/dictate/dictate.sh` — Bash script for Termux:
  - `dictate start` — records mic (`termux-microphone-record -f ...m4a -l 0`) and posts an
    ongoing "Recording… tap Stop" notification with a Stop button.
  - `dictate stop` — stops recording, ffmpeg → 16 kHz mono, `whisper-cli -m ggml-base.en.bin`,
    writes transcript to clipboard via `termux-clipboard-set`, shows high-priority "Copied"
    notification.
  - `dictate <file>` — transcribe an existing audio file to the clipboard (for testing).
- On-phone steps to verify + widget setup are given to the user (below).
