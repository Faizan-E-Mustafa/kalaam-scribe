# 03: Record-and-clip dictation script

**What to build:** A single command (and home-screen widget shortcut) on the phone that records the microphone via Termux:API, transcribes locally with whisper.cpp `base.en` (`whisper-cli` on the GGML model), writes the transcript to the Android clipboard, and shows a "Copied" notification. The user stops recording without typing (on-screen tap or volume button).

**Blocked by:** 02 (Termux environment on phone)

**Status:** done (mic flow + clipboard + notify verified on phone; widget = partial, see finding)

- [x] Command records mic via termux-microphone-record with a hands-free stop
- [x] Recording/content is transcribed locally with `base.en` (file + mic paths verified on phone)
- [x] Transcript is written to the clipboard via termux-clipboard-set (verified)
- [x] A termux-notification tells the user it's ready to paste (verified)
- [x] A home-screen widget/tile launches the flow in one tap (works while Termux open; see finding)

## Acceptance criteria notes
Done when a spoken English test phrase ends up on the phone clipboard as text. Achieved.

## Delivered
- `tools/dictate/dictate.sh` — Bash script for Termux:
  - `dictate start` — records mic (`termux-microphone-record -f ...m4a -l 0`) and posts an
    ongoing "Recording… tap Stop" notification with a Stop button.
  - `dictate stop` — stops recording, ffmpeg → 16 kHz mono, `whisper-cli -m ggml-base.en.bin`,
    writes transcript to clipboard via `termux-clipboard-set`, shows high-priority "Copied"
    notification.
  - `dictate <file>` — transcribe an existing audio file to the clipboard (for testing).
- `tools/dictate/shortcuts/dictate-start.sh` — Termux:Widget one-tap launcher.
- On-phone steps to verify + widget setup are given to the user.

## Finding (rough edge, feeds Phase 2)
The Termux:Widget shortcut **only fires while the Termux app process is alive**; Android
(esp. the phone's 4GB RAM) kills Termux in the background so a home-screen tap fails when the
app is closed. Workaround for the prototype: launch `dictate start` from within an open Termux
session. This strengthens the case that the Phase 2 native app (foreground service / IME) is
required, not just nice-to-have. Also, Termux:Widget cannot do hold-to-record; that too is a
Phase 2 native-app feature.
