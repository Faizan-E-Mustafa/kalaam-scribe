# 03: Record-and-clip dictation script

**What to build:** A single command (and home-screen widget shortcut) on the A50 that records the microphone via Termux:API, transcribes locally with faster-whisper `base.en`, writes the transcript to the Android clipboard, and shows a "Copied" notification. The user stops recording without typing (on-screen tap or volume button).

**Blocked by:** 02 (Termux environment on A50)

**Status:** ready-for-agent

- [ ] Command records mic via termux-microphone-record with a hands-free stop
- [ ] Recording is transcribed locally with `base.en`
- [ ] Transcript is written to the clipboard via termux-clipboard-set
- [ ] A termux-notification tells the user it's ready to paste
- [ ] A home-screen widget/tile launches the flow in one tap

## Acceptance criteria notes
Done when a spoken English test phrase ends up on the A50 clipboard as text.
