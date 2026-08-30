#!/data/data/com.termux/files/usr/bin/bash
# Termux:Widget home-screen shortcut: tap to start dictation.
# Recording runs until you tap "Stop" in the ongoing notification.
# NOTE (Phase 2): Termux:Widget cannot distinguish press-hold vs release, so this
# is tap-to-start / tap-Stop; true hold-to-record needs the native app. Also,
# the widget only fires while the Termux app process is alive — Android kills
# it in the background, so the native app (foreground service/IME) is required
# for a reliable launch surface.
export PATH="/data/data/com.termux/files/usr/bin:$PATH"
export HOME="/data/data/com.termux/files/home"
exec /data/data/com.termux/files/usr/bin/dictate start
