# Kalaam Scribe v1.0.2

Patch release adding a scrollbar and character count to the transcript card, and
making the copied notification show the end of a dictation. No model, engine, or
accuracy changes.

## Improvements

- **Visible transcript scrollbar** — the transcript card is a fixed-height window that
  auto-scrolls to the newest text. That is right for proving the tail landed, but
  nothing told you the text was long or that the card scrolls at all. An
  always-visible track-and-thumb scrollbar now runs down the right edge whenever the
  text overflows, and disappears when it fits. The thumb's length shows how much is
  visible; its position shows where you are, sitting at the bottom when auto-scrolled
  to the end.
- **Character count** — the transcript footer reads
  `Transcribed in 3.4 s · 2,399 chars`, and the copy confirmation says how many
  characters it copied.
- **Copied notification shows the tail** — the collapsed notification only has room for
  about two lines, and it used to show the *start* of the transcript, which is already
  visible on screen. It now shows the last 140 characters, so a glance at the
  notification confirms the dictation finished with real words. The expanded
  notification still carries the full text.

## Regression guards

The scrollbar is covered by JVM layout tests (Robolectric + Compose): present for a
2,399-character transcript, absent for a short one, across small-, medium-, and
large-phone sizes. The existing Copy and Full-text layout regressions stay green.

## Requirements

Same as v1.0.0: Android 8.0+ (API 26+), arm64-v8a. Installs over v1.0.1 in place.
See `RELEASE_NOTES_v1.0.0.md`.

## Development

Build instructions and the signed-APK release process: see `android/SETUP.md` §4.5.
