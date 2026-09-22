# Kalaam Scribe v1.0.1

Patch release fixing small-phone layout bugs from v1.0.0. No feature changes.

## Fixes

- **Welcome screen overflow** — after picking a dictation language, the model list could push the Continue button below the fold on small phones. The screen now scrolls, so Continue is always reachable.
- **Transcript clipping** — a very long transcript could grow past the screen edge and push the Copy button out of view. The transcript body is now capped and scrolls inside its card, keeping Copy and history on-screen.

## Regression guards

Both bugs are now covered by JVM layout tests (Robolectric + Compose) that render the screens at small-, medium-, and large-phone sizes, so the fixes can't silently regress.

## Requirements

Same as v1.0.0: Android 8.0+ (API 26+), arm64-v8a. See `RELEASE_NOTES_v1.0.0.md`.

## Development

Build instructions and the signed-APK release process: see `android/SETUP.md` §4.5.