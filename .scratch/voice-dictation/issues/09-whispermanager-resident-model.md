# 09: WhisperManager (resident model lifecycle)

**What to build:** The core model-lifecycle component: a manager that holds one
resident Model loaded in memory and reuses it across repeated dictations, so a
model file is never reloaded per inference. Reloads only on Switch (unload then
load) or cold start, and applies the Model's language mode. This is the primary
guard for the "no reload per dictation" requirement (ADR 0004).

**Blocked by:** 08 (AAR dry-run spike on device).

**Status:** ready-for-agent

## Acceptance criteria

- [ ] Unit test: repeated transcribe calls reuse the resident Model (load count
      stays 1, not per-inference).
- [ ] A `switchTo(otherModel)` unloads the current Model and loads the new one,
      reflected in the load count.
- [ ] The configured language mode (auto, `en`, or explicit) is applied on
      transcribe, including the Roman-Urdu auto-detect path.
- [ ] The audio pipeline is serialized: a dictation cannot start while a
      transcription is running.
- [ ] Unit tests run on the dev machine without a device (the WhisperManager
      unit seam).
