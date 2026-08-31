# 09: WhisperManager (resident model lifecycle)

**What to build:** The core model-lifecycle component: a manager that holds one
resident Model loaded in memory and reuses it across repeated dictations, so a
model file is never reloaded per inference. Reloads only on Switch (unload then
load) or cold start, and applies the Model's language mode. This is the primary
guard for the "no reload per dictation" requirement (ADR 0004).

**Blocked by:** 08 (AAR dry-run spike on device).

**Status:** done

## Implementation

- Added a `WhisperEngine` seam + `WhisperModelRef` opaque handle so the manager
  is JVM-unit-testable without the AAR/device. Production impl `AarWhisperEngine`
  wraps `dev.ffmpegkit.whisper.Whisper` (loadModel/transcribe/releaseModel), mapping
  `LanguageMode.Auto -> "auto"` (Roman-Urdu auto-detect, never force `ur`) and
  `English -> "en"`.
- `WhisperManager(baseDir, engine)` guards residency: loads once on first use /
  cold start, reuses the resident Model across repeated transcribes, reloads only
  on `switchTo` (unload then load), and `shutdown()` frees it. Audio pipeline is
  serialized with a dedicated `Mutex`; `switchTo` waits for any in-flight
  transcription so a Model is never unloaded mid-run.
- `Model` data class + `LanguageMode` enum (Auto/English) named per CONTEXT.md.
- Tests: `WhisperManagerTest` (fake engine, 4 tests) — reuse/no-reload, switch
  unloads+loads, language mode applied, serialization. Run via
  `./gradlew :app:testDebugUnitTest` (no device). 4/4 pass.
- Added test dep `kotlinx-coroutines-test:1.9.0`.

## Acceptance criteria

- [x] Unit test: repeated transcribe calls reuse the resident Model (load count
      stays 1, not per-inference).
- [x] A `switchTo(otherModel)` unloads the current Model and loads the new one,
      reflected in the load count.
- [x] The configured language mode (auto, en) is applied on transcribe, including
      the Roman-Urdu auto-detect path.
- [x] The audio pipeline is serialized: a dictation cannot start while a
      transcription is running.
- [x] Unit tests run on the dev machine without a device (the WhisperManager
      unit seam).
