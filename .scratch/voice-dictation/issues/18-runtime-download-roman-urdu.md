# 18: Android app downloads Roman-Urdu at runtime

**What to build:** The model picker offers two Roman-Urdu Models that the app can
download from HuggingFace at the tap of a button (with progress), rather than
requiring a manual copy. The quantized q4_0 model is the on-launch default; the
unquantized f16 model is an optional higher-quality choice.

**Blocked by:** 17 — Upload Roman-Urdu GGML models to HuggingFace (provides the
public URLs)

**Status:** done

## Implementation

- `ModelCatalog`: added `RU_HF` const; the two Roman-Urdu entries now have
  `sourceUrl` pointing at `femustafa/voicedictation-models`; `roman-urdu-q4_0`
  (`ggml-model-q4_0.bin`, 139 MB, Auto) is now `isDefault = true`; English q8
  flipped to `isDefault = false`. Removed the old `roman-urdu-q8` entry
  (superseded by q4_0).
- `Model.isRomanUrdu()`: now matches `roman-urdu-q4_0` + `roman-urdu-f16`.
- Tests updated: `ModelCatalogTest` (default = RU q4_0; RU entries hosted and
  auto-detect; 10 hosted entries split 8 whisper.cpp + 2 RU_HF),
  `ModelTest` (RU ids), `WhisperManagerTest` fixture id.
- Picker/`ModelPickerViewModel` needed no change: `download()` already handles
  any entry with a non-null `sourceUrl`, and `defaultOrPersistedId()` falls back
  to the catalog default when the default file is missing.

## Verify

- [x] Both Roman-Urdu catalog entries carry a public `sourceUrl` (no longer `null`)
- [x] The q4_0 Roman-Urdu entry is the on-launch default; English q8 is no longer default
- [x] Picker's Download button fetches the Roman-Urdu model from HF with progress
- [x] Unit tests updated to reflect the new catalog (model count, default, hosted URLs)
- [x] `./gradlew testDebugUnitTest` green — 18/18 tests pass (JAVA_HOME = `~/android-tooling/jdk-17.0.20.1+1`)
