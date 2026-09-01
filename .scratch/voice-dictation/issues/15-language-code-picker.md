# 15: Language-code picker for multilingual models

**What to build:** On the main dictation screen, let the user pick a
whisper-supported language for **multilingual (auto-detect)** Models. The chosen
code is passed to `WhisperConfig.language` at transcribe time and persisted
across launches. Empty selection = auto-detect.

**Design (user, 2026-09-01):**
- **Dropdown only** — an `ExposedDropdownMenuBox` listing the 100
  whisper-supported languages (`openai/whisper` `tokenizer.py` `LANGUAGES`:
  code + name). No free typing, so no invalid-input path.
- **Single global persisted pref** (`SharedPreferences` `language_code`) —
  applies to whichever multilingual Model is active. Null/blank = `auto`.
- **Always visible at top, state-aware (user refinement):** the dropdown is shown
  on the main dictation screen at all times. When the active Model is
  multilingual (non-Roman-Urdu) it is enabled and shows the selection /
  "Auto-detect". When the active Model cannot use an override, it is **disabled**
  and shows the *effective* value: `English (fixed)` for English-only Models and
  `Auto-detect (fixed)` for Roman-Urdu. Roman-Urdu must never force `ur` (must
  stay roman/auto); English Models are fixed `en` (CONTEXT.md).

**Blocked by:** 13 (Foreground service — the service calls `whisper.transcribe`,
and the manager supplies the language).

**Status:** in progress (implemented, build + unit tests pass; awaiting A50 on-device verify)

## Acceptance criteria

- [ ] On the dictation screen, the language dropdown is always visible at the top;
      when the active Model is multilingual (non-Roman-Urdu, non-English) it is
      enabled and lists each language's name + 2-letter code plus "Auto-detect".
- [ ] When the active Model is English-only or Roman-Urdu, the dropdown is
      disabled and shows the effective value (`English (fixed)` / `Auto-detect
      (fixed)`); it cannot be opened.
- [ ] Selecting a language changes the code passed to the engine:
      `WhisperConfig.language = <code>`; "Auto-detect" → `"auto"`. English (`en`)
      and Roman-Urdu (`auto`, never `ur`) Models are unaffected.
- [ ] The choice persists across app restarts (SharedPreferences).
- [ ] Empty/unset selection means auto-detect.

## Implementation notes

- `Model.kt` — `WhisperLanguages` object: 100 `(code, name)` pairs + lookup;
  `Model.canOverrideLanguage` helper (Auto && !Roman-Urdu) with
  `isRomanUrdu` ids.
- `WhisperEngine.kt` — `transcribe(model, audioPath, languageMode, language: String?)`.
- `AarWhisperEngine.kt` — `WhisperConfig(language = language ?: languageMode.whisperLanguage)`.
- `WhisperManager.kt` — `languageCode: StateFlow<String?>` + `setLanguageCode`,
  resolved in `doTranscribe` (English/Roman-Urdu → always null).
- `VoiceDictationApp.kt` — owns `SharedPreferences`; loads on init, persists on set.
- `DictationViewModel.kt` — expose `languageCode` + `setLanguageCode`.
- `DictationScreen.kt` — `ExposedDropdownMenuBox` shown always at top; enabled only
  when `currentModel.canOverrideLanguage`, disabled otherwise with a fixed label
  (`English (fixed)` / `Auto-detect (fixed)`). Expansion is gated by `enabled`
  (this material3 version has no `enabled` param on `ExposedDropdownMenuBox`).
- Tests: `WhisperManagerTest` (update FakeEngine; override-to-multilingual,
  not-to-English/Roman-Urdu, blank→auto); new `ModelTest` for `WhisperLanguages`
  (100 entries) + `canOverrideLanguage`.

## Verification

`./gradlew :app:testDebugUnitTest :app:assembleDebug`; reinstall on A50; pick a
multilingual Model → choose a language from the dropdown → record → confirm the
selected language drives transcription.

## Implementation (2026-09-01)

- `Model.kt` — added `WhisperLanguages` (100 `code`→`name` pairs, matching
  `openai/whisper` `LANGUAGES` exactly), `Model.canOverrideLanguage` (Auto &&
  !Roman-Urdu), and `isRomanUrdu(modelId)` for ids `roman-urdu-q8`/`roman-urdu-f16`.
  Note: whisper's `LANGUAGES` dict is **100** codes (`num_languages=99` is a
  separate tokenizer truncation default, not the dict size).
- `WhisperEngine.kt` / `AarWhisperEngine.kt` — `transcribe(...)` takes an extra
  `language: String?`; `WhisperConfig(language = language ?: languageMode.whisperLanguage)`.
- `WhisperManager.kt` — `languageCode: StateFlow<String?>`, `setLanguageCode(code)`,
  and in `doTranscribe` the override is applied only when
  `residentModel.canOverrideLanguage` (English/Roman-Urdu always pass null).
- `VoiceDictationApp.kt` — owns `SharedPreferences` (`language_code`), restores on
  init, persists on `setLanguageCode`.
- `DictationViewModel.kt` — exposes `languageCode` + `setLanguageCode`.
- `DictationScreen.kt` — an `ExposedDropdownMenuBox` (Material3, `@OptIn`) shown
  **always at the top**, listing "Auto-detect" + the 100 languages; pulled from
  the `canOverrideLanguage` gate (user refinement) and now disabled with an
  effective-value label for English/Roman-Urdu Models. Selection persists.
- Tests: `WhisperManagerTest` (updated FakeEngine signature; added
  override-to-multilingual, not-to-English/Roman-Urdu, blank→auto) and new
  `ModelTest` (`WhisperLanguages`=100, `canOverrideLanguage` rules, `isRomanUrdu`).
  18 unit tests pass.
- On-device verification pending (device not connected to adb at commit time).
