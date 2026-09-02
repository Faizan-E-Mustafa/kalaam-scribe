# TICKET 26: Track upstream Dolphin language/region support (sherpa-onnx #3904)

## Goal
Track `k2-fsa/sherpa-onnx#3904` — "Dolphin: expose language and region tokens" — and apply
the fix here when it lands. This is the "other model" gap from the Whisper language fix
(ticket 22): Whisper can pin a user-picked language, Dolphin CTC cannot.

## Acceptance Criteria
- [ ] Recheck sherpa-onnx releases when the upstream issue ships (see [Status] note below)
- [ ] `OfflineDolphinModelConfig` gains `language` / `region` fields in the Java API we use
- [ ] `DolphinCtcEngine.transcribe()` passes a user-picked language code (e.g. `"ur"`) to
      the recognizer the same way `SherpaWhisperEngine` does via `setConfig()`
- [ ] Urdu/other-language dictation on Dolphin CTC models no longer re-guesses script per
      segment (no cross-script leakage like the Arabic→Japanese / Filipino→CJK reports)
- [ ] `lang` field in the result is populated when the model provides it

## Status: needs-triage

## Context
- Upstream issue: https://github.com/k2-fsa/sherpa-onnx/issues/3904 (opened 2026-08-28; open,
  unassigned, no PR). Scope is two pieces:
  1. Export Dolphin's attention decoder to ONNX beside `model.onnx`.
  2. Add `language`/`region` to `OfflineDolphinModelConfig`, prepend `<lang><region>` tokens
     to the decoder prompt, surface through C/C++/bindings.
- Previously closed as unsupported: #2293, #2587, #2963. #3904 answers the
  "advantages/disadvantages" question from #2293 with concrete numbers and a repro
  (6-second Philippine segments getting CJK chars).
- Existing gap in this app: `OfflineDolphinModelConfig` has only a `model` field, no language
  (confirmed while fixing Whisper in ticket 22). Feature tracked upstream in #3904.
- Workaround while open: Dolphin CTC currently auto-detects per decode; we deliberately
  excluded Dolphin from the ticket-22 Whisper fix. If upstream is slow, an intermediate
  mitigation could be post-filtering decoder output by script (last resort; fails for
  same-script pairs like Malay/Indonesian).

## Current app state (fixed for Whisper, not Dolphin)
- `SherpaWhisperEngine`: user-picked language passed via `recognizer.setConfig()` before each
  decode (ticket 22). Works for Whisper only.
- `DolphinCtcEngine`: out of scope for the Whisper fix; no language config available.
- `WhisperManager.doTranscribe()` already resolves a nullable `language` from the Model's
  `canOverrideLanguage` and passes it to `engine.transcribe()`, so the plumbing is ready once
  the engine can accept it.

## Acceptance check cadence
- Revisit when sherpa-onnx tags a release after the #3904 decoder-export + config work lands,
  or when the Java binding exposes `OfflineDolphinModelConfig.setLanguage()`/`setRegion()`.

## Dependencies
- Upstream: k2-fsa/sherpa-onnx#3904 (decoder export + config/binding changes)
- App: `DolphinCtcEngine.kt`, `Model.kt` (`canOverrideLanguage` logic for Dolphin models),
  `ModelCatalog.kt` (`dolphinCtcModels`)

## Labels
dolphin-ctc, sherpa-onnx, language, upstream-tracker, urdu
