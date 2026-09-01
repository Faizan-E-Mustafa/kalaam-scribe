# 19: Update spec + deploy docs for Roman-Urdu download

**What to build:** Bring the domain docs in line with the new reality: Roman-Urdu
GGML files are now hosted on HuggingFace and downloadable, q4_0 is the default
model, and f16 is an optional quality tier. Old references to q8_0/local-copy are
corrected.

**Blocked by:** 18 — Android app downloads Roman-Urdu at runtime (docs should
reflect the shipped behaviour), but can be drafted earlier since the model-table
facts are fixed by 16/17.

**Status:** done

## Changes

- `spec.md` — model catalog bullet + table now list **ten** models (added the
  four tiny variants that were previously missing), with Roman-Urdu **q4_0** as
  the default (139 MB) and f16 as the optional quality tier. Sources note:
  rows 1–2 downloadable from `femustafa/voicedictation-models`; 3–10 from the
  whisper.cpp repo. Added a note that f16 is larger/slower and not the default.
- `spec.md` — "six Models" references corrected to ten (two places).
- Deploy ticket #06 — `ggml-model-q8_0.bin` (251 MB) → `ggml-model-q4_0.bin`
  (139 MB) in What-to-build, validation, steps, and bench note; added the
  downloadable-source hint.
- `issues/15` — `isRomanUrdu` ids note now mentions q4_0 superseding q8 (ticket 18).
- `issues/10` — added a forward-note that ticket 18 changed RU hosting/default.
- `ModelCatalogTest` doc comment — dropped stale "six Models" phrase.

## Verify

- [x] `spec.md` model table lists Roman-Urdu q4_0 as default (139 MB) and f16 (550 MB)
- [x] `spec.md` notes the Roman-Urdu GGML is now downloaded from the new HF repo
- [x] Any stale `q8_0` / "local conversion, copy manually" claims corrected
- [x] Deploy ticket (#06) file references aligned to `ggml-model-q4_0.bin` (~139 MB)
