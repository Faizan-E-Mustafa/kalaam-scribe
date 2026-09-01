# 19: Update spec + deploy docs for Roman-Urdu download

**What to build:** Bring the domain docs in line with the new reality: Roman-Urdu
GGML files are now hosted on HuggingFace and downloadable, q4_0 is the default
model, and f16 is an optional quality tier. Old references to q8_0/local-copy are
corrected.

**Blocked by:** 18 — Android app downloads Roman-Urdu at runtime (docs should
reflect the shipped behaviour), but can be drafted earlier since the model-table
facts are fixed by 16/17.

**Status:** ready-for-agent

- [ ] `spec.md` model table lists Roman-Urdu q4_0 as default (139 MB) and f16 (550 MB)
- [ ] `spec.md` notes the Roman-Urdu GGML is now downloaded from the new HF repo
- [ ] Any stale `q8_0` / "local conversion, copy manually" claims corrected
- [ ] Deploy ticket (#06) file references aligned to `ggml-model-q4_0.bin` (~139 MB)
