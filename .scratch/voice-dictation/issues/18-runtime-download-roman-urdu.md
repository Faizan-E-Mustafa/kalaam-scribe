# 18: Android app downloads Roman-Urdu at runtime

**What to build:** The model picker offers two Roman-Urdu Models that the app can
download from HuggingFace at the tap of a button (with progress), rather than
requiring a manual copy. The quantized q4_0 model is the on-launch default; the
unquantized f16 model is an optional higher-quality choice.

**Blocked by:** 17 — Upload Roman-Urdu GGML models to HuggingFace (provides the
public URLs)

**Status:** ready-for-agent

- [ ] Both Roman-Urdu catalog entries carry a public `sourceUrl` (no longer `null`)
- [ ] The q4_0 Roman-Urdu entry is the on-launch default; English q8 is no longer default
- [ ] Picker's Download button fetches the Roman-Urdu model from HF with progress
- [ ] Unit tests updated to reflect the new catalog (model count, default, hosted URLs)
- [ ] `./gradlew test` green
