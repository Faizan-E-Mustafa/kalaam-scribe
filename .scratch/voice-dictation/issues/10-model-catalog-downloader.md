# 10: Model catalog + downloader

**What to build:** The model picker: a catalog of the six Models (each a GGML
file plus its language mode) and an in-app HuggingFace downloader with progress,
so the user can choose a Model and its language mode and have the app fetch it.

**Blocked by:** 08 (AAR dry-run spike on device).

**Status:** ready-for-agent

## Acceptance criteria

- [ ] The six-catalog Model table (see spec.Notes) is represented in code: file,
      source URL, approx size, language mode, default flag.
- [ ] Unit test: the catalog maps each Model entry to the correct file and
      language mode (the ModelCatalog unit seam).
- [ ] The picker UI lists the Models and lets the user select one and its language
      mode (auto / fixed), and shows download state (not downloaded / downloading
      with progress / ready).
- [ ] Selecting a Model triggers a HuggingFace download with visible progress; no
      size gating.
- [ ] A downloaded Model is persisted and recognised as ready on later launches
      (offline after first download).
- [ ] Default selection on first launch is Roman-Urdu q8_0.
