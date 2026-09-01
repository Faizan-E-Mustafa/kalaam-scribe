# 10: Model catalog + downloader

**What to build:** The model picker: a catalog of the six Models (each a GGML
file plus its language mode) and an in-app HuggingFace downloader with progress,
so the user can choose a Model and its language mode and have the app fetch it.

**Blocked by:** 08 (AAR dry-run spike on device).

**Status:** done

> Updated by ticket 18: Roman-Urdu entries are now hosted + downloadable
> (`roman-urdu-q4_0` is the default; `roman-urdu-q8` removed), and the catalog
> grew to ten Models (tiny variants added later).

## Implementation

- `ModelCatalog` (object) — the six-Model table in code: each `CatalogEntry`
  carries the domain `Model` (id, fileName, languageMode), displayName,
  `sourceUrl` (null for the two locally-converted Roman-Urdu models),
  approxSizeMb, isDefault. Default = Roman-Urdu q8_0 (`ModelCatalog.default`).
- `ModelDownloader` — HttpURLConnection (no new dependency) streaming into
  app **internal** storage (ticket-08 finding), temp-file + rename, progress
  callback (0f..1f), no size gating. Not unit-tested (network); device-verified.
- `ModelPickerViewModel` — per-model `DownloadState` (NotDownloaded /
  Downloading(progress) / Failed / Ready), persisted by file-existence in
  `filesDir` (recognised as Ready on later launches); `download()` drives the
  downloader; `select()` calls `WhisperManager.switchTo`.
- `ModelPickerScreen` (Compose) — lists the catalog with radio select, language
  mode + size + default badge, progress bar, Download/Retry buttons.
- Wired via `RootApp` (owns `WhisperManager` = AarWhisperEngine + filesDir, and
  the picker ViewModel); DictationScreen now shows the resident Model + a Models
  button. Spike button removed from the main flow (spike preserved in SpikeRunner).
- Tests: `ModelCatalogTest` (6) — six models, default = Roman-Urdu q8_0,
  auto-detect + non-hosted for RU, fixed `en` for English, public URLs point at
  whisper.cpp HF and end with fileName, unique ids. 6/6 pass.

## Acceptance criteria

- [x] The six-catalog Model table is represented in code: file, source URL,
      approx size, language mode, default flag.
- [x] Unit test: the catalog maps each Model entry to the correct file and
      language mode (ModelCatalog unit seam). 6/6 pass.
- [x] The picker UI lists the Models and lets the user select one and its language
      mode (auto / fixed), and shows download state (not downloaded / downloading
      with progress / ready).
- [x] Selecting a Model triggers a HuggingFace download with visible progress; no
      size gating. (Downloader device-verified; manual-copy path for RU models.)
- [x] A downloaded Model is persisted and recognised as ready on later launches
      (offline after first download).
- [x] Default selection on first launch is Roman-Urdu q8_0.
