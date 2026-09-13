# 30: DolphinAttn — VAD-segmented per-utterance decode (pseudo-streaming)

**What to build:** Add Silero-VAD-based segmentation to `DolphinAttnEngine` so long
dictation clips are split into per-utterance segments, each beam-searched independently,
and the per-segment transcripts are joined. This lowers the worst-case single-decode time
and matches how sherpa-onnx practically handles Dolphin (whole-clip VAD → offline decode
per utterance) instead of one monolithic encode+decode of the whole clip.

**Blocked by:** 28/29 (the DakeQQ ONNX pair, graph surgery, beam logic, Android port all
done — this ticket reuses that engine loop verbatim). Not blocked by 26 (upstream sherpa
#3904). Does NOT depend on any upstream sherpa Dolphin streaming support (none exists).

**Status:** ready-for-agent

## Decisions (2026-09-05)

- **Approach: VAD per-segment decode** (not true chunked streaming of the encoder). The
  DakeQQ encoder is a fused non-streaming attention encoder (EBranchformer) that ticket 28
  found cannot even be torch-exported, let alone chunked for streaming — true incremental
  streaming is out of scope.
- **VAD source: sherpa-onnx AAR** (already pinned 1.13.4): `com.k2fsa.sherpa.onnx.Vad` /
  `VoiceActivityDetector` with `VadModelConfig` + `SileroVadModelConfig`.
- **VAD model artifact:** `silero_vad.onnx` sourced as a catalog download under
  `femustafa/voicedictation-models` (consistent with the app's existing downloader +
  filesDir pattern), with an asset-bundled fallback if we prefer zero pre-download.

## Scope callout (set expectations)

This does **NOT** reduce total latency for one continuous utterance — VAD segments sum to
roughly the same encoder+decode work, and audio still arrives as a whole clip (decode is
still batch, after Stop). It gives per-phrase results and a hard per-utterance decode
ceiling. True incremental streaming requires a chunked streamable encoder export, out of
scope here.

## Acceptance Criteria

- [x] `DolphinAttnEngine` splits the decoded clip into VAD-derived utterance segments and
      runs the existing beam search per segment, concatenating the transcripts.
- [x] Each segment is an independent decode: it gets its own `ur`/`PK` prefix, its own KV
      cache, and length-normalized beam selection (ticket-29 correctness preserved;
      `beamSearch` is called once per segment).
- [x] If VAD yields no segments (model missing / near-empty / engine error), fall back to
      the current single whole-clip decode (preserves today's behavior + the
      "No speech detected" semantics). Log which path ran. Extra guard: if every decoded
      segment is blank, also falls back to whole-clip.
- [x] VAD model is loaded once as resident (reusing the ADR-0004 resident-model pattern,
      not re-init per inference).
- [x] App builds; JVM unit tests green (`VadSegmentationTest`: windowing, drain loop,
      flush tail, empty-input — 5 tests, all pass via fake `VadLike`, no native lib).
- [ ] On-device RTF re-benchmarked on a long clip (>20 s), single-shot vs VAD-segmented,
      documented (expect lower per-segment decodeMs even if total wall-clock is similar).

## Implementation outline (revised 2026-09-05 — as built)

1. **`android/app/src/main/java/dev/femustafa/voicedictation/VadSegmentation.kt`** (new) —
   `SpeechUtterance` data class, `VadLike` seam (the sherpa `Vad` surface the loop needs),
   `SherpaVad` adapter wrapping `com.k2fsa.sherpa.onnx.Vad`, and `segmentAudioWithVad` —
   a faithful port of sherpa's canonical `VadNonStreamingDolphinCtc.java` loop: push
   512-sample windows, drain queued `SpeechSegment`s the moment `isSpeechDetected()`,
   then flush once and drain the tail.
2. **`DolphinAttnEngine.kt`** — `DolphinAttnModelRef` gains a resident `vad: Vad?`; `load()`
   initializes it from the `silero_vad.onnx` sibling (no-arg configs + setters, the
   1.13.4 style — **there is no `.builder()` factory in 1.13.4**, unlike current sherpa
   master) and degrades to `null` on any failure; `transcribe` → `transcribeClip` decodes
   per segment and joins with `" "`; `release()` releases the VAD.
3. **`ModelCatalog.kt`** — `SHERPA_SILERO_VAD` constant pointing at the official
   `github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx` (verified
   reachable, ~629 KB; same GitHub-release hosting the existing Dolphin CTC repos use).
4. **`ModelDownloader.kt`** — VAD is a third concurrent sibling in
   `ensureDolphinAttnSiblings` (reuses the existing `.part`+rename downloader) and is
   gated in `dolphinAttnComplete` via `vadFileName()`/`MIN_VAD_BYTES` so its presence
   drives the download.
5. **`VadSegmentationTest.kt`** (new) — 5 JVM tests with a fake `VadLike` (no native lib):
   empty input, 512-window+tail windowing, ordered segment collection,
   no-speech→empty, flush-surfaces-tail.

`decodeTokens` is unchanged (operates per segment); only the `transcribe` orchestration
changed (via `transcribeClip`).

## Correctness constraints (carry from ticket 29 — MUST NOT regress)

- Language pin `ur`/`PK` prefix **per segment**, not just the first.
- KV cache is **per segment**, never shared across segments.
- fp16 KV dtype, full-logits output (`/output_layer/Gemm_output_0`), `BEAM_SIZE=1`,
  length-normalized selection, layer/head/dim derived from the encoder — all unchanged.

## Open items to confirm during implementation

1. ~~VAD model source~~ **RESOLVED: sherpa GitHub-release URL** added to the catalog and
   downloaded as a dolphin-attn sibling — no HF repo upload needed, consistent with how
   the Dolphin CTC repos are hosted. Asset-bundling not taken.
2. ~~Exact sherpa 1.13.4 Java VAD API~~ **RESOLVED at compile time**: `Vad(AssetManager,
   VadModelConfig)` + no-setter-no-builder configs confirmed against the 1.13.4 API jar;
   classes `VadModelConfig`, `SileroVadModelConfig`, `Vad`, `SpeechSegment` all present.
   Note: **no `VoiceActivityDetector` class in 1.13.4** — the queue-based `Vad` API is the
   one that exists.
3. Interim per-segment partial results surfaced to the UI, or only final concatenated
   text (minimal: final text only, which is this app's native behavior). **Kept:** final
   concatenated text (open if we later want per-utterance streaming to the UI).

## Verification

- **Laptop**: extend/reuse `tools/dolphin-onnx/scripts/verify_dakeqq_beam.py` with a
  python `silero-vad` segmentation pass on real Urdu clips to lock expected per-segment
  transcripts before porting.
- **JVM unit tests** (no device) via the `WhisperEngine`/`WhisperModelRef` seam.
- **On-device (the phone)**: re-run the ticket-29 RTF benchmark on a long clip, single-shot vs
  VAD-segmented.

## Comments

- **2026-09-05**: Created from the streaming plan. sherpa-onnx does NOT provide streaming
  Dolphin (offline CTC only, per research); this ticket implements sherpa's practical
  Dolphin pattern = VAD → per-utterance offline decode. Scope confined to `DolphinAttnEngine`;
  ticket 22's stale "VAD via separate Vad class" note is superseded by this concrete work.
- **2026-09-05 (implementation)**: All code changes landed and verified:
  `VadSegmentation.kt`, `DolphinAttnEngine.kt` (resident VAD + `transcribeClip`), catalog
  constant, downloader sibling+gate, and `VadSegmentationTest` (5 tests).
  `./gradlew :app:testDebugUnitTest` green (32 tests total). `assembleDebug` APK builds.
  `lintDebug` reports only the **pre-existing** `AudioRecorder.kt:37` MissingPermission
  (untouched). On-device RTF benchmark on a >20 s clip (single-shot vs VAD-segmented) is
  the remaining acceptance item — needs the phone + installed APK.
