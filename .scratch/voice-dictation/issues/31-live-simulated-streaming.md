# 31: Live simulated streaming — per-utterance transcription while recording (sherpa + DolphinAttn)

**What to build:** Move the VAD loop from "run at Stop over the whole clip" into the live
capture path so partial text appears on screen **while the user is still recording** —
sherpa-onnx's **simulated streaming ASR** pattern (text lands at each utterance boundary as
VAD closes a segment, not word-by-word). Enabled for every sherpa-onnx-backed model
(**ONNX Whisper** in `SherpaWhisperEngine`, **Dolphin CTC** in `DolphinCtcEngine`) **and**
the standalone **Dolphin attention** model in `DolphinAttnEngine`. The **GGML whisper.cpp**
backend (`AarWhisperEngine`) stays batch-only — its AAR has no incremental API, so it keeps
today's stop-then-transcribe path.

**Blocked by:** 30 (the `Vad`/`VadLike` seam + per-utterance offline decode core it added;
this ticket reuses `segmentAudioWithVad`/`SherpaVad` and the resident-VAD wiring). Builds on
21/22 (sherpa integration + sherpa VAD notes). Not blocked by anything else.

**Status:** ready-for-human (implementation + JVM tests done; final on-device A50
verification below is a human step)

## Decisions (2026-09-05, before implementation)

- **Pattern: sherpa "simulated streaming ASR".** Feed mic samples to the sherpa `Vad` as they
  are captured; the moment VAD closes a `SpeechSegment` (end-of-speech detected), decode that
  segment offline on a background worker and publish its text. This is exactly how sherpa's
  own `SherpaOnnxSimulateStreamingAsr` demo (Whisper tiny, Dolphin-base APKs) behaves, and it
  is the **only** streaming a non-streaming model can get: `OfflineRecognizer`/`OfflineStream`
  is strictly batch, and the fused DolphinAttn encoder cannot be chunked.
- **NO mid-utterance partials.** sherpa has no incremental-decode API for these models;
  re-decoding a growing utterance is a custom hack (quadratically-growing cost) and is out of
  scope. Surface point = **completed utterances only**.
- **Scope of backends:** sherpa Whisper ONNX + Dolphin CTC (both are sherpa
  `OfflineRecognizer`s → a shared session helper) and DolphinAttn (custom ORT beam search →
  a parallel session implementation). GGML → `createSession()` returns null; service falls back
  to the current batch path.
- **Threading (hard rule):** `Vad.acceptWaveform` runs only on the capture thread (cheap); the
  recognizer / ORT sessions run only on ONE decode worker; capture must never block on decode.
  This mirrors ticket 30's VAD usage scaled to live.
- **`maxSpeechDuration = 5.0s` stays** — it becomes the live phrase splitter. Flagged as an
  open item (below): raising it lengthens live phrases but raises the per-utterance decode
  ceiling on the A50.
- **Session lifetime:** the session borrows the resident model for the whole recording; it is
  created at recording start (after the model is guaranteed resident) and finalized at Stop.
  Indivisible with `switchTo`/`shutdown` during a recording (not reachable in the UI).
- **Fallback preserved:** if the session yields zero segments (e.g. near-silence), `flush()`
  falls back to a whole-clip decode of the accumulated audio (DolphinAttn already has that
  path; sherpa engines gain it). No streaming session (GGML / creation failure / no model) →
  service runs today's `whisper.transcribe(wav)` untouched.

## Scope callout (set expectations)

- Text appears **sentence-by-sentence** as you pause, not word-by-word. Each phrase costs its
  normal decode (~1–1.5 s on the A50 for a few tokens) after VAD closes it — perceived
  latency ~ one phrase's decode, instead of everything at Stop.
- Same weights, same decoder → **final text is identical to today's** (VAD segmentation +
  join). This is a UX/latency change, not an accuracy change.
- WAV is still written and used for the no-speech fallback + consistency with the batch path;
  the notification's live text is left out of scope (can follow up).

## Acceptance Criteria

- [x] `WhisperEngine` gains a streaming seam (`createSession(...): TranscriptionSession?` +
  a `TranscriptionSession` interface: non-blocking `accept(samples)`, ordered
  `partials` flow, `suspend flush(): String`, `close()`). GGML returns null.
- [x] Live partials: while `recording`, each completed VAD utterance's text is published and
  the on-screen transcript grows as the user speaks; DolphinAttn, sherpa Whisper ONNX, and
  Dolphin CTC all show this.
- [x] Stop = `flush()`: VAD tail utterance decoded, final text joined, copy/notify/history
  run exactly as today on the final transcript.
- [x] DolphinAttn correctness (ticket 29, MUST NOT regress): per-segment `ur`/`PK` prefix,
  per-segment KV cache, fp16 KV, length-normalized beam — unchanged.
- [x] Threading: VAD touched only by the capture thread; recognizer/ORT sessions only by the
  single decode worker; mic capture never blocks on decode. **Segment queue is unbounded
  (`Channel.UNLIMITED`)** — decided after the drop-at-flush finding (see Comments), replacing
  the original "bounded segment queue OK" wording.
- [x] Fallback: GGML models and "session could not start" run today's whole-clip batch
  `transcribe`; zero segments → whole-clip decode of accumulated audio → blank text keeps the
  existing "No speech detected" handling.
- [x] Unit tests (JVM, fake `VadLike`, no native lib): the live drainer emits segments
  immediately per window + on `flush` tail; `createSession` returns null for a
  non-streaming engine. `:app:testDebugUnitTest` stays green (**53 tests today**, incl. the
  new 4-case `StreamingSessionNoDropTest`); `assembleDebug` builds; `lintDebug` adds no new
  errors (the pre-existing `AudioRecorder.kt:45` MissingPermission stays).
- [ ] On-device (A50): dictating a multi-phrase clip shows partials appearing per phrase while
  recording; Stop's final text equals the ticket-30 batch result for the same audio; log the
  per-phrase publish latency. **(pending — human/device step)**

## Implementation outline

1. **`WhisperEngine.kt`** — add `TranscriptionSession` interface (`accept(ShortArray)`,
   `partials: kotlinx.coroutines.flow.SharedFlow<String>`, `suspend flush(): String`,
   `close()`) and `suspend fun createSession(model, languageMode, language):
   TranscriptionSession?` (null = this backend cannot stream).
2. **`DualFormatWhisperEngine`** — route `createSession` to the owning backend; GGML → null;
   DolphinAttn/ONNX/DolphinCTC create their session. Holds a session-open guard so
   `switchTo`/`shutdown` are rejected while a session is live.
3. **`AarWhisperEngine`** — `createSession` returns null (whisper.cpp AAR stays batch).
4. **`SherpaOfflineSession`** (new, shared by sherpa Whisper + Dolphin CTC) — wraps a
   sherpa `OfflineRecognizer` + a per-session `SherpaVad`; `accept()` feeds 512-window chunks
   and enqueues closed `SpeechSegment`s to an **unbounded `Channel`** (`Channel.UNLIMITED`);
   the decode worker createStream→`acceptWaveform`→`decode`→`getResult`→release per segment
   and emits onto `partials`; `flush()` = `vad.flush()` + **close the channel then
   `worker.join()`** (so every queued segment, live and tail, is decoded — nothing dropped on
   a slow worker) + return joined text; zero segments → fall back to whole-clip decode of
   accumulated audio. Owns a per-session `CoroutineScope`; `close()` cancels it. (Whisper
   applies its resolved language via the existing `applyLanguage` before the session starts;
   Dolphin ignores language as today.) Tested through the `SherpaRecognizerLike` seam.
5. **`DolphinAttnEngine`** — `DolphinAttnSession` (parallel shape): `accept()` feeds
   windows to the resident `vad`; closed segments are converted back to `ShortArray` and the
   decode worker runs the existing `beamSearch` + `decodeTokens` per segment (per-segment KV
   cache, ur/PK prefix — unchanged) from an **unbounded `Channel`**; `flush()` = drain the VAD
   tail into the channel, then close + `worker.join()` (same no-drop guarantee) + return
   joined text; zero segments → whole-clip `beamSearch(ref, accumulated)` fallback (ticket-30
   semantics). Tested through the new `DolphinRecognizerLike` seam (interface implemented by
   the engine + a test-only session ctor).
6. **`VadSegmentation.kt`** — add a live drainer shared by both session types: push one
   window at a time into `VadLike` and return any segments closed by that push
   (`vad.isSpeechDetected()` → drain `front()`/`pop()`), plus a `flushAndDrain()` for the
   tail; keep the existing batch `segmentAudioWithVad` for whole-clip fallback. Adds a
   ShortArray↔FloatArray converter used by the sessions.
7. **`WhisperManager.kt`** — `suspend fun createSession(): TranscriptionSession?`:
   ensure the model is loaded (expose/adjust `ensureLoaded`), then `engine.createSession(...)`
   with the same language resolution as `doTranscribe`. Document that the session borrows the
   resident model for its whole lifetime.
8. **`AudioRecorder.kt`** — optional `frameListener: (ShortArray) -> Unit` (settable before
   `start`), invoked from the capture loop with each `read()` chunk; WAV writing unchanged.
   Hardware seam — verified on-device, not unit-tested.
9. **`DictationService.kt`** —
   **START:** if `whisper.createSession()` succeeds, start rec, wire
   `frameListener = { session.accept(it) }`, and collect `session.partials` in
   `serviceScope`, updating `app.setTranscript(accumulatedJoin)` as phrases land
   (mirror the existing "transcribing while recording" flagging; keep listening header).
   Session null → record as today (no live path).
   **STOP:** `recorder.stop()` → `setRecording(false)` → `session.flush()` → final text →
   copy/notify/history exactly as today (blank → "No speech detected"). Null session → run the
   existing `whisper.transcribe(wav)` batch path unchanged.
10. **`DictationScreen.kt`** — while `recording`, render a non-null `transcript` as the
    growing live text (small branch; today it only shows after Stop). No layout rewrite.
11. **New JVM tests** — `LiveVadDrainerTest` (fake `VadLike`: per-window emission, tail on
    flush, empty-input) and a `WhisperManager`/engine test that `createSession` is null for a
    non-streaming fake engine and non-null for a streaming fake; keep the existing
    `VadSegmentationTest`, `WhisperManagerTest` suites passing.

## Correctness constraints (carry from tickets 29 → 30 — MUST NOT regress)

- DolphinAttn: `ur`/`PK` prefix **per segment**; KV cache **per segment**, never shared;
  fp16 KV, full-logits selection, `BEAM_SIZE=1`, length-normalized selection — all unchanged.
- Sherpa engines: identical `applyLanguage`/`resolveLanguage` semantics as batch `transcribe`;
  a fresh `OfflineStream` per segment (never reused); stream released in `finally`.
- Segment order is preserved end-to-end (capture order = channel order = `partials` order);
  `flush()` **joins the decode worker** after closing the channel, so it never races ahead of
  or drops queued segments — live and tail are all decoded before the final transcript is
  composed.

## Open items to confirm during implementation

1. **`maxSpeechDuration`:** keep 5.0 s (live phrase cap) vs raise (longer phrases, higher
   per-utterance decode ceiling). Verify per-phrase publish latency on the A50; tune there.
   *(Still open — tune on-device.)*
2. **Segment queue bound — RESOLVED (2026-09-06):** unbounded. The original plan assumed a
   bounded queue with back-pressure; on-device observation showed the live decode worker
   (beam search) can fall behind capture and a small bound silently **dropped segments 6+**
   via `trySend`+drop, corrupting the transcript. Both sessions now use
   `Channel(Channel.UNLIMITED)` and `flush()` joins the worker, so no segment is ever dropped
   at enqueue or at stop. Covered by `StreamingSessionNoDropTest`.
3. **`transcribing` flag semantics** during live decode — decide between "stays false while
   recording (partials are live)" and a tri-state; keep the change minimal.
4. **Live text in the recording notification** — explicitly out of scope here; add a follow-up
   ticket if wanted.

## Verification

- **JVM unit tests** (no device): the live drainer + session-null behavior via the
  `VadLike`/`WhisperEngine` seams, plus the new `StreamingSessionNoDropTest` (no-drop on a
  slow worker, tail preserved); **53 tests, all green**.
- **Build:** `./gradlew :app:assembleDebug` and `:app:testDebugUnitTest`; `lintDebug` no new
  errors (the pre-existing `AudioRecorder.kt:45` MissingPermission stays).
- **On-device A50:** multi-phrase Urdu dictation — confirm partials appear per phrase while
  recording, Stop final equals ticket-30 batch output for the same clip, and log
  per-phrase publish latency.

## Comments

- **2026-09-05**: Created from the "show transcription as each VAD segment is processed"
  plan. Research re-confirmed: sherpa offers **no** streaming for Whisper or Dolphin (Dolphin
  is `OfflineDolphinModel : OfflineCtcModel`, Whisper is offline-only; `OnlineModelConfig` has
  no entry for either). Its official answer is *simulated streaming* = live VAD → offline
  decode per closed utterance, shipped in `android/SherpaOnnxSimulateStreamingAsr` (Whisper
  tiny + Dolphin-base APKs). That is exactly the pattern encoded here, extended to the
  custom DolphinAttn engine. GGML (whisper.cpp AAR) is deliberately excluded — its API is
  batch-only.
- **2026-09-06**: **No-drop fix + deterministic tests** (final chunk of this ticket).
  - **Root cause found:** the sessions used a 5-segment bounded `Channel` with
    `trySend`+drop. When the live decode worker (DolphinAttn beam search) fell behind the
    capture thread, segments 6+ were silently dropped mid-recording; `flush()` also
    `scope.cancel()`d the worker without joining it, dropping anything still queued at Stop.
  - **Fix:** both `DolphinAttnSession` and `SherpaOfflineSession` now use
    `Channel(Channel.UNLIMITED)`; `flush()` drains the VAD tail into the channel, closes it,
    then **`worker.join()`s** before cancelling, so every enqueued segment (live + tail) is
    decoded. Workers collect into `decodedTexts` (authoritative transcript, read after join);
    `partials` stays best-effort UI.
  - **Duplicate-transcript bug found after the no-drop build landed:** the old flush composed
    the final text by prepending the UI's `partialsSnapshot` to the joined `decodedTexts`.
    With the worker joined, `decodedTexts` already contains every live *and* tail segment, so
    the prepend doubled the live portion (on-device log showed the first phrases twice).
    Fixed: the final transcript is now just `decodedTexts.joinToString(" ")`; the
    `partialsSnapshot` parameter is accepted for signature compatibility but unused.
    `StreamingSessionNoDropTest` now passes a realistic non-empty snapshot and asserts
    no duplication (4/4 green; full suite 53 passing).
  - **First-utterance skip (VAD state carry-over) — fixed:** DolphinAttn reuses ONE
    resident silero `Vad` (created at model load) for every streaming session *and* batch
    transcribe, and it was never reset. Silero is stateful (speech-start flag + buffered
    tail); a previous dictation's `flush()` leaves it mid-speech, which could swallow or
    merge the next recording's first utterance. Fix: `ref.vad.reset()` at session creation
    in `DolphinAttnEngine.createSession` and at the top of the batch `segmentWithVad`
    (clean slate per usage). Sherpa sessions were already safe — they build a fresh `Vad`
    per session. Verified on-device: 4 back-to-back recordings, first utterance captured
    every time (the earlier profile showed the culprit pattern: `flush()` at ~2.7 s before
    speech, controller state shared).
  - **Testability seams:** `SherpaOfflineSession` gained a `SherpaRecognizerLike` ctor;
    `DolphinAttnSession` gained a matching `DolphinRecognizerLike` ctor (new `internal
    interface` implemented by `DolphinAttnEngine`), so both sessions are JVM-testable without
    native libs or `android.content.Context`.
  - **Tests:** `StreamingSessionNoDropTest` (4 cases) pushes 7/8 segments (> the old bound)
    through live + tail paths and asserts none are dropped; stable across repeated runs.
    Full `:app:testDebugUnitTest` passes (53 tests). `lintDebug` unchanged.
  - **Remaining:** on-device A50 verification (final AC) + tune open items 1 (`maxSpeechDuration`)
    and 3 (`transcribing` flag) to taste.