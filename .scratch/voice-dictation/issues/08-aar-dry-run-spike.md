# 08: AAR dry-run spike on device

**What to build:** Add the whisper.cpp prebuilt AAR
(`dev.ffmpegkit-maintained:whisper-android`) to the scaffolded app and prove it
loads a real GGML Model and transcribes a known audio sample on the A50. This
de-risks the engine choice before the model manager and pipeline are built.

**Blocked by:** 07 (Android toolchain + project scaffold).

**Status:** done

## Resolution

- AAR resolved (`dev.ffmpegkit-maintained:whisper-android:1.0.0`). Its Kotlin API
  (package `dev.ffmpegkit.whisper`) differs from the README's `com.whispercpp.whisper`:
  `Whisper.loadModel(context, path)` (suspend) -> `WhisperModel`,
  `Whisper.transcribe(model, audioPath, config)` (suspend) -> `WhisperResult` (`.text`),
  `WhisperConfig(language=…, translate=…, threads=…, maxSegmentLength=…, printTimestamps=…)`.
- Built APK ships `lib/arm64-v8a/libwhisper.so` + `libc++_shared.so` (A50 = arm64-v8a).
- Spike wrote a `SpikeRunner` that loads `ggml-base-q8_0.bin` (82MB quantized) and
  transcribes `jfk.wav`, showing the text on screen + logs.
- **Key finding (storage):** reading the model from `/sdcard/Download/` failed
  ("failed to initialize model") on Android 11 lookup/scoped storage - no storage
  permission was declared. Fix: keep model + audio in the app's **internal files
  dir** (`context.filesDir`), reached via `adb shell run-as <pkg>`. Internal
  (app-private) storage needs no runtime permission. The final WhisperManager
  should load Models from app-private storage, not public Download.
- Received correction: use app-internal storage rather than public dirs.

## Acceptance criteria

- [x] The app depends on the AAR and builds for arm64-v8a.
- [x] On the A50, the app loads a quantized GGML model file (ggml-base-q8_0.bin)
      and transcribes jfk.wav, returning correct text:
      "And so my fellow Americans, ask not what your country can do for you,
      ask what you can do for your country." Confirms quantized models work
      through the AAR.
- [x] A small sample transcription is produced and visible in the UI + log,
      demonstrating the end-to-end native-inference path.
