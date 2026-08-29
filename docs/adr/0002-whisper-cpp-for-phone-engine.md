# whisper.cpp as the on-phone engine (faster-whisper stays on the Linux harness)

We run transcription on the phone with **whisper.cpp** (GGML/C++), using the
`base.en` model in a quantized form. The Linux validation harness keeps
**faster-whisper**. Chose this because on Termux/Android, faster-whisper's
`ctranslate2` engine has no aarch64 wheel and must be compiled from source on the
phone (15–30 min, ~2GB disk, OpenBLAS + Android-specific pthread patches via
`termux_CTranslate2`). whisper.cpp builds natively in minutes with `cmake`/`clang`,
is CPU/battery friendly, supports `base.en`, and is stable on Android when built
with `GGML_NO_OPENMP=ON`. Same model family and settings philosophy, far lighter
toolchain burden on the target device.
