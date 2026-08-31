# Prebuilt whisper.cpp AAR on Android (no NDK source build)

For the native Android app we package whisper.cpp via a prebuilt AAR
(`dev.ffmpegkit-maintained:whisper-android` on Maven Central) instead of vendoring
and compiling whisper.cpp ourselves with CMake/NDK/JNI. Chose this so the app
builds with a single Gradle dependency and no native toolchain, at the cost of
less control over the bundled build. This extends ADR 0002 (whisper.cpp remains
the engine) — only the Android packaging changes; the Linux harness keeps
faster-whisper. Models are GGML/GGUF files loaded at runtime from app storage,
including quantized forms (q8_0/q5_1), matching the model the Termux setup already
runs.
