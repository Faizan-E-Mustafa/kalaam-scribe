# Android: keep sherpa-onnx and onnxruntime-android on the same ORT version

The Android app ships exactly one `libonnxruntime.so` per ABI, but two independent
native consumers link against it: sherpa-onnx's `libsherpa-onnx-jni.so` (Dolphin
CTC, sherpa Whisper) and onnxruntime-android's `libonnxruntime4j_jni.so` (the
`ai.onnxruntime` glue behind `DolphinAttnEngine`). They must therefore be built
against the **same** ORT version, and we must never rely on `pickFirsts` plus "one
version wins, it's a superset" reasoning.

Reason: Android's linker (bionic, `find_verdef_version_index`) resolves shared
library symbols by **exact ELF symbol-version name**. If a consumer needs
`OrtGetApiBase@VERS_1.27.1` but the only provider exports `VERS_1.24.3`,
`dlopen` fails at runtime (only index-1 BASE symbols fall back to the global
scope; the ORT C-API entry point is versioned in both builds). This is a load-time
crash, not a link error, so it only surfaces on-device.

Real-world trap (Sep 2026): sherpa-onnx 1.13.5 bundles ORT **1.27.1**, but no
`onnxruntime-android:1.27.1` exists on Maven Central or in the ORT v1.27.1 GitHub
release assets. So "align both to 1.27.1" is impossible and "bump ORT to 1.27.0"
blindly still mismatches (`VERS_1.27.0` vs `VERS_1.27.1`). The fix that works is
to pick a sherpa release whose bundled ORT exactly equals a published
`onnxruntime-android` version and pin both in `libs.versions.toml`: we chose
**sherpa-onnx 1.13.4 + onnxruntime-android 1.27.0** (sherpa 1.13.4 is itself built
and linked against ORT 1.27.0, verified via `readelf --version-info` on both
AARs). sherpa-onnx is fetched from its official GitHub-release AAR through a
content-scoped Ivy repo in `settings.gradle.kts` (`@aar`) instead of JitPack.

Guardrails when changing either version: bump `sherpaOnnx` and `onnxruntime`
**together**, then verify the merged APK — exactly one `libonnxruntime.so` per
ABI, and the provider plus `libsherpa-onnx-jni.so` plus
`libonnxruntime4j_jni.so` all carry the same `VERS_*` (check with
`readelf --version-info`). Before choosing a pair, confirm the ORT AAR exists at
that version (`repo1.maven.org/maven2/com/microsoft/onnxruntime/onnxruntime-android/`)
and that the sherpa release bundles exactly that ORT
(sherpa-onnx `CHANGELOG.md`: "Use onnxruntime X for Android"). This decision is
enforced in code comments at the dependency sites and in ticket 29.