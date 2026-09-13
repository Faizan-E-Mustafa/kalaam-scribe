# 25: Reduce app size (APK is ~108 MB)

**What to build:** Investigate and reduce the APK size. Debug APK is ~108 MB
(141 MB uncompressed). The two dominant components are native libs and an
unshrunk Compose DEX.

**Blocked by:** nothing.

**Status:** needs-triage

## Findings

Debug APK `android/app/build/outputs/apk/debug/app-debug.apk` ≈ **108 MB**
(141 MB uncompressed).

### 1. Native libs — `lib/` ≈ 87 MB (80%)

- `libonnxruntime.so` ≈ **59 MB** across 3 ABIs (arm64-v8a 20.68, x86_64 23.84,
  armeabi-v7a 14.33). Bundled by `com.github.k2-fsa.sherpa-onnx:sherpa-onnx:1.13.4`
  (its AAR ships 119.66 MB native total, filtered to 3 ABIs by
  `ndk.abiFilters`). **Required** for ONNX inference — not reducible without
  dropping ONNX support.
  <!-- sizes above are from the pre-2026-09-05 build; after the ORT realignment
       (1.24.3→1.27.0, ticket 29) libonnxruntime comes from onnxruntime-android 1.27.0
       and measures arm64-v8a 27.98 / x86_64 33.97 / armeabi-v7a 20.00 MB. -->
- `sherpa-onnx-jni.so` / `sherpa-onnx-c-api.so` / `sherpa-onnx-cxx-api.so`
  ≈ **15 MB** across 3 ABIs. Part of sherpa-onnx, required.
- `libwhisper.so` ≈ 1.5 MB (arm64 only) from `dev.ffmpegkit-maintained:whisper-android`.
  Required for GGML inference.
- `libc++_shared.so` ≈ 1.2 MB. Shared C++ runtime, required.

### 2. DEX — ≈ 53 MB (multidex: classes.dex 42.64 + classes5.dex 10.52)

- `androidx/compose/material` (old Material 2 library) **23,391 classes** —
  pulled in **transitively** by `androidx.compose:compose-bom:2024.12.01`.
  The app only uses `material3` + `material-icons-extended`, **never** the old
  material theme/widgets. Largest single DEX contributor.
- `androidx/compose/foundation` 3,100, `androidx/compose/ui` 1,787,
  `androidx/compose/material3` 1,355, `androidx/compose/runtime` 1,477,
  `androidx/compose/animation` 822 — required by Compose UI.
- `kotlin/jvm`, `java/lang`, `kotlinx/coroutines` etc. — small, required.

### Root causes

1. sherpa-onnx bundles ONNX Runtime (`libonnxruntime.so`) — core dependency.
2. Compose BOM pulls in old `androidx.compose.material:material` (23K classes),
   mostly unused.
3. `isMinifyEnabled = false` in `android/app/build.gradle.kts` — **no R8
   shrinking**, so every class from every library is kept verbatim.

## Proposed fixes (prioritized)

1. **Enable R8 minification** (`isMinifyEnabled = true` + `proguard-rules.pro`
   rules for sherpa-onnx, which uses reflection/native). Biggest single win —
   strips unused classes/methods from all dependencies; could cut DEX 50-70%.
2. **Exclude old `androidx.compose.material:material` from the BOM** — keep
   `material3` + `material-icons-extended`. `material3` does NOT depend on the
   old material base module, so exclusion is safe. Removes ~23K unused classes.
3. **Drop `x86_64` from `ndk.abiFilters`** if no emulator support needed
   (~24 MB native; x86_64 is emulator-only). Target device is
   arm64-v8a/armeabi-v7a.
4. **Switch `material-icons-extended` → `material-icons-core`** if only a
   subset of icons is used (reduces icon footprint).

## Verify

- [ ] Record before/after APK sizes (`assembleDebug` + `assembleRelease`).
- [ ] R8 on: confirm app still launches, transcribes with GGML + ONNX models.
- [ ] Report size delta per change.

## Comments