# 20: Versioning convention for releases

**What to build:** A working convention for versioning and tagging app releases,
so each milestone build is identifiable on the device and in git history.

**Blocked by:** nothing.

**Status:** ready-for-agent

## Convention

- `versionName` = semantic version (e.g. `1.1`), bumped for each release
  milestone (one ticket or a coherent group of tickets).
- `versionCode` = monotonic integer, bumped alongside (`2`, `3`, ...).
- Each release is tagged in git as `v<versionName>` (e.g. `v1.1`) on `development`.
- The APK stays local + gitignored (`android/app/build`); git holds source +
  tags only. No APK committed to the repo.
- Record the release in the ticket log: version, tag, what it shipped, and the
  `adb install -r` path.

## This release: v1.1 (tickets 16–20)

- `versionCode = 2`, `versionName = "1.1"` — Roman-Urdu runtime download:
  hosted GGML files, q4_0 as default catalog model, docs updated, versioning
  convention added.
- Tag: `v1.1`.
- APK: `android/app/build/outputs/apk/debug/app-debug.apk` (debug-signed,
  ~19 MB; models download separately from `femustafa/voicedictation-models`).
- Install: `adb install -r android/app/build/outputs/apk/debug/app-debug.apk`.

## Verify

- [ ] `versionCode`/`versionName` bumped in `android/app/build.gradle.kts`
- [ ] `./gradlew :app:assembleDebug` builds; APK reports `1.1`
- [ ] `adb install -r` on device; app version shows `1.1`
- [ ] commit + `git tag v1.1` on `development`