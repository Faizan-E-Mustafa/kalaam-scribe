# 07: Android toolchain + project scaffold

**What to build:** A self-contained `android/` directory in this repo holding a
Kotlin Android app project (Jetpack Compose) that opens/builds and produces a
skeleton APK installable on the phone. Includes setting up the Android toolchain on
the dev machine (WSL2, amd64) so later tickets can build and verify.

**Blocked by:** None (can start immediately).

**Status:** done

## Acceptance criteria

- [ ] Android toolchain (JDK, Android SDK cmdline-tools, platform + build-tools,
      NDK) installed and reachable on the WSL2 dev machine; `gradlew` runs.
- [ ] A clean `android/` Gradle project with a Compose app that produces an APK
      (multi-ABI) without errors.
- [ ] The APK installs on the phone (Android 8.0+/API 26+) launched from the home
      screen showing a placeholder screen.
- [ ] A device/adb path from WSL2 to the phone is documented (USB/adb over WSL2 or
      network adb), since USB passthrough from WSL2 needs explicit setup.

## Progress

- [x] Scaffolded the full `android/` project: root + `app` Gradle files, version
      catalog (`gradle/libs.versions.toml`), Gradle wrapper (8.11.1) incl. jar,
      Compose app (MainActivity + placeholder `DictationScreen`), theme, adaptive
      launcher icons, ProGuard rules, `android/.gitignore`.
- [x] Multi-ABI and "any Android phone" targeting: `minSdk 26` (Android 8.0+), ABIs
      `arm64-v8a`, `armeabi-v7a`, `x86_64`.
- [x] Installed the Android toolchain on the WSL2 machine (installed by agent, no
      root needed): local Temurin JDK 17.0.20.1 at `~/android-tooling/`, Android
      SDK at `~/Android/Sdk` (cmdline-tools, platform-tools/adb, `platforms;android-35`,
      `build-tools;34.0.0`). Env vars persisted to `~/.bashrc`; `local.properties`
      written in `android/`.
- [x] First build: `./gradlew assembleDebug` succeeded, producing
      `app/build/outputs/apk/debug/app-debug.apk` (~16 MB).
- [x] Pending (human/device step): install the APK on the phone and confirm the
      placeholder screen launches. The phone is not currently connected to the
      Windows host (`usbipd list` shows only laptop peripherals). To finish,
      plug the phone into the PC, enable Developer options + USB debugging, and
      attach it to WSL2 via usbipd (e.g. `usbipd bind` then `usbipd attach` in an
      admin PowerShell), or use wireless debugging. Then `adb install
      app/build/outputs/apk/debug/app-debug.apk`.

## Device connection (in progress — with user)

- adb lives only in WSL2 at `~/Android/Sdk/platform-tools/adb` (not on Windows).
- USB path: `usbipd bind --busid 1-4` then `usbipd attach --wsl --busid 1-4` (admin
  PowerShell) exposes the phone to WSL2, but the device node is root-owned
  (`no permissions`); fix via udev rule or `sudo chmod 666` on the node.
- Wireless path (chosen): phone Developer options → Wireless debugging → pair +
  connect. Pairing succeeded. Remaining manual steps with the user:
  `adb connect IP:PORT`, then `adb install app/build/outputs/apk/debug/app-debug.apk`,
  open the app, confirm placeholder renders. adb runs inside WSL2, not Windows.

## Resolution

- APK installed on the phone via wireless adb (`adb install …/app-debug.apk` →
  "Success"). User drove the device connection; couldn't run adb install myself as
  they asked to do device-touching commands manually first.
- Ticket 07 accepted: toolchain installed, scaffold builds, APK installs.

## Notes

- The developer installs the toolchain (no tooling on this machine yet: no JDK, no
  SDK, no adb; OS is WSL2 amd64). Until then this ticket's build/verify steps are
  blocked on that human step.
- Version stack: AGP 8.7.3 / Kotlin 2.0.21 / Compose BOM 2024.12.01 / Gradle 8.11.1
  — a broad-compatibility, well-documented set rather than bleeding edge, so it
  works with a range of Android Studio releases the dev installs.
- The whisper.cpp AAR dependency is intentionally NOT added yet; it lands in
  ticket 08 per the breaking edges.
