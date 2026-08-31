# Voice Dictation — Android build + on-device runbook

This documents how to reproduce, on a fresh machine, everything we did so far for
the native Android app: installing the build toolchain, building the scaffold APK,
connecting an Android device, and running the whisper.cpp AAR on-device spike.

It is written with **portable placeholders** so it survives in a public repo. The
original machine was **Windows 11 host running WSL2 (Ubuntu 26.04, amd64)**, building
for a **Samsung Galaxy A50 (SM-A505FN, Android 11 / API 30, arm64-v8a)**.

> Conventions used below:
> - `~/...` = your Linux home dir inside WSL2.
> - `<USER>` = your Windows username (e.g. `femustafa`).
> - `HOSTNAME` = your WSL2 machine name.

---

## 1. Overview of the final setup (verified)

| Item | Value |
|---|---|
| Build OS | WSL2 (Ubuntu), amd64 |
| JDK | Temurin **17.0.20.1** at `~/android-tooling/jdk-17.0.20.1+1` |
| Android SDK | `~/Android/Sdk` (cmdline-tools, platform-tools, `platforms;android-35`, `build-tools;34.0.0`) |
| Gradle | 8.11.1 (wrapper, no system Gradle needed) |
| AGP | 8.7.3 |
| Kotlin | 2.0.21 |
| Compose BOM | 2024.12.01 |
| whisper.cpp AAR | `dev.ffmpegkit-maintained:whisper-android:1.0.0` |
| Target phone | Samsung A50, arm64-v8a, Android 11 (API 30) |

**Key facts discovered (avoid repeating them):**

- `adb` lives **only inside WSL2** at `~/Android/Sdk/platform-tools/adb`. It is not on
  the Windows side unless you install platform-tools there. Run adb **in WSL2**.
- The whisper AAR's Kotlin package is **`dev.ffmpegkit.whisper`** (not
  `com.whispercpp.whisper` as some READMEs claim). API:
  - `Whisper.loadModel(context, path): WhisperModel` (suspend)
  - `Whisper.transcribe(model, audioPath, config): WhisperResult` (suspend), `.text`
  - `WhisperConfig(language=…, translate=…, threads=…, maxSegmentLength=…, printTimestamps=…)`
- On Android 11 (API 30) scoped storage, an app **cannot read arbitrary files in
  `/sdcard/Download/` without a storage permission** (`Whisper.loadModel` throws
  `failed to initialize model`). **Load Models from the app's private internal
  storage** (`context.filesDir`), reached from adb via
  `adb shell run-as <package>`. No runtime permission needed there.

---

## 2. Install the JDK (user-space, no root)

These steps install to your home dir, so they need no `sudo` password.

```bash
mkdir -p ~/android-tooling
cd ~/android-tooling
# Download Temurin JDK 17 (user-space tarball build for Linux x64)
wget -q "https://api.adoptium.net/v3/binary/latest/17/ga/linux/x64/jdk/hotspot/normal/eclipse" -O jdk17.tar.gz
tar -xzf jdk17.tar.gz
rm jdk17.tar.gz
# The extracted dir name varies; the verified one is:
# ~/android-tooling/jdk-17.0.20.1+1
```

Add to `~/.bashrc` (or your shell profile) and reload:

```bash
export JAVA_HOME="$HOME/android-tooling/jdk-17.0.20.1+1"
export ANDROID_HOME="$HOME/Android/Sdk"
export PATH="$JAVA_HOME/bin:$ANDROID_HOME/platform-tools:$PATH"
```

```bash
source ~/.bashrc
java -version   # expect 17.x
```

---

## 3. Install the Android SDK (user-space)

Download **command-line tools** (Linux) and unzip into `~/Android/Sdk`:

```bash
mkdir -p ~/Android/Sdk/cmdline-tools
cd /tmp
wget -q "https://dl.google.com/android/repository/commandlinetools-linux-11076708_latest.zip" -O clt.zip
unzip -q clt.zip -d ~/Android/Sdk/cmdline-tools
mv ~/Android/Sdk/cmdline-tools/cmdline-tools ~/Android/Sdk/cmdline-tools/latest
```

Accept licenses and install the SDK packages:

```bash
yes | sdkmanager --licenses
sdkmanager "platform-tools" "platforms;android-35" "build-tools;34.0.0" "cmdline-tools;latest"
```

Verify:

```bash
adb version          # copied from platform-tools, runs in WSL2
sdkmanager --list_installed
```

> Note: `local.properties` in the project sets `sdk.dir=/home/<USER>/Android/Sdk`.
> Adjust it to your own SDK path (it is gitignored).

---

## 4. Build the app

From the repo root, the Android project is in `android/`:

```bash
cd android
./gradlew :app:assembleDebug
```

On success the APK is at:

```
android/app/build/outputs/apk/debug/app-debug.apk   (~16 MB for the scaffold)
```

The project uses the Gradle **wrapper** (8.11.1), so no system Gradle install is
required. First build downloads dependencies (a few minutes).

---

## 5. Connect an Android device

Two options. `adb` runs **inside WSL2**.

### 5a. Wireless debugging (recommended; avoids USB permission fiddling)

1. On the phone: **Settings → Developer options → Wireless debugging → ON**.
2. Tap **Pair device with pairing code** → note `IP:PAIR_PORT` + 6-digit code.
3. In WSL2, pair (one-time):
   ```bash
   adb pair IP:PAIR_PORT CODE
   # e.g. adb pair 192.168.1.20:37000 123456
   ```
4. From the **Wireless debugging main screen** (not the pairing dialog) read the
   main `IP:PORT`, then connect:
   ```bash
   adb connect IP:PORT
   adb devices -l     # expect: <IP:PORT> device
   ```

### 5b. USB via WSL2/usbipd (needs Windows admin PowerShell)

The phone must appear on the Windows USB bus first (plug in a **data** cable, set
phone USB mode to **File Transfer/MTP**, enable **USB debugging**). Then:

```powershell
# admin PowerShell on Windows
usbipd list                                  # note the BUSID of the Samsung/ADB device, e.g. 1-4
usbipd bind --busid 1-4
usbipd attach --wsl --busid 1-4
```

Back in WSL2, adb often reports `no permissions (missing udev rules)`. Fix with a
udev rule (needs sudo), or grant access to the device node:

```bash
# Option: grant write access to the exposed USB node
sudo chmod 666 /dev/bus/usb/001/001 /dev/bus/usb/001/002
adb kill-server && adb start-server
adb devices -l
```

> In practice **wireless (5a)** is far less fiddly; we recommend it.

---

## 6. Install the APK on the device

```bash
adb install -r android/app/build/outputs/apk/debug/app-debug.apk
```

(Output: `Success`.)

---

## 7. Reproduce the on-device spike (ticket 08)

Goal: prove the whisper AAR loads a **quantized GGML model** and transcribes a known
WAV on the A50.

### 7a. Obtain test assets

```bash
mkdir -p android/spike   # gitignored
curl -fsSL -o android/spike/ggml-base-q8_0.bin \
  "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base-q8_0.bin"
curl -fsSL -o android/spike/jfk.wav \
  "https://github.com/ggerganov/whisper.cpp/raw/master/samples/jfk.wav"
```

### 7b. The spike reads from the app's **internal** files dir

`SpikeRunner.kt` loads from `context.filesDir` (`/data/data/<pkg>/files`) — not
`/sdcard/Download/` (see the scoped-storage note in section 1).

### 7c. Push assets into the app's internal storage

```bash
# stage (world-readable) then copy as the app via run-as
adb push android/spike/ggml-base-q8_0.bin /data/local/tmp/
adb push android/spike/jfk.wav /data/local/tmp/
adb install -r android/app/build/outputs/apk/debug/app-debug.apk
adb shell "run-as dev.femustafa.voicedictation sh -c 'cp /data/local/tmp/ggml-base-q8_0.bin /data/local/tmp/jfk.wav files/ && ls -la files/'"
```

> The app package id is `dev.femustafa.voicedictation`. A reinstall (`install -r`)
> preserves app internal data, so the copies survive.

### 7d. Run and observe

```bash
adb shell am start -n dev.femustafa.voicedictation/.MainActivity
# tap "Run spike (jfk.wav)" on the phone
adb logcat -s WhisperSpike
```

Expected logcat:

```
WhisperSpike: model loaded ok
WhisperSpike: transcription: And so my fellow Americans, ask not what your country can do for you, ask what you can do for your country.
WhisperSpike: spike complete
```

### 7e. Clean up the device (optional)

```bash
adb shell rm -f /sdcard/Download/ggml-base-q8_0.bin /sdcard/Download/jfk.wav
adb shell rm -f /data/local/tmp/ggml-base-q8_0.bin /data/local/tmp/jfk.wav
adb uninstall dev.femustafa.voicedictation
```

---

## 8. Troubleshooting quick reference

| Symptom | Cause / fix |
|---|---|
| `adb: no permissions` | Fix USB node perms (section 5b) or use wireless. |
| `failed to initialize model` | App can't read the model path (scoped storage). Use internal `filesDir` + `run-as`, not `/sdcard/Download/`. |
| `SpikeRunner: spike complete` with no `transcription:` | The early-return "MISSING model/audio" path ran; verify the files are in the dir the code reads and that the rebuilt APK is installed. |
| Wrong package for whisper API | It is `dev.ffmpegkit.whisper`, not `com.whispercpp.whisper`. |
| `usbipd: Access denied` | Run `usbipd bind`/`attach` from an **admin** PowerShell. |

---

## Related docs

- `android/gradle/libs.versions.toml` — dependency versions.
- `android/local.properties` — SDK path (gitignored).
- `.scratch/voice-dictation/issues/07-android-toolchain-scaffold.md` — scaffold ticket.
- `.scratch/voice-dictation/issues/08-aar-dry-run-spike.md` — spike ticket + storage lesson.
- `docs/adr/0003-prebuilt-whisper-cpp-aar-on-android.md`, `docs/adr/0004-resident-model-lifecycle.md` — decisions behind this.
