# Kalaam Scribe — Android build + on-device runbook

This documents how to reproduce, on a fresh machine, everything we did so far for
the native Android app: installing the build toolchain, building the scaffold APK,
connecting an Android device, and running the whisper.cpp AAR on-device spike.

It is written with **portable placeholders** so it survives in a public repo. The
original machine was **Windows 11 host running WSL2 (Ubuntu 26.04, amd64)**, building
for a **an Android phone (arm64-v8a, Android 11/API 30)**.

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
| Target phone | Android device, arm64-v8a, Android 11 (API 30) |

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

> **Important for anyone cloning the repo:** `android/local.properties` is **gitignored** and **will not be committed** to the public repository. It is a local-only file that tells Gradle where your Android SDK is installed. You must create it yourself with your own SDK path (e.g., `sdk.dir=/path/to/your/Android/Sdk`) before building. Without it, Gradle will fail with "SDK location not found". See the section above on installing the Android SDK.

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

## 4.5 Releasing a signed APK to GitHub

Users sideload the app from a GitHub Release (not the Play Store), so the APK is
signed with a **stable self-signed key** — enough for Android to install it, with
no Google account or paid certificate. The signature must stay **the same key
across releases**, otherwise Android rejects an update over an existing install
("integrity check failed") and users must uninstall/reinstall.

### One-time setup: generate the signing key

```bash
cd android
keytool -genkeypair -v -keystore app-signing.p12 -alias kalaam-scribe \
  -keyalg RSA -keysize 2048 -validity 10000 -storetype pkcs12 \
  -dname "CN=Kalaam Scribe, O=Kalaam Scribe, C=US"
```

- Set a password (min 6 chars) and **store it in a password manager** — it is the
  only credential guarding the release. Anyone with the password + `.p12` can sign
  releases.
- `app-signing.p12` is gitignored — never commit it. It does **not** need to be
  in the repo; it lives on your machine only.
- For convenience, save the password in a gitignored `android/.env` as
  `APK_SIGNING_PASSWORD=...` so you don't retype it.

### Signing config

`android/app/build.gradle.kts` already wires the `release` build type to this
keystore (reads the password from the `APK_SIGNING_PASSWORD` env var). The
`debug` build stays unsigned. No changes needed here unless the key is
regenerated.

### Build the signed release APK

```bash
cd android
# pick up the password from .env (or export APK_SIGNING_PASSWORD=... manually)
export APK_SIGNING_PASSWORD="$(cut -d= -f2- .env)"
./gradlew :app:assembleRelease
```

Output APK:

```
android/app/build/outputs/apk/release/app-release.apk   (~123 MB)
```

> Note: a `.env` password containing shell-special characters (e.g. `(`) breaks
> `source .env`; the `$(cut -d= -f2- .env)` form above reads it verbatim and avoids
> that.

### Verify the signature is embedded

This AGP version embeds the signature in the binary `AndroidManifest.xml` rather
than as separate `META-INF/*.rsa` files, so checking the zip for signature files
returns nothing. Instead confirm your certificate bytes are inside the APK:

```bash
cd android
export APK_SIGNING_PASSWORD="$(cut -d= -f2- .env)"
keytool -exportcert -keystore app-signing.p12 -alias kalaam-scribe \
  -storepass "$APK_SIGNING_PASSWORD" -rfc -file /tmp/kalaam.pem
python3 - <<'EOF'
import glob, base64
pem = open('/tmp/kalaam.pem').read()
b64 = ''.join(l for l in pem.splitlines() if l and not l.startswith('-----'))
der = base64.b64decode(b64)
apk = glob.glob('app/build/outputs/apk/release/*.apk')[0]
print('cert found in release APK:', der in open(apk, 'rb').read())
EOF
```

Expect `cert found in release APK: True`.

### Bump the version (each release)

In `android/app/build.gradle.kts`, bump **both** `versionCode` and `versionName`
(e.g. `1.0.2` → `1.0.3`), then rebuild. Keep the same keystore so updates install
in place.

> `versionCode` must increase on every release — Android uses it to order upgrades,
> so a rebuild at the same code will refuse to install over an existing app. Only
> `versionName` is user-visible.

### Write the release notes

Add a section for the new version at the top of `CHANGELOG.md`, newest first. That
file is the source of truth; the GitHub release body is copied from it, so never
hand-edit a release body on github.com. Don't repeat the requirements — they live
once in the [README](../README.md#requirements) and haven't changed since v1.0.0.

### Commit, tag, and publish

Tag **after** merging to `main`, never on a side branch. v1.0.1 shipped without
the transcript scrollbar because the tag was cut on a branch that later merged into
`main`; tagging the merge commit is what guarantees the tag matches the APK.

```bash
cd android/..   # repo root

# 1. Merge the work into main first, and confirm main is what you built from.
git checkout main && git merge development

# 2. Commit the version bump and the CHANGELOG section.
git add android/app/build.gradle.kts CHANGELOG.md
git commit -m "Bump to v1.0.3 (patch: <short summary>)"

# 3. Tag the merge/bump commit, then push both.
git tag -a v1.0.3 -m "v1.0.3: <one-line summary>"
git push origin main && git push origin v1.0.3
```

Then publish, extracting the matching section as the body:

```bash
# Extract the v1.0.3 section of CHANGELOG.md into a temp file, then:
gh release create v1.0.3 android/app/build/outputs/apk/release/app-release.apk \
  --title "v1.0.3" --notes-file /tmp/v1.0.3-notes.md --latest
```

To extract the section (from the `## v1.0.3` heading up to the next `## `):

```bash
awk '/^## v1\.0\.3$/{f=1} f&&/^## /&&!/^## v1\.0\.3$/{exit} f' CHANGELOG.md \
  | sed -e '1d' -e '/./,$!d' -e 's/^### /## /' > /tmp/v1.0.3-notes.md
```

> Never commit the APK or the keystore — `android/.gitignore` excludes both, and
> the APK is a release *asset*, not a tracked file.

---

## 5. Connect an Android device

Two options. `adb` runs **inside WSL2**.

### 5a. Wireless debugging (recommended; avoids USB permission fiddling)

1. On the phone: **Settings → Developer options → Wireless debugging → ON**.
2. Tap **Pair device with pairing code** → note `IP:PAIR_PORT` + 6-digit code.
3. In WSL2, pair (one-time):
    ```bash
    adb pair IP:PAIR_PORT CODE
    # Example: adb pair <IP>:<PORT> 123456
    # Replace <IP> and <PORT> with your actual device IP and port
    # (found in Settings → Developer options → Wireless debugging → Pair device)
    # Example: adb pair 192.168.1.20:37000 123456
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
usbipd list                                  # note the BUSID of the Android ADB device, e.g. 1-4
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
WAV on the phone.

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
    adb shell "run-as dev.femustafa.kalaamscribe sh -c 'cp /data/local/tmp/ggml-base-q8_0.bin /data/local/tmp/jfk.wav files/ && ls -la files/'"
```

> The app package id is `dev.femustafa.kalaamscribe`. A reinstall (`install -r`)
> preserves app internal data, so the copies survive.

### 7d. Run and observe

```bash
    adb shell am start -n dev.femustafa.kalaamscribe/.MainActivity
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
    adb uninstall dev.femustafa.kalaamscribe
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
| `SigningConfig "release" is missing required property "keyPassword"` | Add `keyPassword = System.getenv("APK_SIGNING_PASSWORD")` to the signing config (same value as `storePassword`). |
| `source .env` fails with `syntax error` | The password has shell-special chars (e.g. `(`). Read it with `export APK_SIGNING_PASSWORD="$(cut -d= -f2- .env)"` instead of `source`. |

---

## Related docs

- `android/gradle/libs.versions.toml` — dependency versions.
- `android/local.properties` — SDK path (gitignored).
- `.scratch/voice-dictation/issues/07-android-toolchain-scaffold.md` — scaffold ticket.
- `.scratch/voice-dictation/issues/08-aar-dry-run-spike.md` — spike ticket + storage lesson.
- `docs/adr/0003-prebuilt-whisper-cpp-aar-on-android.md`, `docs/adr/0004-resident-model-lifecycle.md` — decisions behind this.
