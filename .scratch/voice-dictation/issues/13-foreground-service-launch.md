# 13: Foreground service + launch surface

**What to build:** Reliable up-front launch: a foreground service that runs only
while recording/transcribing (then stops — no permanent service), plus an app icon
and a home-screen widget/shortcut to start dictation from the home screen in the
background on the A50.

**Scope decision (user, 2026-08-31):** Foreground service **only** — the
home-screen widget/shortcut is deferred. The service delivers the core
background-reliability fix (dictate while in WhatsApp); the widget is tracked as a
follow-up.

**Blocked by:** 12 (Dictation pipeline).

**Status:** done (foreground service device-verified; widget deferred)

## On-device verification (2026-08-31)

Record → pressed Home / switched to WhatsApp while recording → background
recording kept running (notification stayed) → tapped Stop in the notification →
"Copied" notification with the transcript, clipboard filled. Foreground
service survives backgrounding and the notification is a working stop surface.

## Acceptance criteria

- [x] A foreground service starts when a dictation begins and stops when the
      transcription finishes; its persistent notification is the recording/stop
      surface.
- [x] No permanent/always-on service; battery cost is limited to active
      recording/transcription (START_NOT_STICKY).
- [~] Dictation can start from a home-screen widget/shortcut and keeps running
      while the user is in another app. — Widget/shortcut **deferred** (scope
      decision); "keeps running while in another app" verified via the foreground
      service.
- [x] The resident Model stays hot in memory across these launches even though no
      service runs when idle (ADR 0004 distinction).

## Implementation

- `VoiceDictationApp` (custom `Application`) — holds the **single** shared
  `WhisperManager` (so the service never double-loads the resident model) and a
  state bus (`recording`/`transcribing`/`transcript`/`error` StateFlows) the
  service writes and the Activity observes.
- `DictationService` — **foreground service** that owns `AudioRecorder`. Starts
  (foreground, `FOREGROUND_SERVICE_TYPE_MICROPHONE`) on Record, keeps the mic
  + resident model hot while the user switches apps, and its recording
  notification is the **Stop** surface (action → stop → transcribe → clipboard +
  "Copied" → stopSelf). START_NOT_STICKY; never always-on.
- `DictationViewModel` — now only starts/stops the service and mirrors shared
  state (mic permission gating stays in the UI).
- Manifest: `FOREGROUND_SERVICE` + `FOREGROUND_SERVICE_MICROPHONE`, service
  declared with `foregroundServiceType="microphone"`, `VoiceDictationApp` as the
  Application.
- Build + 10/10 unit tests pass; WhisperManagerTest still verifies resident reuse.

