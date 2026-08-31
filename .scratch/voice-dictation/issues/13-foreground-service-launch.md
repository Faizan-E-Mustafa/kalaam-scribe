# 13: Foreground service + launch surface

**What to build:** Reliable up-front launch: a foreground service that runs only
while recording/transcribing (then stops — no permanent service), plus an app icon
and a home-screen widget/shortcut to start dictation from the home screen in the
background on the A50.

**Blocked by:** 12 (Dictation pipeline).

**Status:** ready-for-agent

## Acceptance criteria

- [ ] A foreground service starts when a dictation begins and stops when the
      transcription finishes; its persistent notification is the recording/stop
      surface.
- [ ] No permanent/always-on service; battery cost is limited to active
      recording/transcription.
- [ ] Dictation can start from a home-screen widget/shortcut and keeps running
      while the user is in another app (the A50 background-reliability fix that
      Termux could not provide).
- [ ] The resident Model stays hot in memory across these launches even though no
      service runs when idle (ADR 0004 distinction).
