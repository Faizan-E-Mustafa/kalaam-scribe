# 01: Unified settings sheet with collapsible advanced settings

**What to build:** Tapping a gear icon on the dictation screen opens a modal
bottom sheet that replaces both the old Settings and Model screens. The sheet
shows a Language dropdown and a collapsible Model selector up top (only the
current model is visible until the user expands it), then a collapsible
"Advanced settings" section holding the whisper-threads, transcription-mode,
VAD, and dolphin-beam controls. Model rows use a download icon instead of a
"Download" button, in the sheet and on first-run onboarding alike. Nothing else
in the flow changes: Save persists, Reset restores defaults, and the sheet
dismisses back to dictation.

**Blocked by:** None (can start immediately).

**Status:** ready-for-agent

- [ ] Dictation screen shows only a gear icon (no Settings chip, no model-name chip); tapping it opens the sheet.
- [ ] Sheet shows Language and a collapsible Model selector that lists models filtered by language when expanded.
- [ ] All prior settings (whisper threads, transcription mode, VAD sliders, dolphin beam size) live under a collapsible "Advanced settings" section, collapsed by default.
- [ ] Download control in model rows is an icon (not a text button) on both the sheet's model list and the onboarding screen.
- [ ] Reset restores defaults; Save persists; the sheet closes on dismiss and on Save.
- [ ] No standalone Settings/Models screens remain (dead code removed).