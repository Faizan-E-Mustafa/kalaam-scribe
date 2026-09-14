# Spec: Unified Settings Panel

> Redesign the app's settings + model-selection UI into a single panel so the
> dictation surface is clean. Captures the agreed design before implementation.

Status: ready-for-agent

## Problem Statement

Settings and model selection currently live on two separate full-screen
destinations reachable from the dictation screen's top chips: a "Settings" chip
opens `SettingsScreen` (threads, transcription mode, VAD sliders, dolphin beam
size) and a model-name chip opens `ModelPickerScreen` (language + full model
list). The dictation header is cluttered (status + Settings + model-name chips),
model selection buries the user under the whole catalog, and "download" is a
text button.

## Solution

One settings entry point: a small gear icon on the dictation screen that opens a
**modal bottom sheet**. The sheet is organized so only the common choices are
exposed and the power-user knobs sit behind a collapsible section.

Sheet contents, top to bottom:

1. **Language** dropdown — always visible.
2. **Model** selector — always visible but **collapsible**: shows the current
   model; expands to reveal the (language-filtered) model list, so the catalog is
   never shown all at once.
3. **Advanced settings** — collapsible section (collapsed by default) holding
   everything that used to live in `SettingsScreen`: whisper threads,
   transcription mode, VAD sliders, dolphin beam size.
4. **Reset / Save** controls, same semantics as today.

The download affordance in model rows becomes a **download icon** (no text
button) — both inside the sheet's model list and on the first-run onboarding
screen.

## Design decisions

- One gear icon (material `Settings`) on the dictation screen replaces the Settings
  chip and the model-name chip. The model name is no longer shown in the dictation
  header; the active model is visible in the sheet's Model selector instead.
- The bottom sheet is hosted by `RootApp` over the dictation screen; no standalone
  `SettingsScreen`/`ModelPickerScreen` destinations remain.
- `ModelRow` is shared by the sheet and onboarding, so the download-icon change
  lands in both at once.
- Settings keep the current Save/Reset persistence behaviour
  (`VoiceDictationApp` shared prefs); nothing about model loading or download
  semantics changes.

## Out of scope

- Reordering/naming the models themselves, changing the catalog, or changing the
  onboarding flow's structure (only its download control becomes an icon).
- Changing what each setting controls or its defaults.