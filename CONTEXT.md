# Voice Dictation (Local)

Converts spoken input into WhatsApp message text entirely on-device (no cloud, no
WhatsApp API). Runs as a Termux prototype on a Samsung A50; the Linux desktop is
only a development/validation harness.

## Language

**Dictation**:
Speaking into the mic to produce text the user then sends themselves.
_Avoid_: Voice-to-text, transcription, automessaging

**Transcript**:
The text produced from a single dictation utterance.
_Avoid_: Caption, output, answer

**Automessaging**:
Programmatically sending a message without the user confirming it. Explicitly out
of scope; it requires a WhatsApp API, which is rejected.
_Avoid_: Auto-send, voice-reply

**Dictation clip**:
A single recorded utterance that is transcribed into one transcript.
_Avoid_: Recording, segment, note

**Voice-send (deferred)**:
An optional mode that types a transcript and presses Enter automatically. Deferred;
off by default to avoid sending into an unintended field.
_Avoid_: Auto-insert

**Model**:
A downloadable ASR model the app uses, described by the model file plus the
language mode it runs in (auto-detect, a fixed language, or the Roman-Urdu
fixed `en` mode). A model is selected or switched independently of any dictation.
_Avoid_: Engine, variant, STT

**Resident model**:
The single Model kept loaded in memory while the app process lives, so that
repeated dictations reuse it without reloading. Replaced only when the user
switches to another Model. Distinct from a permanent service: the Model stays hot
even though no background service runs when idle.
_Avoid_: Cached model, loaded model

**Switch**:
Replacing the resident model: unload the current one, load the newly selected
one, then use it for subsequent dictations. Noticeably slower than dictation
(seconds) because a new model file is loaded.
_Avoid_: Change model, reload

**Model picker**:
The settings surface where the user chooses which Model is resident and its
language mode.
_Avoid_: Settings, preferences
