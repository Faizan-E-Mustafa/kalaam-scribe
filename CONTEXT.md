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
