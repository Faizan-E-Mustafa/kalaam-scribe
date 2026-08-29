# 01: Linux dev harness for faster-whisper

**What to build:** Validate that faster-whisper with the `base.en` model installs and transcribes a known English audio sample on the developer's Linux machine. This proves the model choice and catches environment/wheel problems (notably Python 3.14) before anything touches the phone.

**Blocked by:** None (can start immediately)

**Status:** ready-for-agent

- [x] faster-whisper installs in a Python venv (uv-managed Python 3.11) on the Linux box
- [x] `base.en` model loads and transcribes a known English audio sample
- [x] Transcription output is reasonable text (no crash; known sample transcribed)
- [x] Python-3.14 wheel blocker identified and worked around (documented)

## Acceptance criteria notes
Done when a `base.en` transcription of a short clean English clip succeeds locally and the approach for the phone (int8, CPU, vad) is confirmed usable.

## Outcome (implemented)
- `tools/validate_stt/` created: `pyproject.toml` (uv, Python 3.11), `validate_stt.py`,
  `README.md`, `.gitignore`.
- Verified: faster-whisper 1.2.1 + ctranslate2 4.8.1 install under uv-managed Python 3.11.
- Verified: `base.en` (int8, CPU, vad_filter=True) transcribes the known
  `hf-internal-testing` "Mary had a little lamb" sample; language en (p=1.00).
- Model cached offline in `~/.cache/huggingface/hub/models--Systran--faster-whisper-base.en`.
- Blocker: system Python 3.14 has no ctranslate2 wheel; worked around with uv-managed
  Python 3.11 (documented in the harness README). Note: this ctranslate2-on-pip
  fragility is precisely why the phone (ticket 02) uses whisper.cpp instead of
  faster-whisper (ADR 0002).
