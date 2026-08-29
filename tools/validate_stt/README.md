# validate-stt — dev harness

Validates faster-whisper (`base.en`) transcription on the Linux dev machine before
anything touches the phone. This is a **development harness only**; the runtime
target is the Samsung A50 (see `.scratch/voice-dictation/spec.md`).

## Why Python 3.11 (the blocker workaround)

The system Python is 3.14, for which `ctranslate2` (the engine behind
`faster-whisper`) has no wheel yet (`pip install faster-whisper` fails to install a
working native backend). We therefore use `uv` to manage a Python 3.11 environment
and install dependencies. `uv` is a single binary installed to `~/.local/bin`, so no
sudo is needed.

## Setup

```sh
# one-time: ensure uv is on PATH
export PATH="$HOME/.local/bin:$PATH"

cd tools/validate_stt
uv sync        # creates .venv with Python 3.11 + faster-whisper, downloads base.en on first run
```

## Run

```sh
uv run validate_stt.py                                # downloads + runs the known sample
uv run validate_stt.py /path/to/your-audio.mp3        # run against your own audio
uv run validate_stt.py clip.wav --model small         # try a different model size
```

Without an argument the harness downloads a known English sample
(`hf-internal-testing/dummy-audio-samples`, "Mary had a little lamb") and
transcribes it with `base.en`, `int8`, CPU, `vad_filter=True` — the same settings the
phone will use.

The model is cached offline in `~/.cache/huggingface/hub/` after the first run.
