# HF safetensors → whisper.cpp GGML converter

`convert-h5-to-ggml.py` convconverts a Hugging Face fine-tuned Whisper model
(safetensors) into the legacy whisper.cpp `.bin` (ggml) format, which
`whisper-cli` and `whisper-quantize` consume.

`mel_filters.npz` is the OpenAI Whisper mel-filter bank required by the
converter (from `openai/whisper` `whisper/assets/`).

## Usage

Known-good invocation for a multilingual fine-tune like
`cheetos18/whisper-small-roman-urdu`:

```
uv run -- python convert-h5-to-ggml.py \
    /path/to/hf-model-dir \
    /path/to/openai-whisper-repo   # only mel_filters.npz is needed
    /path/to/output-dir
```

Requirements (in the `tools/validate_stt` venv): `torch`, `transformers`,
`numpy`. Open a shell with that venv active (`~/.local/bin/uv run -- python`).

Notes learned in the field:

- The HF repo's `model.safetensors` is stored with Git LFS; a plain `git clone`
  leaves a pointer stub. Fetch the real file with:
  `curl -sL -o model.safetensors https://huggingface.co/<org>/<repo>/resolve/main/model.safetensors`
- Output is `ggml-model.bin` in **f16** by default (≈487 MB for small).
- To quantize, apply afterward:
  `whisper-quantize ggml-model.bin ggml-model-q4_0.bin q4_0`
  (silent `--no-print` variants exist; q4_0/q5_0/q8_0 all verified for the
  Roman-Urdu model).
- **Language flag matters:** on a fine-tuned multilingual model, run
  `whisper-cli -l auto` (NOT `-l ur`) to get clean Roman/Latin output; forcing
  the native language token leaks native-script tokens.