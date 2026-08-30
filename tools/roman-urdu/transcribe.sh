#!/usr/bin/env bash
# Transcribe a WAV file with the converted Roman-Urdu whisper.cpp model.
#
# Usage:
#   ./transcribe.sh /path/to/clip.wav
#
# Requirements (all present from the validation work):
#   - whisper-cli (prebuilt ubuntu-x64 binary, at $WHISPER_CLI)
#   - ggml-model-q8_0.bin (converted + quantized Roman-Urdu model)
#   - torch's libgomp.so.1 for the prebuilt binary (at $LD_LIBRARY_PATH)
#
# Set these to point at your actual files if paths differ.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL="${RU_MODEL:-${SCRIPT_DIR}/ggml-model-q4_0.bin}"

WHISPER_CLI="${WHISPER_CLI:-/tmp/opencode/whisper-bin-ubuntu-x64/whisper-cli}"
TORCH_LIB="/home/femustafa/projects/learning_ws/tools/validate_stt/.venv/lib/python3.11/site-packages/torch/lib"

AUDIO="$1"
if [ -z "$AUDIO" ]; then
  echo "usage: $0 <audio.wav>" >&2
  exit 1
fi
if [ ! -f "$AUDIO" ]; then
  echo "error: audio file not found: $AUDIO" >&2
  exit 1
fi

export LD_LIBRARY_PATH="$TORCH_LIB:${LD_LIBRARY_PATH:-}"

echo "model : $MODEL"
echo "audio : $AUDIO"
echo "=== transcription ==="
"$WHISPER_CLI" \
  -m "$MODEL" \
  -f "$AUDIO" \
  -l auto \
  -nt
echo "=== end ==="
