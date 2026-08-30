#!/usr/bin/env bash
# Download a Hugging Face Whisper fine-tune and convert it to whisper.cpp GGML
# (f16). Captures the exact flow used to build the Urdu-script comparison set:
#
#   1. Resolve model.safetensors via the HF LFS URL (plain git clone leaves a
#      pointer stub) and VERIFY it with safetensors before trusting it —
#      interrupted downloads silently truncate and break conversion later.
#   2. Write vocab.json + added_tokens.json (the converter needs them; many
#      repos only ship tokenizer.json).
#   3. Run tools/roman-urdu/converters/convert-h5-to-ggml.py -> ggml-model.bin.
#
# Usage:
#   ./fetch_and_convert.sh <owner/repo> <model-name> [--verify-only]
#
#   ./fetch_and_convert.sh Qasimhassan65/whisper-small-urdu small_Qasim
#
# Output dirs (override with): HF_DIR, OUT_DIR, WORK_DIR.
# Requires the validate_stt venv (torch/transformers/safetensors) and the
# whisper-openai repo for mel_filters.npz (see converters/README.md).

set -euo pipefail

REPO="${1:?usage: $0 <owner/repo> <model-name> [--verify-only]}"
NAME="${2:?usage: $0 <owner/repo> <model-name> [--verify-only]}"
VERIFY_ONLY=0
[[ "${3:-}" == "--verify-only" ]] && VERIFY_ONLY=1

HF_DIR="${HF_DIR:-/var/tmp/urdu-eval/models}"
OUT_DIR="${OUT_DIR:-/var/tmp/urdu-eval/out}"
WORK_DIR="${WORK_DIR:-/var/tmp/urdu-eval/work}"

VENV_PY="${VENV_PY:-/home/femustafa/projects/learning_ws/tools/validate_stt/.venv/bin/python}"
WHISPER_SRC="${WHISPER_SRC:-/tmp/opencode/whisper-openai}"
CONVERTER="/home/femustafa/projects/learning_ws/tools/roman-urdu/converters/convert-h5-to-ggml.py"

BASE="https://huggingface.co/$REPO/resolve/main"
DIR="$HF_DIR/$NAME"
export LD_LIBRARY_PATH="/home/femustafa/projects/learning_ws/tools/validate_stt/.venv/lib/python3.11/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"
mkdir -p "$DIR" "$OUT_DIR" "$WORK_DIR"

needed=(config.json model.safetensors)
for f in "${needed[@]}"; do
  if [ ! -s "$DIR/$f" ]; then
    echo ">> fetching $f"
    curl -sL -o "$DIR/$f" "$BASE/$f"
  fi
done

echo ">> building vocab.json + added_tokens.json from tokenizer.json (if present)"
if [ -s "$DIR/tokenizer.json" ] && [ ! -s "$DIR/vocab.json" ]; then
  "$VENV_PY" - "$DIR" <<'PY'
import json, sys
d = json.load(open(sys.argv[1] + "/tokenizer.json"))
json.dump(d["model"]["vocab"], open(sys.argv[1] + "/vocab.json", "w"))
json.dump({a["content"]: a["id"] for a in d["added_tokens"]},
          open(sys.argv[1] + "/added_tokens.json", "w"))
print("wrote vocab.json + added_tokens.json")
PY
elif [ ! -s "$DIR/vocab.json" ]; then
  echo "!! no tokenizer.json and no vocab.json — supply vocab.json manually"; exit 1
fi

echo ">> verifying $REPO model.safetensors integrity"
if ! "$VENV_PY" -c "import safetensors.torch; safetensors.torch.load_file('$DIR/model.safetensors', device='cpu')" 2>/dev/null; then
  echo "!! safetensors INVALID/truncated — re-downloading with verification"
  [ -f "$DIR/model.safetensors.tmp" ] && rm -f "$DIR/model.safetensors.tmp"
  ok=0
  for attempt in 1 2 3; do
    curl -sL -o "$DIR/model.safetensors.tmp" "$BASE/model.safetensors"
    if "$VENV_PY" -c "import safetensors.torch; safetensors.torch.load_file('$DIR/model.safetensors.tmp', device='cpu')" 2>/dev/null; then
      mv "$DIR/model.safetensors.tmp" "$DIR/model.safetensors"
      ok=1; echo "   valid on attempt $attempt"; break
    fi
    rm -f "$DIR/model.safetensors.tmp"; sleep 3
  done
  [ "$ok" = 1 ] || { echo "!! giving up after 3 attempts"; exit 1; }
fi

if [ "$VERIFY_ONLY" = 1 ]; then
  echo "Verified: $REPO -> $DIR (conversion skipped)"
  exit 0
fi

echo ">> converting to GGML f16"
mkdir -p "$WORK_DIR/$NAME"
"$VENV_PY" "$CONVERTER" "$DIR" "$WHISPER_SRC" "$OUT_DIR/$NAME"

echo "Done. Model: $OUT_DIR/$NAME/ggml-model.bin ($(stat -c%s "$OUT_DIR/$NAME/ggml-model.bin") bytes)"
