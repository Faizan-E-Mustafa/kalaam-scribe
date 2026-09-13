#!/data/data/com.termux/files/usr/bin/bash
#
# bench — benchmark whisper.cpp models on the phone for a recorded clip (ticket 04).
#
# Usage (on the phone, in Termux):
#   bench <audio-file> [texts...]
#
# Converts the audio to 16 kHz mono WAV, then runs whisper-cli with each of the
# three candidate models (tiny/base/small, English), timing each run. For each
# model it prints the wall-clock time (real time) and the raw transcript so the
# user can judge qualitative accuracy. Results are appended to a results file.
#
# The clip length (in seconds) is printed too, so "real-time factor" can be
# eyeballed: RTF = transcript_time / clip_length. RTF < 1 means faster than real
# time.
#
# Models are looked up from the standard whisper.cpp install. Override per-model
# with e.g. BENCH_MODEL_TINY=/path/to/ggml-tiny.en.bin

set -uo pipefail

PREFIX="/data/data/com.termux/files/usr"
export PATH="${PREFIX}/bin:$PATH"
export HOME="${HOME:-${PREFIX}/home}"

WHISPER_CLI="${WHISPER_CLI:-${HOME}/whisper.cpp/build/bin/whisper-cli}"
MODELS_DIR="${HOME}/whisper.cpp/models"
MODEL_TINY="${BENCH_MODEL_TINY:-${MODELS_DIR}/ggml-tiny.en.bin}"
MODEL_BASE="${BENCH_MODEL_BASE:-${MODELS_DIR}/ggml-base.en.bin}"
MODEL_SMALL="${BENCH_MODEL_SMALL:-${MODELS_DIR}/ggml-small.en.bin}"

WORK="${WORK:-${PREFIX}/tmp/bench}"
WAV="${WORK}/bench.wav"
RESULTS="${WORK}/bench-results.txt"

mkdir -p "$WORK"

if [[ $# -lt 1 ]]; then
  echo "usage: bench <audio-file>" >&2
  exit 1
fi
SRC="$1"

# Convert once to 16 kHz mono PCM; each model transcribes the same clip.
ffmpeg -y -loglevel error -i "$SRC" -ar 16000 -ac 1 -c:a pcm_s16le "$WAV" || {
  echo "ffmpeg conversion failed for: $SRC" >&2
  exit 1
}

DUR="$(ffprobe -v error -show_entries format=duration -of default=nw=1:nk=1 "$WAV" 2>/dev/null | cut -d. -f1)"
[[ -n "$DUR" ]] || DUR="?"

{
  echo "===== $(date)   src=$SRC   clip_len=${DUR}s ====="
  echo "results file: $RESULTS"
} | tee -a "$RESULTS"

run_model() {
  local label="$1" model="$2"
  if [[ ! -f "$model" ]]; then
    echo "-- $label ($model) NOT FOUND" | tee -a "$RESULTS"
    return
  fi
  echo "-- $label (${model##*/})" | tee -a "$RESULTS"
  local start end
  start="$(date +%s.%N)"
  "$WHISPER_CLI" -m "$model" -f "$WAV" -l en -nt -t "${BENCH_THREADS:-8}" 2>/dev/null \
    | sed -e 's/^\[[^]]*\] *//' | tr '\n' ' ' | sed -e 's/  */ /g' -e 's/^ //' -e 's/ $//' \
    | tee -a "$RESULTS"
  end="$(date +%s.%N)"
  printf "\ntranscription time: %.2fs (clip=${DUR}s -> RTF≈%.2f)\n" \
    "$(echo "$end $start" | awk '{print $1-$2}')" \
    "$(echo "$end $start $DUR" | awk '{print ($1-$2)/$3}')" | tee -a "$RESULTS"
  echo | tee -a "$RESULTS"
}

run_model "tiny"  "$MODEL_TINY"
run_model "base"  "$MODEL_BASE"
run_model "small" "$MODEL_SMALL"

echo "Done. Full log: $RESULTS" | tee -a "$RESULTS"
