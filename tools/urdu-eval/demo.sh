#!/usr/bin/env bash
# Interactive live-voice demo: record one clip, transcribe with all 4 Urdu
# models (-l ur), and write the results to a UTF-8 HTML report that renders
# Urdu script correctly in a browser.
#
# Run in your own terminal. After each recording the report opens in your
# default browser so you can visually check the Urdu output. Loop repeats.

set -euo pipefail

WHISPER_CLI=/tmp/opencode/whisper-bin-ubuntu-x64/whisper-cli
LD_LIBRARY_PATH="/home/femustafa/projects/learning_ws/tools/validate_stt/.venv/lib/python3.11/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"
export LD_LIBRARY_PATH

OUT_DIR="${URDU_EVAL_DIR:-/var/tmp/urdu-eval}/clips"
REPORT="${URDU_EVAL_DIR:-/var/tmp/urdu-eval}/report_live.html"
CLIP="$OUT_DIR/live_demo.wav"
MEDIA_DIR="${URDU_EVAL_DIR:-/var/tmp/urdu-eval}/out"

declare -A MODELS
declare -A MODELS_Q
MODELS["base (kingabzpro)"]="$MEDIA_DIR/base_king/ggml-model.bin"
MODELS["small (Qasim)"]="$MEDIA_DIR/small_Qasim/ggml-model.bin"
MODELS["hybrid (sny)"]="$MEDIA_DIR/hybrid_sny/ggml-model.bin"
MODELS["multilingual (openai)"]="$MEDIA_DIR/oai_small/ggml-model.bin"
MODELS_Q["base (kingabzpro)"]="$MEDIA_DIR/base_king/ggml-model-q4_0.bin"
MODELS_Q["small (Qasim)"]="$MEDIA_DIR/small_Qasim/ggml-model-q4_0.bin"
MODELS_Q["hybrid (sny)"]="$MEDIA_DIR/hybrid_sny/ggml-model-q4_0.bin"
MODELS_Q["multilingual (openai)"]="$MEDIA_DIR/oai_small/ggml-model-q4_0.bin"

transcribe() {
  local model="$1"
  "$WHISPER_CLI" -m "$model" -f "$CLIP" -l ur -nt 2>/dev/null | sed '/^[[:space:]]*$/d'
}

open_report() {
  if command -v explorer.exe >/dev/null 2>&1; then
    explorer.exe "$REPORT" 2>/dev/null || true
  fi
}

echo "=============================================================="
echo " Live Urdu dictation demo — 4 models, one live clip at a time"
echo "=============================================================="

while true; do
  echo
  echo ">>> Press ENTER to START recording, speak, then press ENTER again to STOP. <<<"
  read -r _
  parecord --device="RDPSource" --file-format=wav --rate=44100 --channels=1 "$CLIP" &
  PID=$!
  read -r _
  kill "$PID" 2>/dev/null
  wait "$PID" 2>/dev/null || true

  SZ=$(stat -c%s "$CLIP")
  DUR=$(awk "BEGIN{printf \"%.1f\", $SZ/88200}")
  echo "Recorded $DUR s ($SZ bytes). Transcribing with all models (this takes a moment)..."
  echo

  {
    echo '<!DOCTYPE html><html lang="ur" dir="rtl"><head><meta charset="utf-8">'
    echo '<title>Urdu Model Demo</title>'
    echo '<style>'
    echo 'body{font-family:"Noto Nastaliq Urdu","Jameel Noori Nastaleeq","Noto Naskh Arabic",serif;'
    echo '      font-size:34px;line-height:2;padding:24px;background:#fafafa;color:#111}'
    echo 'h1,h2{font-family:sans-serif;direction:ltr;text-align:left}'
    echo 'h1{font-size:22px}h2{font-size:16px;color:#555;margin-top:26px;border-bottom:1px solid #ddd}'
    echo '.src{font-size:18px;color:#777;direction:ltr;text-align:left;font-family:sans-serif}'
    echo '.tag{font-size:16px;color:#888;direction:ltr;font-family:sans-serif}'
    echo '.ok{color:#1a7f37}.warn{color:#b35900}.bad{color:#c62828}'
    echo '.box{background:#fff;border:1px solid #eee;border-radius:8px;padding:10px 18px;margin:8px 0}'
    echo '</style></head><body>'
    echo "<h1>Live Urdu dictation — clip of $DUR s</h1>"

    for label in "${!MODELS[@]}"; do
      model=${MODELS[$label]}
      echo "<h2>${label}</h2>"
      echo "<div class='box'><span class='tag'>f16  :</span>&nbsp; $(transcribe "$model")</div>"
      if [ -f "${MODELS_Q[$label]}" ]; then
        echo "<div class='box'><span class='tag'>q4_0 :</span>&nbsp; $(transcribe "${MODELS_Q[$label]}")</div>"
      fi
    done

    echo '</body></html>'
  } > "$REPORT"

  echo "Report written to: $REPORT"
  open_report
  echo
  echo "--------------------------------------------------------------"
  printf 'Record another clip? (Enter=again, q=quit): '
  read -r again
  [[ "$again" == "q" ]] && break
done
echo "Done."
