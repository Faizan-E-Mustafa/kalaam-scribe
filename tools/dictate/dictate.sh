#!/data/data/com.termux/files/usr/bin/bash
#
# dictate — record a WhatsApp dictation clip and put the transcript on the
# Android clipboard (ticket 03).
#
# Flow: tap (via Termux:Widget) -> record until Stop -> transcribe with
# whisper.cpp base.en -> copy transcript to clipboard -> "Copied" notification
# -> user long-press-pastes into WhatsApp -> user sends manually.
#
# Stop is hands-free: an ongoing notification shows a Stop button; the volume
# key on some builds also stops recording. No typing required.
#
# Usage:
#   dictate start   begin recording (shows the ongoing "Tap Stop" notification)
#   dictate stop    stop recording, transcribe, copy to clipboard, notify
#   dictate <file>  transcribe an existing audio file to the clipboard

set -euo pipefail

PREFIX="/data/data/com.termux/files/usr"
BIN="${PREFIX}/bin"
HOME_DIR="${HOME:-${PREFIX}/home}"

# ---- paths (adjust if your install differs) --------------------------------
WHISPER_CLI="${WHISPER_CLI:-${HOME_DIR}/whisper.cpp/build/bin/whisper-cli}"
MODEL="${MODEL:-${HOME_DIR}/whisper.cpp/models/ggml-base.en.bin}"
WORK="${WORK:-/data/data/com.termux/files/usr/tmp/dictate}"
RAW_WAV="${WORK}/raw.wav"     # recorded by Termux:API (m4a/aac)
WORK_WAV="${WORK}/clip.wav"   # 16 kHz mono for whisper
OUT_TXT="${WORK}/clip.txt"

NOTIF_ID="dictate-rec"
LOG="${WORK}/dictate.log"

mkdir -p "$WORK"

log() { printf '%s %s\n' "$(date '+%H:%M:%S')" "$*" >>"$LOG"; }
err() { log "ERROR: $*"; termux-notification --id "$NOTIF_ID" --title "Dictation failed" --content "$*"; }

# notify "title" "content" — a high-priority one-shot, removed from the tray
# so the "Copied" result shows cleanly.
notify() {
  termux-notification --priority high --title "$1" --content "$2" \
    --button1 "Dismiss" --button1-action "termux-notification-remove --id done" --id done
}

# transcribe_to_clipboard <audio> : the core local STT step.
transcribe_to_clipboard() {
  local src="$1"
  # whisper.cpp wants 16 kHz mono PCM; Termux:API gives us an .m4a/.aac.
  ffmpeg -y -loglevel error -i "$src" -ar 16000 -ac 1 -c:a pcm_s16le "$WORK_WAV"

  "$WHISPER_CLI" -m "$MODEL" -f "$WORK_WAV" -l en -nt >"$OUT_TXT" 2>>"$LOG"
  local text
  text="$(sed -e 's/^\[[^]]*\] *//' "$OUT_TXT" | tr '\n' ' ' | sed -e 's/  */ /g' -e 's/^ //' -e 's/ $//')"

  if [[ -z "$text" ]]; then
    err "No speech detected"
    exit 1
  fi

  printf '%s' "$text" | termux-clipboard-set
  log "copied: $text"
  notify "Copied" "$text"
}

do_start() {
  log "start"
  # remove any previous recording so the recorder won't refuse to overwrite it
  rm -f "${RAW_WAV}.m4a" "$WORK_WAV" "$OUT_TXT"
  termux-notification --id "$NOTIF_ID" --ongoing --priority max \
    --title "Recording… tap Stop when done" \
    --content "Speak your message now." \
    --button1 "Stop" --button1-action "${BIN}/dictate stop"
  # Record until stopped. Termux:API shows its own recording indicator.
  termux-microphone-record -f "${RAW_WAV}.m4a" -l 0
}

do_stop() {
  log "stop"
  termux-microphone-record -q >/dev/null 2>&1 || true
  termux-notification-remove --id "$NOTIF_ID" >/dev/null 2>&1 || true
  local src
  src="${RAW_WAV}.m4a"
  if [[ ! -f "$src" || ! -s "$src" ]]; then
    err "No recording found"
    exit 1
  fi
  transcribe_to_clipboard "$src"
}

case "${1:-}" in
  start) do_start ;;
  stop)  do_stop ;;
  *)
    if [[ -n "${1:-}" && -f "$1" ]]; then
      transcribe_to_clipboard "$1"
    else
      echo "usage: dictate start | stop | <audio-file>"
      exit 1
    fi
    ;;
esac
