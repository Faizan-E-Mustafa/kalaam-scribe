#!/usr/bin/env bash
# Record a live-voice Urdu clip via the WSLg mic bridge (RDPSource).
#
# Usage:
#   ./record_clip.sh <output.wav> [duration_seconds]
#
# It waits for you to press ENTER, then records for DURATION seconds (default 6)
# while you speak directly into the mic, saving the WAV to <output.wav> at the
# native 44.1 kHz mono rate.
#
# Run it in your own terminal so you control when you speak. Example:
#   ./record_clip.sh /var/tmp/urdu-eval/clips/clip_x.wav 6
#
# A clip already exists at /var/tmp/urdu-eval/clips; use a new filename each
# time so you don't overwrite a good take.

set -euo pipefail

OUT="${1:?usage: $0 <output.wav> [duration_seconds]}"
DUR="${2:-6}"

printf 'Record: %s for %ss\n' "$OUT" "$DUR"
printf 'Press ENTER when you are ready to speak, then speak immediately...\n'
read -r _

printf '>>> RECORDING for %ss now. SPEAK INTO THE MIC <<<\n' "$DUR"
parecord --device="RDPSource" --file-format=wav --rate=44100 --channels=1 "$OUT" &
PID=$!
sleep "$DUR"
kill "$PID" 2>/dev/null
wait "$PID" 2>/dev/null || true

SZ=$(stat -c%s "$OUT" 2>/dev/null || echo 0)
printf 'Saved %s (%s bytes). Done.\n' "$OUT" "$SZ"
