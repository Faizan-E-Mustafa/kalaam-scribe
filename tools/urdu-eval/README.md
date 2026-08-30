# Urdu-script model evaluation

Interactive harness for comparing Urdu-script Whisper fine-tunes on **live
microphone voice** (via the WSLg mic bridge) before picking one to deploy to the
A50.

## Scripts

- `demo.sh` — interactive live-voice demo. Press Enter to start recording, speak,
  press Enter again to stop, then it transcribes that one clip with every model
  under `-l ur` and writes the results to a UTF-8 HTML report that renders Urdu
  correctly in a browser (terminal fonts can't shape Arabic script). Repeats
  until you type `q`, or opens the report each time.
- `record_clip.sh` — one-off helper that records a single clip to a given WAV.
- `fetch_and_convert.sh` — download a HF Whisper fine-tune and convert it to
  GGML f16. Resolves the LFS `model.safetensors`, verifies its integrity
  (truncated downloads otherwise fail conversion silently later), writes
  `vocab.json`/`added_tokens.json` from `tokenizer.json` when missing, then runs
  the converter. `--verify-only` skips conversion.

  ```
  ./fetch_and_convert.sh Qasimhassan65/whisper-small-urdu small_Qasim
  ```

## Requirements

- `parecord` + `libpulse` (WSLg PulseAudio client) to read the mic bridge
  (`RDPSource`, `PULSE_SERVER`).
- Prebuilt `whisper-cli` and torch's `libgomp.so.1` (see
  `../roman-urdu/converters/README.md`).

## Models compared

| Label                    | Source                                | Size class |
|--------------------------|---------------------------------------|------------|
| base (kingabzpro)        | `kingabzpro/whisper-base-urdu-full`   | base       |
| small (Qasim)            | `Qasimhassan65/whisper-small-urdu`    | small      |
| hybrid (sny)             | `sny2ksa/whisper-urdu-ultimate-hybrid-v2` | small |
| multilingual (openai)    | `openai/whisper-small` (baseline)     | small      |

Converted f16 GGML models and clips live under `URDU_EVAL_DIR`
(default `/var/tmp/urdu-eval`), not in this repo (gitignored).

## Usage

```
URDU_EVAL_DIR=/var/tmp/urdu-eval ./demo.sh
```

## Field notes

- **Language flag:** these Urdu-script fine-tunes are run with `-l ur`.
  `-l auto` makes Qasim revert to Roman/Latin (Assalamu alaikum); `-l ur` gives
  correct Urdu script (السلام علیکم).
- **Script display:** raw Urdu output looks broken in most terminals because the
  font can't shape Arabic script; always view via the generated HTML report.
- The base multilingual model tends to hallucinate longer rambling output on
  short/quiet clips.
