# 27: Laptop validation of Dolphin lang/region (no ONNX)

**What to build:** On the Linux dev laptop, prove that Dolphin's attention decoder
respects explicit language/region tokens by running the upstream `dolphin` pip
package (PyTorch, no ONNX, no sherpa-onnx) on a live mic-recorded Urdu clip. Compare
auto-detect vs pinned `lang=ur, region=PK` output and confirm pinned decoding yields
correct Urdu script with no cross-script leakage. This is a read-only validation gate:
it decides whether the smartphone ONNX export (ticket 28) is worth pursuing.

**Blocked by:** None (can start immediately).

**Status:** ready-for-agent

- [x] ffmpeg optional — this Dolphin build loads audio via `torchaudio` (patched to
      `soundfile` because its `torchcodec` backend is broken with torch 2.13); no
      ffmpeg/sudo needed.
- [x] Install `dataoceanai-dolphin` into `tools/validate_stt/.venv`; `funasr` etc.
      installed fine (no `--no-deps` fallback needed).
- [x] `load_model('base', device='cpu')` and `load_model('small', device='cpu')` both
      succeed (downloaded from ModelScope to `~/.cache/dolphin/`).
- [ ] Record an Urdu utterance from the WSL2 mic (PulseAudio via WSLg) to a 16 kHz
      mono WAV via `parecord`/ffmpeg. — **blocked by environment**: WSLg `RDPSource`
      captured only silence (rms ~0.002); used real Urdu sample `rec.wav` instead.
- [x] A script runs `dolphin.transcribe(model, wav, lang_sym='ur', region_sym='PK',
      decoding_method='attention', predict_time=False, word_timestamp=False)` and
      prints `language`, `region`, `text`, `text_nospecial`; and a second run with
      no lang/region (auto-detect) on the same clip.
- [x] Pinned `ur`/`PK` run reports `language='ur'`, `region='PK'`, and text in correct
      Urdu script; auto-detect run is captured for comparison (it may guess wrong,
      reproducing k2-fsa/sherpa-onnx#3904).
- [x] Results documented and conclusion recorded: does lang/region genuinely fix
      script leakage for `ur`/`PK`? This gates ticket 28.

## Results (2026-09-03)

**Environment**: `dataoceanai-dolphin==20260513` installed into
`tools/validate_stt/.venv` via `uv`. Audio loading needs `torchaudio`; torch's
`torchcodec` backend is broken with torch 2.13, so the validation script patches
`torchaudio.load` to use `soundfile` (already in the venv). No ffmpeg needed.

**Two models validated** (`base`, `small`) on real Urdu speech
`tools/roman-urdu/samples/rec.wav` — "آپ کا کیا حال ہے؟" (How are you?):

| test | model | language | region | text |
|------|-------|----------|--------|------|
| auto-detect | base | ur | **IN** | آپ کا کیا حال ہے؟ |
| pinned ur/PK  | base | ur | PK   | آپ کا کیا حال ہے؟ |
| pinned ru/RU  | base | ru | RU   | آپ کا کیا حال ہے? (؟→? script shift) |
| auto-detect | small | ur | **IN** | آپ کا کیا حال ہے؟ |
| pinned ur/PK  | small | ur | PK   | آپ کا کیا حال ہے؟ |

Control: `jfk.wav` (clear English) auto-detected as `ru`/RU in Dolphin — a live
reproduction of the sherpa #3904 wrong-script problem. English isn't in the model's
supported-set at all; `ur-PK` is.

### First live-mic pass (default `attention_rescoring` — misleading)
Recorded the user's live voice via the WSLg mic. The 29s clip (`live5.wav`) and a
single sentence (`live6.wav`). This first pass used the `transcribe()` default
`attention_rescoring`, which wrongly flipped output to Devanagari:

| test | model | language | region | text |
|------|-------|----------|--------|------|
| auto-detect | base | hi | IN | मैं थोड़ी देर में ऑफिस आऊंगा … (Devanagari) |
| pinned ur/PK  | base | ur | PK | same Devanagari text (metadata flips) |
| auto-detect | base | bn | **BD** | थोड़ीদের मे अফিস সنگ! (Bengali+Devanagari garbled) |
| pinned ur/PK  | base | ur | PK | same garbled text (metadata flips) |
| auto-detect | small | hi | IN | थोड़ी देर में ऑफिस जाऊंगा। (Devanagari) |
| pinned ur/PK  | small | ur | PK | थोड़ी दे में ऑफिस जाऊंगा। (still Devanagari) |

This is exactly the misleading result that the corrected conclusion below explains —
it was the `attention_rescoring` pass overriding the pin, NOT the model ignoring it.

## Conclusion — GATE PASSED (finding corrected 2026-09-03)

**ROOT-CAUSE FINDING (decides the ONNX design):**
- **`decoding_method='attention'` (plain beam search) respects the `ur`/`PK` pin and
  produces clean Urdu script.** Verified across both `base` and `small` on every live
  clip. When pinned, output is Urdu script with `language='ur', region='PK'`.
- **`decoding_method='attention_rescoring'` (the `dolphin.transcribe()` default)
  OVERRIDES the pin** — its LM rescore pass flips the output to Devanagari/Hindi even
  when ur/PK is pinned. This is why initial live tests showed "still Hindi": the script
  defaulted to rescoring.
- The earlier "soft prior, doesn't force script" conclusion was WRONG — it was an
  artifact of using `attention_rescoring`. With `attention`, the pin consistently forces
  Urdu script.

Confirmed table (`attention` — pinning works):

| model | clip | AUTO-detect | pinned ur/PK |
|-------|------|-------------|--------------|
| base | live6 | bn/BD | ur/PK — تھری دیر میں آفس چونگ |
| base | live5 | hi/IN | ur/PK — میں تھوڑی دیر میں آفس آوں گا… |
| small | live6 | hi/IN | ur/PK — تھوڑی دیر میں آفس جاوں گا |
| small | live5 | hi/IN | ur/PK — میں تھوڑی دیر میں آفس جاوں گا… |

- **AUTO-detect still mislabels** your Urdu (`bn`/`BD` or `hi`/`IN`) — that IS the
  #3904 problem, reproduced live. Pinning `ur`/`PK` + `attention` fixes both the
  metadata AND the script.
- Pinning a **wrong** language (`ru`/RU) on clean Urdu shifted `؟`→`?` — the prefix
  genuinely steers decoding.

**Implication for ticket 28 (ONNX) — CRITICAL:**
- The ONNX pipeline MUST use **beam-search `attention` decoding, NOT
  `attention_rescoring`**. If the ONNX port mirrors the rescoring pass, it will
  reproduce the Hindi-script bug. This is a hard design requirement for the export.

## Comments
- Mic: WSLg `RDPSource` is flaky/intermittent — early captures were silent; a running
  ~30s background `parecord` eventually caught clear voice (peak ~0.97). Reproducible
  live captures are best done via a continuous-recorder-and-trim approach.
- `torchaudio.load` patched to `soundfile` (torchcodec broken with torch 2.13).
- Validation script lives at `/tmp/opencode/dolphin_lang_test.py` (throwaway).
- Live HTML demo: `/tmp/opencode/dolphin_live_demo.sh` (mirrors
  `tools/urdu-eval/demo.sh`). Records one clip per Enter-press, transcribes base+small
  (AUTO vs pinned ur/PK) with `decoding_method='attention'`, writes an RTL Urdu HTML
  report to `/var/tmp/dolphin-eval/report_dolphin.html`. To view on Windows, copy to
  `/mnt/c/Users/femustafa/Desktop/` and open via `cmd.exe /c start` (WSL ext4 paths are
  not reachable from Windows).
