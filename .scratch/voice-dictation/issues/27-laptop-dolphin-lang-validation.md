# 27: Laptop validation of Dolphin lang/region (no ONNX)

**What to build:** On the Linux dev laptop, prove that Dolphin's attention decoder
respects explicit language/region tokens by running the upstream `dolphin` pip
package (PyTorch, no ONNX, no sherpa-onnx) on a live mic-recorded Urdu clip. Compare
auto-detect vs pinned `lang=ur, region=PK` output and confirm pinned decoding yields
correct Urdu script with no cross-script leakage. This is a read-only validation gate:
it decides whether the smartphone ONNX export (ticket 28) is worth pursuing.

**Blocked by:** None (can start immediately).

**Status:** ready-for-agent

- [ ] Install `ffmpeg` in WSL2 (Dolphin's `load_audio` shells out to ffmpeg).
- [ ] Install `dataoceanai-dolphin` into `tools/validate_stt/.venv`; if the `funasr`
      dependency fails to build, fall back to installing Dolphin from source with
      `--no-deps` plus `modelscope`, `addict`, `pydub` (short clips never run VAD).
- [ ] `load_model('base', device='cpu')` and `load_model('small', device='cpu')` both
      succeed (first run downloads from ModelScope to `~/.cache/dolphin/`).
- [ ] Record an Urdu utterance from the WSL2 mic (PulseAudio via WSLg) to a 16 kHz
      mono WAV via `parecord`/ffmpeg.
- [ ] A script runs `dolphin.transcribe(model, wav, lang_sym='ur', region_sym='PK',
      decoding_method='attention', predict_time=False, word_timestamp=False)` and
      prints `language`, `region`, `text`, `text_nospecial`; and a second run with
      no lang/region (auto-detect) on the same clip.
- [ ] Pinned `ur`/`PK` run reports `language='ur'`, `region='PK'`, and text in correct
      Urdu script; auto-detect run is captured for comparison (it may guess wrong,
      reproducing k2-fsa/sherpa-onnx#3904).
- [ ] Results documented and conclusion recorded: does lang/region genuinely fix
      script leakage for `ur`/`PK`? This gates ticket 28.

## Comments
