# TICKET 21: sherpa-onnx model conversion (Roman Urdu)

## Goal
Convert the `cheetos18/whisper-small-roman-urdu` HuggingFace model to
sherpa-onnx-compatible ONNX format (encoder + decoder + tokens.txt) and enable
it as a downloadable catalog model in the Android app's ONNX tier.

## Status — DONE

All acceptance criteria met; Android catalog entries and tests updated.

### What was done

1. **Conversion script** — `tools/roman-urdu/converters/convert-h5-to-onnx.py`
   Converts `cheetos18/whisper-small-roman-urdu` (safetensors, not GGML) to
   sherpa-onnx's custom ONNX format: `AudioEncoderTensorCache` /
   `TextDecoderTensorCache` wrappers from `openai-whisper`, with a full HF ↔
   whisper weight-name mapping. Key fix: `dynamo=False` (legacy TorchScript
   exporter) avoids OOM + external `.onnx.data` files on 8 GB laptops.

2. **Converted files** — `tools/roman-urdu/onnx-out/`
   | File | Size |
   |---|---|
   | `roman-urdu-encoder.onnx` | 391 MB |
   | `roman-urdu-decoder.onnx` | 534 MB |
   | `roman-urdu-encoder.int8.onnx` | 108 MB |
   | `roman-urdu-decoder.int8.onnx` | 251 MB |
   | `roman-urdu-tokens.txt` | 798 KB |

   Hosted naming is identity-based (`roman-urdu-*.{onnx,tokens.txt}`), shared
   across fp32/int8 tiers. Local naming in the app uses id-based
   (`roman-urdu-fp32-*`, `roman-urdu-int8-*`).

3. **Sherpa-onnx smoke test** — both fp32 and int8 models load and decode a
   Roman-Urdu sample (`rec.wav`) via `OfflineRecognizer.from_whisper()`:
   ```
   fp32: 'aap ka kya haal hai?'
   int8: 'aap ka kya haal hai'
   ```

4. **Android catalog wiring** (`ModelCatalog.kt`)
   Two new `onnxModels` entries added:
   - `roman-urdu-fp32` — fp32 tier, `LanguageMode.RomanUrdu`, approxSizeMb 925
   - `roman-urdu-int8` — int8 tier, `LanguageMode.RomanUrdu`, approxSizeMb 359
   Both point at `$RU_HF/roman-urdu-{encoder,encoder.int8}.onnx`.

5. **`isRomanUrdu()` updated** (`Model.kt`) — covers the two new ONNX ids so
   `canOverrideLanguage` stays false.

6. **Tests updated and passing**
   - `ModelCatalogTest.onnxCatalogHasSixIdentitiesInTwoPrecisionTiers` (5→6
     identities, 10→12 entries)
   - `ModelCatalogTest.romanUrduOnnxModelsUseFixedEnglishAndProjectRepo`
     (new test)
   - `ModelTest.isRomanUrduMatchesRomanUrduIds` (covers `roman-urdu-fp32`,
     `roman-urdu-int8`)
   - `ModelTest.englishAndRomanUrduCannotOverrideLanguage` (covers ONNX
     roman-urdu entry)

## Acceptance Criteria — DONE

- [x] GGML model converted to ONNX using optimum or equivalent tooling
- [x] ONNX model files (`encoder.onnx`, `decoder.onnx`, `tokens.txt`) placed in
      project's model directory
- [x] Conversion script documented and tested on the laptop (arm64 Linux)
- [x] Verified that sherpa-onnx can load the converted ONNX model

## Notes

- The source is the HuggingFace safetensors checkpoint (`model.safetensors`),
  not the GGML `ggml-model-q4_0.bin`. Conversion is: safetensors → openai-whisper
  state dict → custom ONNX export (sherpa-onnx graph).
- The sherpa-onnx export requires a custom graph (cross-attention KV caches) that
  `optimum` does not produce; must follow `sherpa-onnx/scripts/whisper/export-onnx.py`.
- Roman-Urdu always runs `language="en"` (never `ur`, which leaks Urdu script
  tokens). This is handled by `LanguageMode.RomanUrdu` → `resolveLanguage` → `"en"`.

## Dependencies
- Host machine: Python 3.10+, `optimum[onnxruntime]`, `openai-whisper`, `sherpa-onnx`
- Target: Android arm64-v8a
