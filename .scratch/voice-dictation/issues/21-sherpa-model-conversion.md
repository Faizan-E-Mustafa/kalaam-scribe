# TICKET 21: sherpa-onnx model conversion (Roman Urge q4_0)

## Goal
Convert the existing GGML Roman Urdu q4_0 model to ONNX format for use with sherpa‑onnx on Android.

## Acceptance Criteria
- [ ] GGML model `ggml-model-q4_0.bin` converted to ONNX using `optimum` or equivalent tooling
- [ ] ONNX model files (`encoder_model.onnx`, `decoder_model_merged.onnx`) placed in the project's model directory
- [ ] Conversion script documented and tested on the laptop (arm64 Linux)
- [ ] Verified that sherpa‑onnx can load the converted ONNX model

## Notes
- The Roman Urdu model is the primary target; English Tiny/Small ONNX models already exist on HuggingFace and need no conversion.
- Estimated effort: 2‑3 hours on a host machine with Python / optimum installed.

## Dependencies
- Host machine: Python 3.10+, `optimum[onnxruntime]`
- Target: Android A50 (arm64‑v8a)

## Labels
roman-urdu, ONNX, sherpa-onnx, conversion