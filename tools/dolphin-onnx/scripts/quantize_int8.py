#!/usr/bin/env python3
"""Quantize Dolphin attention fp32 ONNX pairs to int8.

Runs graph-surgery (add logits output) + onnxruntime dynamic int8 quantization
for both encoder and decoder of base and small variants.

Usage:
  python quantize_int8.py

Source files (read):
  ~/.cache/dolphin/dakeqq/dolphin-base/dolphin-base-encoder.onnx
  ~/.cache/dolphin/dakeqq/dolphin-base/dolphin-base-decoder.onnx
  ~/.cache/dolphin/dakeqq/dolphin-small/dolphin-small-encoder.onnx
  ~/.cache/dolphin/dakeqq/dolphin-small/dolphin-small-decoder.onnx

Output files (written):
  ~/.cache/dolphin/dakeqq/int8-base/encoder.onnx
  ~/.cache/dolphin/dakeqq/int8-base/decoder.onnx
  ~/.cache/dolphin/dakeqq/int8-small/encoder.onnx
  ~/.cache/dolphin/dakeqq/int8-small/decoder.onnx
"""

import os
import sys
import shutil
from pathlib import Path

import onnx
from onnx import helper, shape_inference, TensorProto

# Use uv-run python's onnxruntime
from onnxruntime.quantization import QuantType, quantize_dynamic


HOME = Path.home()
CACHE = HOME / ".cache" / "dolphin" / "dakeqq"
OUT_BASE = CACHE / "int8-base"
OUT_SMALL = CACHE / "int8-small"


def add_logits_output_and_save(src_path: str, dst_path: str) -> None:
    """Graph surgery: expose full-logits output on a DakeQQ decoder ONNX, save to dst."""
    model = onnx.load(src_path)
    g = model.graph

    full_logits = None
    slice_input_of = None
    for n in g.node:
        if n.op_type == "ArgMax":
            pass
        if n.op_type == "Slice":
            slice_input_of = n

    for n in g.node:
        if (n.op_type == "Gemm" and slice_input_of is not None
                and n.output[0] == slice_input_of.input[0]):
            full_logits = n.output[0]

    if full_logits is None:
        raise RuntimeError(f"Could not find full-logits Gemm in {src_path}")

    existing = {o.name for o in g.output}
    if full_logits not in existing:
        try:
            inf = shape_inference.infer_shapes(model)
            vi_map = {vi.name: vi for vi in inf.graph.value_info}
        except Exception:
            inf, vi_map = model, {}

        if full_logits in vi_map:
            g.output.append(helper.make_value_info(full_logits, vi_map[full_logits].type))
        else:
            g.output.append(helper.make_tensor_value_info(full_logits, TensorProto.FLOAT, None))

    onnx.checker.check_model(model)
    onnx.save(model, dst_path)
    print(f"  surgery saved: {dst_path}  ({os.path.getsize(dst_path) / 1e6:.1f} MB)")


def quantize_and_save(src: str, dst: str, label: str):
    """Quantize an ONNX model to int8 dynamic."""
    print(f"\n[quantize] {label}")
    print(f"  input : {src}  ({os.path.getsize(src) / 1e6:.1f} MB)")

    if "decoder" in Path(src).name:
        tmp_surged = src + ".surged.tmp.onnx"
        print(f"  surgery: adding logits output...")
        add_logits_output_and_save(src, tmp_surged)
        quantize_dynamic(
            model_input=tmp_surged,
            model_output=dst,
            op_types_to_quantize=["MatMul"],
            weight_type=QuantType.QInt8,
        )
        os.remove(tmp_surged)
    else:
        quantize_dynamic(
            model_input=src,
            model_output=dst,
            op_types_to_quantize=["MatMul"],
            weight_type=QuantType.QInt8,
        )

    size_mb = os.path.getsize(dst) / 1e6
    ratio = size_mb / (os.path.getsize(src) / 1e6)
    print(f"  output: {dst}  ({size_mb:.1f} MB,  {ratio:.1%} of fp32)")


def main():
    variants = [
        ("base", CACHE / "dolphin-base", OUT_BASE),
        ("small", CACHE / "dolphin-small", OUT_SMALL),
    ]

    for name, src_dir, out_dir in variants:
        out_dir.mkdir(parents=True, exist_ok=True)

        enc_src = src_dir / f"dolphin-{name}-encoder.onnx"
        enc_dst = out_dir / "encoder.onnx"
        dec_src = src_dir / f"dolphin-{name}-decoder.onnx"
        dec_dst = out_dir / "decoder.onnx"

        quantize_and_save(str(enc_src), str(enc_dst), f"{name} encoder")
        quantize_and_save(str(dec_src), str(dec_dst), f"{name} decoder")

        # Copy units.txt from the fp16 variant (same vocab)
        units_src = CACHE / f"{name}-fp16-arm" / "units.txt"
        if not units_src.exists():
            units_src = CACHE / name / "units.txt"
        units_dst = out_dir / "units.txt"
        if units_src.exists():
            shutil.copy2(units_src, units_dst)
            print(f"  copied units.txt → {units_dst}")
        else:
            print(f"  WARNING: no units.txt found at {units_src}")

        print(f"\n[{name}] output in {out_dir}:")
        for f in sorted(out_dir.iterdir()):
            print(f"  {f.name}  ({f.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
