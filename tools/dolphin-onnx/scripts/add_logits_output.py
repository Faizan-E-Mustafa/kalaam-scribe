"""Graph surgery on DakeQQ's Dolphin decoder: expose the FULL output logits.

DakeQQ's exported decoder.onnx only outputs `max_logit_id` (the argmax token id).
That is enough for greedy decoding but NOT for attention beam search. This script
appends two earlier tensors to the model's output list, both produced by the
output layer:

  * `/output_layer/Gemm_output_0`  — the FULL logits over the whole vocabulary.
    Feed this to beam search. (DakeQQ's own greedy code uses only max_logit_id.)
  * `/Slice_3_output_0`            — the language-sliced logits. The
    language_start/language_end inputs slice the logits to the language-token
    range [136:269]; correct for LID/auto-detect but WRONG for content: beam
    search on it yields language-token garbage (<tk><ml><mrj>...).

Nodes are detected by structure, not hardcoded names, so it works for
dolphin-base and dolphin-small as long as the graph keeps the same shape:
ArgMax's input = sliced logits; the Gemm feeding ArgMax (via the output layer) =
full logits.

Usage:
  python add_logits_output.py decoder.onnx decoder_withlogits.onnx
"""
import sys

import onnx
from onnx import helper, shape_inference
from onnx import TensorProto


def inferred_shape(graph, name):
    """Return a TensorShapeProto for `name` from shape-inferred value_info, or None."""
    for vi in graph.value_info:
        if vi.name == name and vi.type.HasField("tensor_type"):
            return vi.type.tensor_type.shape
    return None


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else "decoder.onnx"
    dst = sys.argv[2] if len(sys.argv) > 2 else "decoder_withlogits.onnx"

    model = onnx.load(src)
    g = model.graph

    # Locate by structure: ArgMax's input is the language-sliced logits; the
    # Gemm feeding that Slice is the full logits.
    sliced_logits = full_logits = None
    slice_input_of = None
    for n in g.node:
        if n.op_type == "ArgMax":
            sliced_logits = n.input[0]          # e.g. /Slice_3_output_0
        if n.op_type == "Slice":
            slice_input_of = n
    for n in g.node:
        if n.op_type == "Gemm" and slice_input_of is not None \
           and n.output[0] == slice_input_of.input[0]:
            full_logits = n.output[0]           # e.g. /output_layer/Gemm_output_0

    if sliced_logits is None:
        print("ERROR: no ArgMax node found; is this a DakeQQ decoder?")
        sys.exit(1)
    if full_logits is None:
        print("ERROR: could not find the full-logits Gemm feeding the ArgMax Slice")
        sys.exit(1)

    existing = {o.name for o in g.output}
    to_add = [t for t in (full_logits, sliced_logits) if t not in existing]
    if not to_add:
        print("logits outputs already present; nothing to do")
        return

    # Shape-infer so we can declare real output shapes (onnx checker rejects None).
    try:
        inf = shape_inference.infer_shapes(model)
        vi_map = {vi.name: vi for vi in inf.graph.value_info}
    except Exception:
        inf, vi_map = model, {}

    for t in to_add:
        if t in vi_map:
            g.output.append(helper.make_value_info(t, vi_map[t].type))
            print(f"  output {t} shape: {vi_map[t].type.tensor_type.shape}")
        else:
            g.output.append(helper.make_tensor_value_info(t, TensorProto.FLOAT, None))
            print(f"  output {t}: (shape unavailable, using float32)")

    onnx.checker.check_model(model)
    onnx.save(model, dst)
    print(f"added outputs: {to_add}")
    print(f"wrote {dst}")


if __name__ == "__main__":
    main()
