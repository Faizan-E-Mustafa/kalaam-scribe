"""Export Dolphin's attention DECODER to ONNX via torch.jit.trace (CPU, base/small).

This is the working decoder export. It feeds RAW log-mel features to a torch
encoder at runtime; only the decoder is exported. See README.md for the full
story (the encoder could not be exported; see ticket 28 findings).

Interface of the exported graph
  memory       : float32 [B, T', 512]   encoder output
  memory_mask  : bool     [B, T']       2D encoder padding mask
  tgt          : int64    [B, L]        target token ids (SOS + prefix + content)
  logits       : float32 [B, L, vocab]  log-softmax over vocab (vocab = 40002)

Verified numerically: max abs diff vs torch decoder is ~1e-6.

Regenerates `decoder.onnx` in the artifact directory (gitignored).
"""
import argparse
import os

import numpy as np
import torch


def make_pad_mask(lengths, maxlen):
    seq = torch.arange(maxlen, device=lengths.device)[None, :]
    return seq >= lengths[:, None]


def subsequent_mask(size, device="cpu"):
    ret = torch.ones(size, size, dtype=torch.bool, device=device)
    return torch.tril(ret, diagonal=0).unsqueeze(0)


class DecoderWrapper(torch.nn.Module):
    """Wraps the Dolphin decoder to expose (memory, mask, tgt) -> log-probs."""

    def __init__(self, dec):
        super().__init__()
        self.dec = dec

    def forward(self, memory, memory_mask, ys_in_pad):
        B, L = ys_in_pad.shape
        ys_in_lens = torch.full((B,), L, dtype=torch.long, device=ys_in_pad.device)
        pad = ~make_pad_mask(ys_in_lens, L).unsqueeze(1)
        m = subsequent_mask(L, ys_in_pad.device)
        tgt_mask = pad & m
        x = self.dec(memory, memory_mask, ys_in_pad, ys_in_lens)[0]
        return torch.log_softmax(x, dim=-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="base", choices=["base", "small"])
    ap.add_argument("--out", default=None,
                    help="output dir for decoder.onnx (default: ../artifacts)")
    ap.add_argument("--cfg-t", type=int, default=20,
                    help="dummy encoder-time length for tracing (must be > 0)")
    ap.add_argument("--tgt-len", type=int, default=8)
    args = ap.parse_args()

    out_dir = args.out or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "artifacts")
    os.makedirs(out_dir, exist_ok=True)

    os.environ.setdefault("AUDIO_BACKEND", "soundfile")
    os.environ.setdefault("MODELSCOPE_CACHE", os.path.expanduser("~/.cache/modelscope"))
    import dolphin

    model = dolphin.load_model(args.model, device="cpu")
    model.eval()
    dec_w = DecoderWrapper(model.decoder).eval()

    B, Tp, L = 1, args.cfg_t, args.tgt_len
    dummy_mem = torch.randn(B, Tp, 512)
    dummy_mmask = torch.ones(B, Tp, dtype=torch.bool)
    dummy_tgt = torch.randint(3, 100, (B, L))

    with torch.no_grad():
        ref = dec_w(dummy_mem, dummy_mmask, dummy_tgt)
    t_dec = torch.jit.trace(dec_w, (dummy_mem, dummy_mmask, dummy_tgt), check_trace=True)
    with torch.no_grad():
        t_out = t_dec(dummy_mem, dummy_mmask, dummy_tgt)
    print(f"torch-vs-trace max diff: {(t_out - ref).abs().max().item():.3e}")

    path = os.path.join(out_dir, "decoder.onnx")
    torch.onnx.export(
        t_dec, (dummy_mem, dummy_mmask, dummy_tgt), path,
        input_names=["memory", "memory_mask", "tgt"],
        output_names=["logits"],
        dynamic_axes={
            "memory": {0: "batch", 1: "enc_time"},
            "memory_mask": {0: "batch", 1: "enc_time"},
            "tgt": {0: "batch", 1: "tgt_len"},
            "logits": {0: "batch", 1: "tgt_len"},
        },
        opset_version=17,
        dynamo=False,
    )
    print(f"exported {path} ({os.path.getsize(path) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
