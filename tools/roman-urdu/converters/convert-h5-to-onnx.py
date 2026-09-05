#!/usr/bin/env python3
"""Convert the HuggingFace Roman-Urdu Whisper fine-tune to sherpa-onnx ONNX format.

The Roman-Urdu model (`cheetos18/whisper-small-roman-urdu`) is a fine-tune of
OpenAI Whisper small with a 51865-token vocabulary (same architecture and vocab
as multilingual whisper-small). sherpa-onnx serves Whisper models in a custom
ONNX graph format:

  * encoder.onnx
      input:  mel            [n_audio, n_mels, T]
      output: n_layer_cross_k, n_layer_cross_v  [n_text_layer, n_audio, T, n_text_state]

  * decoder.onnx
      input:  tokens, in_n_layer_self_k_cache, in_n_layer_self_v_cache,
              n_layer_cross_k, n_layer_cross_v, offset
      output: logits, out_n_layer_self_k_cache, out_n_layer_self_v_cache

  * tokens.txt  (tiktoken id -> base64 token lines, as written by
                 the openai-whisper tiktoken encoder)

Because the openai-whisper tiktoken tokenizer and the HF tokenizer share the
exact same vocabulary and token IDs for this model, we load the HF weights and
copy them into a stock openai-whisper `Whisper` model, then reuse sherpa-onnx's
tensor-cache export wrappers (AudioEncoderTensorCache / TextDecoderTensorCache)
to emit the ONNX graphs. This mirrors `sherpa-onnx/scripts/whisper/export-onnx.py`.

USAGE:
  python3 convert-h5-to-onnx.py <dir_model> <dir_out> [--quantize]
    <dir_model>  directory containing the HF checkpoint (config.json, vocab.json,
                 added_tokens.json, model.safetensors)
    <dir_out>    output directory for roman-urdu-*.onnx + tokens.txt
    --quantize   also emit int8-quantized encoder/decoder

Outputs:
  <dir_out>/roman-urdu-fp32-encoder.onnx
  <dir_out>/roman-urdu-fp32-decoder.onnx
  <dir_out>/roman-urdu-fp32-tokens.txt   (actually the openai tiktoken asset)
  plus, with --quantize:
  <dir_out>/roman-urdu-int8-encoder.onnx
  <dir_out>/roman-urdu-int8-decoder.onnx
"""

import argparse
import io
import os
import struct
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch import Tensor

import whisper
from whisper.model import (
    AudioEncoder,
    MultiHeadAttention,
    ResidualAttentionBlock,
    TextDecoder,
    disable_sdpa,
)

from transformers import WhisperForConditionalGeneration


# ---------------------------------------------------------------------------
# Weight-name mapping between HF state_dict and openai-whisper state_dict.
#
# HF names look like:  model.encoder.layers.0.self_attn.k_proj.weight
# openai names look like: encoder.blocks.0.attn.key.weight
#
# The map below converts the per-block suffix; the two layer organisations
# (encoder.layers / decoder.layers) also need rewording to blocks.
# ---------------------------------------------------------------------------
BLOCK_SUFFIX_MAP = {
    "self_attn.k_proj": "attn.key",
    "self_attn.q_proj": "attn.query",
    "self_attn.v_proj": "attn.value",
    "self_attn.out_proj": "attn.out",
    "self_attn_layer_norm": "attn_ln",
    "encoder_attn.q_proj": "cross_attn.query",
    "encoder_attn.k_proj": "cross_attn.key",
    "encoder_attn.v_proj": "cross_attn.value",
    "encoder_attn.out_proj": "cross_attn.out",
    "encoder_attn_layer_norm": "cross_attn_ln",
    "fc1": "mlp.0",
    "fc2": "mlp.2",
    "final_layer_norm": "mlp_ln",
}

TOP_LEVEL_MAP = {
    "encoder.layer_norm.bias": "encoder.ln_post.bias",
    "encoder.layer_norm.weight": "encoder.ln_post.weight",
    "encoder.embed_positions.weight": "encoder.positional_embedding",
    "decoder.layer_norm.bias": "decoder.ln.bias",
    "decoder.layer_norm.weight": "decoder.ln.weight",
    "decoder.embed_positions.weight": "decoder.positional_embedding",
    "decoder.embed_tokens.weight": "decoder.token_embedding.weight",
    "proj_out.weight": "proj_out.weight",
}

# HF uses a separate k_proj for cross-attention while openai-whisper does not;
# they are identical kernels, so we alias them.
CROSS_K_SRC = "model.decoder.layers.{i}.encoder_attn.k_proj.weight"


def hf_key_to_whisper(hf_key: str) -> str:
    """Map a HF state_dict key (with 'model.' prefix) to an openai-whisper key."""
    key = hf_key
    if key.startswith("model."):
        key = key[len("model."):]

    # Cross-attention k_proj is emitted once (weight + zero bias) by openai;
    # the HF encoder_attn.k_proj.weight maps to cross_attn.key.weight.
    if key in TOP_LEVEL_MAP:
        return TOP_LEVEL_MAP[key]

    parts = key.split(".")
    if len(parts) >= 4 and parts[1] == "layers":
        scope = parts[0]  # encoder | decoder
        block_idx = parts[2]
        suffix_key = ".".join(parts[3:])  # e.g. "self_attn.k_proj.weight"
        suffix_name = ".".join(suffix_key.split(".")[:-1])  # drop ".weight"/".bias"
        mapped_suffix = BLOCK_SUFFIX_MAP.get(suffix_name, suffix_name)
        return f"{scope}.blocks.{block_idx}.{mapped_suffix}.{suffix_key.split('.')[-1]}"

    raise KeyError(f"Unhandled HF/whisper key: {hf_key}")


def whisper_key_to_hf_src(whisper_key: str) -> str:
    """Inverse: given an openai-whisper key, return the HF source key (with model. prefix).

    Top-level and encoder/decoder weights are one-to-one; cross_attn.key is copied from
    the HF encoder_attn.k_proj. conv1/conv2 keep their identical names. Returns the HF key."""
    if whisper_key in TOP_LEVEL_MAP_INV:
        # e.g. encoder.ln_post.weight -> model.encoder.layer_norm.weight
        return "model." + TOP_LEVEL_MAP_INV[whisper_key]
    if whisper_key.startswith("encoder.conv") or whisper_key.startswith("decoder.conv"):
        # conv1/conv2 weights keep the same name in both frameworks
        return "model." + whisper_key

    parts = whisper_key.split(".")
    if len(parts) >= 4 and parts[1] == "blocks":
        scope = parts[0]
        block_idx = parts[2]
        suffix = ".".join(parts[3:])  # e.g. attn.key.weight
        name = ".".join(suffix.split(".")[:-1])
        rev = {v: k for k, v in BLOCK_SUFFIX_MAP.items()}
        if name == "cross_attn.key":
            # HF has an explicit per-layer encoder_attn.k_proj that matches whisper's
            # cross_attn.key (whisper stores cross key under cross_attn.key).
            return f"model.{scope}.layers.{block_idx}.encoder_attn.k_proj.weight"
        hf_suffix = rev.get(name, name) + "." + suffix.split(".")[-1]
        return f"model.{scope}.layers.{block_idx}.{hf_suffix}"

    raise KeyError(f"Unhandled whisper key: {whisper_key}")


TOP_LEVEL_MAP_INV = {v: k for k, v in TOP_LEVEL_MAP.items()}


def build_state_dict(hf_model, target_model):
    """Load HF weights into an openai-whisper target_model. Mutates target_model."""
    hf_sd = hf_model.state_dict()
    target_keys = list(target_model.state_dict().keys())
    missing = []
    for wkey in target_keys:
        hf_key = whisper_key_to_hf_src(wkey)
        if hf_key not in hf_sd:
            missing.append((wkey, hf_key))
            continue
        target_model.state_dict()[wkey].copy_(hf_sd[hf_key])
    if missing:
        raise RuntimeError(f"Missing HF weights for: {missing}")
    return target_model


@torch.no_grad()
def main():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    parser = argparse.ArgumentParser()
    parser.add_argument("dir_model", type=Path, help="HF checkpoint dir")
    parser.add_argument("dir_out", type=Path, help="output dir")
    parser.add_argument("--quantize", action="store_true", help="also emit int8")
    args = parser.parse_args()

    opset_version = 17
    args.dir_out.mkdir(parents=True, exist_ok=True)

    print("Loading HF model ...")
    hf_model = WhisperForConditionalGeneration.from_pretrained(args.dir_model)
    hf_model.eval()

    # ---- Load a stock openai-whisper small and copy HF weights into it ----
    print("Loading base openai-whisper small ...")
    base = whisper.load_model("small")
    print("Copying HF weights into openai-whisper architecture ...")
    base = build_state_dict(hf_model, base)
    base.eval()

    # The HF model is only needed for its weights; free it before the heavy ONNX
    # exports to keep peak memory within reach of a 8 GB laptop.
    import gc
    del hf_model
    gc.collect()

    model = base
    print(model.dims)
    print(
        f"number of parameters: {sum(p.numel() for p in model.parameters())}"
    )

    # openai-whisper's tiktoken wrapper gives us the token table in the exact
    # sherpa-onnx format (base64 token + id per line).
    tokenizer = whisper.tokenizer.get_tokenizer(
        model.is_multilingual, num_languages=model.num_languages
    )
    print("tokenizer:", type(tokenizer))

    audio = torch.rand(16000 * 2)
    audio = whisper.pad_or_trim(audio)
    n_mels = 80
    mel = (
        whisper.log_mel_spectrogram(audio, n_mels=n_mels)
        .to(model.device)
        .unsqueeze(0)
    )
    batch_size = 1

    encoder = AudioEncoderTensorCache(model.encoder, model.decoder)
    n_layer_cross_k, n_layer_cross_v = encoder(mel)
    print("cross_k shape:", n_layer_cross_k.shape)
    print("cross_v shape:", n_layer_cross_v.shape)

    encoder_filename = args.dir_out / "roman-urdu-fp32-encoder.onnx"
    torch.onnx.export(
        encoder,
        mel,
        encoder_filename,
        opset_version=opset_version,
        input_names=["mel"],
        output_names=["n_layer_cross_k", "n_layer_cross_v"],
        dynamic_axes={
            "mel": {0: "n_audio", 2: "T"},
            "n_layer_cross_k": {1: "n_audio", 2: "T"},
            "n_layer_cross_v": {1: "n_audio", 2: "T"},
        },
        dynamo=False,
    )

    encoder_meta_data = {
        "model_type": "whisper-roman-urdu",
        "version": "1",
        "maintainer": "feminist-mir",
        "n_mels": model.dims.n_mels,
        "n_audio_ctx": model.dims.n_audio_ctx,
        "n_audio_state": model.dims.n_audio_state,
        "n_audio_head": model.dims.n_audio_head,
        "n_audio_layer": model.dims.n_audio_layer,
        "n_vocab": model.dims.n_vocab,
        "n_text_ctx": model.dims.n_text_ctx,
        "n_text_state": model.dims.n_text_state,
        "n_text_head": model.dims.n_text_head,
        "n_text_layer": model.dims.n_text_layer,
        "sot_sequence": ",".join(list(map(str, tokenizer.sot_sequence))),
        "all_language_tokens": ",".join(list(map(str, tokenizer.all_language_tokens))),
        "all_language_codes": ",".join(tokenizer.all_language_codes),
        "sot": tokenizer.sot,
        "sot_index": tokenizer.sot_sequence.index(tokenizer.sot),
        "eot": tokenizer.eot,
        "blank_id": tokenizer.encode(" ")[0],
        "is_multilingual": int(model.is_multilingual),
        "no_speech": tokenizer.no_speech,
        "non_speech_tokens": ",".join(list(map(str, tokenizer.non_speech_tokens))),
        "transcribe": tokenizer.transcribe,
        "translate": tokenizer.translate,
        "sot_prev": tokenizer.sot_prev,
        "sot_lm": tokenizer.sot_lm,
        "no_timestamps": tokenizer.no_timestamps,
    }
    print("encoder_meta_data:", encoder_meta_data)
    add_meta_data(str(encoder_filename), encoder_meta_data)

    n_audio = mel.shape[0]
    tokens = torch.tensor(
        [[tokenizer.sot, tokenizer.sot, tokenizer.sot]] * n_audio
    ).to(model.device)
    decoder = TextDecoderTensorCache(model.decoder, model.dims.n_text_ctx)
    n_layer_self_k_cache = torch.zeros(
        (
            len(model.decoder.blocks),
            n_audio,
            model.dims.n_text_ctx,
            model.dims.n_text_state,
        ),
        device=mel.device,
    )
    n_layer_self_v_cache = torch.zeros(
        (
            len(model.decoder.blocks),
            n_audio,
            model.dims.n_text_ctx,
            model.dims.n_text_state,
        ),
        device=mel.device,
    )
    offset = torch.zeros(1, dtype=torch.int64).to(mel.device)
    logits, _, _ = decoder(
        tokens,
        n_layer_self_k_cache,
        n_layer_self_v_cache,
        n_layer_cross_k,
        n_layer_cross_v,
        offset,
    )
    assert logits.shape == (n_audio, tokens.shape[1], model.dims.n_vocab), logits.shape
    print("logits shape:", logits.shape)

    offset = torch.tensor([tokens.shape[1]], dtype=torch.int64).to(mel.device)
    tokens = torch.tensor([[tokenizer.sot]] * n_audio).to(mel.device)

    decoder_filename = args.dir_out / "roman-urdu-fp32-decoder.onnx"
    torch.onnx.export(
        decoder,
        (
            tokens,
            n_layer_self_k_cache,
            n_layer_self_v_cache,
            n_layer_cross_k,
            n_layer_cross_v,
            offset,
        ),
        decoder_filename,
        opset_version=opset_version,
        input_names=[
            "tokens",
            "in_n_layer_self_k_cache",
            "in_n_layer_self_v_cache",
            "n_layer_cross_k",
            "n_layer_cross_v",
            "offset",
        ],
        output_names=["logits", "out_n_layer_self_k_cache", "out_n_layer_self_v_cache"],
        dynamic_axes={
            "tokens": {0: "n_audio", 1: "n_tokens"},
            "in_n_layer_self_k_cache": {1: "n_audio"},
            "in_n_layer_self_v_cache": {1: "n_audio"},
            "n_layer_cross_k": {1: "n_audio", 2: "T"},
            "n_layer_cross_v": {1: "n_audio", 2: "T"},
        },
        dynamo=False,
    )
    print("Saved:", encoder_filename)
    print("Saved:", decoder_filename)

    # Write tokens.txt using the openai tiktoken asset (same vocab as HF).
    convert_tokens(whisper.__file__, model, args.dir_out / "roman-urdu-fp32-tokens.txt")
    print("Saved:", args.dir_out / "roman-urdu-fp32-tokens.txt")

    if args.quantize:
        from onnxruntime.quantization import QuantType, quantize_dynamic

        print("Generating int8 quantization models ...")
        enc_int8 = args.dir_out / "roman-urdu-int8-encoder.onnx"
        dec_int8 = args.dir_out / "roman-urdu-int8-decoder.onnx"
        quantize_dynamic(
            model_input=str(encoder_filename),
            model_output=str(enc_int8),
            op_types_to_quantize=["MatMul"],
            weight_type=QuantType.QInt8,
        )
        quantize_dynamic(
            model_input=str(decoder_filename),
            model_output=str(dec_int8),
            op_types_to_quantize=["MatMul"],
            weight_type=QuantType.QInt8,
        )
        print("Saved:", enc_int8)
        print("Saved:", dec_int8)


# --- The following are copied/adapted from sherpa-onnx scripts/whisper/export-onnx.py ---

def add_meta_data(filename, meta_data):
    """Add metadata to an ONNX model in place, saving a self-contained
    single-file ONNX (no external .weights/.data), matching the form
    sherpa-onnx publishes for non-large Whisper models."""
    import onnx

    model = onnx.load(filename)
    while len(model.metadata_props):
        model.metadata_props.pop()
    for key, value in meta_data.items():
        meta = model.metadata_props.add()
        meta.key = key
        meta.value = str(value)
    try:
        onnx.save(model, filename, save_as_external_data=False)
    except TypeError:
        # older onnx: save_as_external_data is the default False already
        onnx.save(model, filename)


def modified_audio_encoder_forward(self: AudioEncoder, x: torch.Tensor):
    """AudioEncoder.forward, but supports arbitrary-length audio (T <= 30s)."""
    x = F.gelu(self.conv1(x))
    x = F.gelu(self.conv2(x))
    x = x.permute(0, 2, 1)

    assert (
        x.shape[2] == self.positional_embedding.shape[1]
    ), f"incorrect audio shape: {x.shape}, {self.positional_embedding.shape}"
    assert (
        x.shape[1] == self.positional_embedding.shape[0]
    ), f"incorrect audio shape: {x.shape}, {self.positional_embedding.shape}"
    x = (x + self.positional_embedding[: x.shape[1]]).to(x.dtype)

    for block in self.blocks:
        x = block(x)
    x = self.ln_post(x)
    return x


AudioEncoder.forward = modified_audio_encoder_forward


class AudioEncoderTensorCache(nn.Module):
    def __init__(self, inAudioEncoder: AudioEncoder, inTextDecoder: TextDecoder):
        super().__init__()
        self.audioEncoder = inAudioEncoder
        self.textDecoder = inTextDecoder

    def forward(self, x: Tensor):
        audio_features = self.audioEncoder(x)
        n_layer_cross_k_list = []
        n_layer_cross_v_list = []
        for block in self.textDecoder.blocks:
            n_layer_cross_k_list.append(block.cross_attn.key(audio_features))
            n_layer_cross_v_list.append(block.cross_attn.value(audio_features))
        return torch.stack(n_layer_cross_k_list), torch.stack(n_layer_cross_v_list)


class MultiHeadAttentionCross(nn.Module):
    def __init__(self, inMultiHeadAttention: MultiHeadAttention):
        super().__init__()
        self.multiHeadAttention = inMultiHeadAttention

    def forward(self, x: Tensor, k: Tensor, v: Tensor, mask=None):
        q = self.multiHeadAttention.query(x)
        wv, qk = self.multiHeadAttention.qkv_attention(q, k, v, mask)
        return self.multiHeadAttention.out(wv)


class MultiHeadAttentionSelf(nn.Module):
    def __init__(self, inMultiHeadAttention: MultiHeadAttention):
        super().__init__()
        self.multiHeadAttention = inMultiHeadAttention

    def forward(self, x, k_cache, v_cache, mask):
        q = self.multiHeadAttention.query(x)
        k = self.multiHeadAttention.key(x)
        v = self.multiHeadAttention.value(x)
        k_cache[:, -k.shape[1]:, :] = k
        v_cache[:, -v.shape[1]:, :] = v
        wv, qk = self.multiHeadAttention.qkv_attention(q, k_cache, v_cache, mask)
        return self.multiHeadAttention.out(wv), k_cache, v_cache


class ResidualAttentionBlockTensorCache(nn.Module):
    def __init__(self, inResidualAttentionBlock: ResidualAttentionBlock):
        super().__init__()
        self.originalBlock = inResidualAttentionBlock
        self.attn = MultiHeadAttentionSelf(inResidualAttentionBlock.attn)
        self.cross_attn = (
            MultiHeadAttentionCross(inResidualAttentionBlock.cross_attn)
            if inResidualAttentionBlock.cross_attn
            else None
        )

    def forward(self, x, self_k_cache, self_v_cache, cross_k, cross_v, mask):
        self_attn_x, self_k_cache_updated, self_v_cache_updated = self.attn(
            self.originalBlock.attn_ln(x), self_k_cache, self_v_cache, mask=mask
        )
        x = x + self_attn_x
        if self.cross_attn:
            x = x + self.cross_attn(
                self.originalBlock.cross_attn_ln(x), cross_k, cross_v
            )
        x = x + self.originalBlock.mlp(self.originalBlock.mlp_ln(x))
        return x, self_k_cache_updated, self_v_cache_updated


class TextDecoderTensorCache(nn.Module):
    def __init__(self, inTextDecoder: TextDecoder, in_n_ctx: int):
        super().__init__()
        self.textDecoder = inTextDecoder
        self.n_ctx = in_n_ctx
        self.blocks = []
        for original_block in self.textDecoder.blocks:
            self.blocks.append(ResidualAttentionBlockTensorCache(original_block))

    def forward(self, tokens, n_layer_self_k_cache, n_layer_self_v_cache,
                n_layer_cross_k, n_layer_cross_v, offset):
        x = (
            self.textDecoder.token_embedding(tokens)
            + self.textDecoder.positional_embedding[
                offset[0]: offset[0] + tokens.shape[-1]
            ]
        )
        x = x.to(n_layer_cross_k[0].dtype)

        i = 0
        for block in self.blocks:
            self_k_cache = n_layer_self_k_cache[i, :, : offset[0] + tokens.shape[-1], :]
            self_v_cache = n_layer_self_v_cache[i, :, : offset[0] + tokens.shape[-1], :]
            x, self_k_cache, self_v_cache = block(
                x,
                self_k_cache=self_k_cache,
                self_v_cache=self_v_cache,
                cross_k=n_layer_cross_k[i],
                cross_v=n_layer_cross_v[i],
                mask=self.textDecoder.mask,
            )
            n_layer_self_k_cache[i, :, : offset[0] + tokens.shape[-1], :] = self_k_cache
            n_layer_self_v_cache[i, :, : offset[0] + tokens.shape[-1], :] = self_v_cache
            i += 1

        x = self.textDecoder.ln(x)
        logits = (
            torch.matmul(
                self.textDecoder.token_embedding.weight.to(x.dtype),
                x.permute(0, 2, 1),
            )
            .permute(0, 2, 1)
            .float()
        )
        return logits, n_layer_self_k_cache, n_layer_self_v_cache


def convert_tokens(whisper_file, model, out_path):
    """Write the token table (base64 token + rank per line) for sherpa-onnx."""
    whisper_dir = Path(whisper_file).parent
    multilingual = model.is_multilingual
    tokenizer_file = (
        whisper_dir
        / "assets"
        / (multilingual and "multilingual.tiktoken" or "gpt2.tiktoken")
    )
    if not tokenizer_file.is_file():
        raise ValueError(f"Cannot find {tokenizer_file}")

    with open(tokenizer_file, "r") as f:
        tokens = {
            token: int(rank)
            for token, rank in (line.split() for line in f if line)
        }

    with open(out_path, "w") as f:
        for t, i in tokens.items():
            f.write(f"{t} {i}\n")


if __name__ == "__main__":
    with disable_sdpa():
        main()
