import argparse
import logging

logging.disable(logging.INFO)

import soundfile as sf
import torch
import torchaudio
import numpy as np


def _load_wav(path):
    w, sr = sf.read(path, dtype="float32")
    if w.ndim == 1:
        w = w[None, :]
    else:
        w = w.T
    return torch.from_numpy(w.copy()), sr


torchaudio.load = _load_wav
import dolphin


def run(model, audio, lang, region, method, beam, label):
    print(f"--- {label} ---")
    r = dolphin.transcribe(
        model, audio,
        lang_sym=lang, region_sym=region,
        predict_time=False, word_timestamp=False,
        decoding_method=method, beam_size=beam,
    )
    print(f"language    : {r.language}")
    print(f"region      : {r.region}")
    print(f"text        : {r.text}")
    print(f"text_nospecial: {r.text_nospecial}")
    print()
    return r


def main():
    p = argparse.ArgumentParser()
    p.add_argument("audio", type=str)
    p.add_argument("--model", default="base", choices=["base", "small"])
    p.add_argument("--lang", default="ur")
    p.add_argument("--region", default="PK")
    p.add_argument("--beam", type=int, default=10)
    p.add_argument("--method", default="attention_rescoring",
                   choices=["attention", "attention_rescoring"])
    args = p.parse_args()

    w, sr = torchaudio.load(args.audio)
    dur = w.shape[1] / sr
    rms = float(np.sqrt((w.numpy() ** 2).mean()))
    peak = float(w.abs().max())
    print(f"audio  : {args.audio}")
    print(f"dur    : {dur:.2f}s  sr={sr}  rms={rms:.4f}  peak={peak:.4f}")
    if rms < 0.003:
        print("!! WARNING: audio looks near-silent; check the recording.")
    if sr != 16000:
        print(f"!! WARNING: expected 16kHz, got {sr}")
    print("=" * 60)

    model = dolphin.load_model(args.model, device="cpu")

    run(model, args.audio, None, None, args.method, args.beam,
        "AUTO-DETECT (reproduces sherpa #3904)")
    run(model, args.audio, args.lang, args.region, args.method, args.beam,
        f"PINNED lang={args.lang} region={args.region}")


if __name__ == "__main__":
    main()
