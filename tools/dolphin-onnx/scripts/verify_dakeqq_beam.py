"""Laptop verification spike (ticket 28): DakeQQ matched ONNX encoder+decoder
under attention beam search.

DakeQQ's exported pair (from onnx-community/dataocean-dolphin-asr) is what we
will deploy on the phone. This script confirms that a plain attention beam search
over the decoder's FULL output logits produces correct Urdu when the language pin
(ur/PK) is set via language_start/language_end.

Requires
  - the DakeQQ encoder + decoder .onnx under DAKEQQ_DIR
  - decoder must expose full/sliced logits: run add_logits_output.py on the stock
    decoder to produce decoder_withlogits.onnx
  - Dolphin vocab: ~/.cache/dolphin/base/units.txt + bpe.model

Key findings it reproduces
  * Feeding the FULL logits (`/output_layer/Gemm_output_0`) -> correct Urdu.
  * Feeding the LANG-SLICED logits (`/Slice_3_output_0`, restricted to
    [136:269]) -> garbage language tokens (<tk><ml><mrj>...). Content must use
    the full logits.

Usage
  python verify_dakeqq_beam.py clip.wav [--dakeqq-dir DIR] [--decoder NAME] [--beam N]
"""
import argparse
import os
import sys

import numpy as np
import onnxruntime as ort
import sentencepiece as spm

MODEL_DIR = os.path.expanduser("~/.cache/dolphin/base")
UNITS = os.path.join(MODEL_DIR, "units.txt")
BPE = os.path.join(MODEL_DIR, "bpe.model")

SOS, EOS, NOTS = 39999, 40000, 324
ASR = 6  # <asr> task token
LANG_UR, REGION_PK = 136, 269

NL = 6  # number of decoder layers (base: 6)


def build_inputs(prefix_tokens, history_len, en_keys, en_values, de_keys, de_values,
                 lang_start, lang_end, attention_mask):
    inputs = {}
    for i in range(NL):
        inputs[f"in_de_key_{i}"] = de_keys[i]
        inputs[f"in_de_value_{i}"] = de_values[i]
        inputs[f"en_key_{i}"] = en_keys[i]
        inputs[f"en_value_{i}"] = en_values[i]
    inputs["input_ids"] = np.array([prefix_tokens], dtype=np.int32)
    inputs["history_len"] = np.array([history_len], dtype=np.int64)
    inputs["ids_len"] = np.array([len(prefix_tokens)], dtype=np.int64)
    inputs["language_start"] = np.array([lang_start], dtype=np.int64)
    inputs["language_end"] = np.array([lang_end], dtype=np.int64)
    inputs["attention_mask"] = np.array([attention_mask], dtype=np.int8)
    return inputs


def run_full(dec, prefix_tokens, history_len, en_keys, en_values, de_keys, de_values,
             lang_start, lang_end, attention_mask=0):
    inputs = build_inputs(prefix_tokens, history_len, en_keys, en_values,
                          de_keys, de_values, lang_start, lang_end, attention_mask)
    out = dec.run(None, inputs)
    names = [o.name for o in dec.get_outputs()]
    d = dict(zip(names, out))
    de_keys = [d[f"out_de_key_{i}"] for i in range(NL)]
    de_values = [d[f"out_de_value_{i}"] for i in range(NL)]
    return {
        "de_keys": de_keys,
        "de_values": de_values,
        "argmax": d["max_logit_id"].flat[0],
        "full_logits": d.get("/output_layer/Gemm_output_0"),
        "sliced_logits": d.get("/Slice_3_output_0"),
    }


def get_logprobs(r, use_sliced):
    logits = r["sliced_logits"] if use_sliced else r["full_logits"]
    if logits is None:
        raise RuntimeError("logits output missing — did you run add_logits_output.py?")
    lprobs = logits[0, -1, :] if logits.ndim == 3 else logits[0]
    lprobs = lprobs.astype(np.float64)
    m = lprobs.max()
    return lprobs - (m + np.log(np.sum(np.exp(lprobs - m))))


def beam_search(enc, dec, audio_i16, lang_start, lang_end, token_list,
                beam_size=4, max_len=60, use_sliced=False):
    kv = enc.run(None, {"audio": audio_i16.reshape(1, 1, -1)})
    en_keys, en_values = kv[:NL], kv[NL:]

    prefix = [SOS, lang_start, lang_end, ASR, NOTS]

    def decode_step(tokens, history_len, dk, dv):
        return run_full(dec, [tokens[-1]], history_len, en_keys, en_values, dk, dv,
                        lang_start, lang_end)

    de_keys = [np.zeros((8, 64, 0), dtype=np.float32) for _ in range(NL)]
    de_values = [np.zeros((8, 0, 64), dtype=np.float32) for _ in range(NL)]
    r = run_full(dec, prefix, 0, en_keys, en_values, de_keys, de_values,
                 lang_start, lang_end)
    plogits = get_logprobs(r, use_sliced)

    beams = []
    for idx in np.argsort(-plogits)[:beam_size]:
        beams.append((list(prefix) + [int(idx)], float(plogits[int(idx)]),
                      r["de_keys"], r["de_values"]))

    for step in range(max_len):
        all_cands = []
        for tokens, score, dk, dv in beams:
            if tokens[-1] == EOS:
                all_cands.append((tokens, score, dk, dv))
                continue
            r = decode_step(tokens, len(tokens) - 1, dk, dv)
            logits = get_logprobs(r, use_sliced)
            for idx in np.argsort(-logits)[:beam_size]:
                all_cands.append((tokens + [int(idx)], score + float(logits[int(idx)]),
                                  r["de_keys"], r["de_values"]))
        all_cands.sort(key=lambda x: -x[1])
        beams = all_cands[:beam_size]
        if all(t[-1] == EOS for t, _, _, _ in beams):
            break
    return beams[0][0]


def decode_text(tokens, sp, token_list):
    content = [t for t in tokens[5:] if t not in (0, SOS, EOS, NOTS)]
    syms = [token_list[t] for t in content if t < len(token_list)]
    return sp.DecodePieces(syms), content


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wav")
    ap.add_argument("--dakeqq-dir", default=os.path.expanduser(
        "~/.cache/dolphin/dakeqq/dolphin-base"))
    ap.add_argument("--encoder", default="dolphin-base-encoder.onnx")
    ap.add_argument("--decoder", default="dolphin-base-decoder_withlogits.onnx")
    ap.add_argument("--beam", type=int, default=4)
    args = ap.parse_args()

    token_list = [ln.strip().split(" ")[0] for ln in open(UNITS)]
    sp = spm.SentencePieceProcessor()
    sp.load(BPE)

    enc = ort.InferenceSession(
        os.path.join(args.dakeqq_dir, args.encoder), providers=["CPUExecutionProvider"])
    dec = ort.InferenceSession(
        os.path.join(args.dakeqq_dir, args.decoder), providers=["CPUExecutionProvider"])
    print("Decoder inputs :", [i.name for i in dec.get_inputs()])
    print("Decoder outputs:", [o.name for o in dec.get_outputs()])

    import soundfile as sf
    audio_f, sr = sf.read(args.wav, dtype="float32")
    if audio_f.ndim > 1:
        audio_f = audio_f[:, 0]
    audio_i16 = (audio_f * 32768).astype(np.int16)

    print(f"\n=== {os.path.basename(args.wav)} ===")
    for use_sliced in (False, True):
        label = "FULL-logits" if not use_sliced else "LANG-SLICED"
        print(f"\n--- beam search with {label} ---")
        tokens = beam_search(enc, dec, audio_i16, LANG_UR, REGION_PK, token_list,
                             beam_size=args.beam, use_sliced=use_sliced)
        text, content = decode_text(tokens, sp, token_list)
        print(f"  RESULT [{label}]: {text}")
        print(f"  content tokens: {content}")


if __name__ == "__main__":
    main()
