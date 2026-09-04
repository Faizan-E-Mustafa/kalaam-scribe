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


def build_inputs(prefix_tokens, history_len, en_keys, en_values, de_keys, de_values,
                 lang_start, lang_end, attention_mask, nl):
    inputs = {}
    for i in range(nl):
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
             lang_start, lang_end, nl, attention_mask=0):
    inputs = build_inputs(prefix_tokens, history_len, en_keys, en_values,
                          de_keys, de_values, lang_start, lang_end, attention_mask, nl)
    out = dec.run(None, inputs)
    names = [o.name for o in dec.get_outputs()]
    d = dict(zip(names, out))
    de_keys = [d[f"out_de_key_{i}"] for i in range(nl)]
    de_values = [d[f"out_de_value_{i}"] for i in range(nl)]
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


def beam_search(enc, dec, audio_i16, lang_start, lang_end, token_list, nl,
                beam_size=4, max_len=60, use_sliced=False):
    kv = enc.run(None, {"audio": audio_i16.reshape(1, 1, -1)})
    en_keys, en_values = kv[:nl], kv[nl:]
    head_dim = en_keys[0].shape[0]
    d_model = en_keys[0].shape[1]
    # fp16 model tiers (e.g. fp16/arm) keep their KV cache in float16; the dtype
    # of the encoder cross-KV output is the source of truth for the cache type.
    kv_dtype = en_keys[0].dtype

    prefix = [SOS, lang_start, lang_end, ASR, NOTS]

    def decode_step(tokens, history_len, dk, dv):
        return run_full(dec, [tokens[-1]], history_len, en_keys, en_values, dk, dv,
                        lang_start, lang_end, nl)

    de_keys = [np.zeros((head_dim, d_model, 0), dtype=kv_dtype) for _ in range(nl)]
    de_values = [np.zeros((head_dim, 0, d_model), dtype=kv_dtype) for _ in range(nl)]
    r = run_full(dec, prefix, 0, en_keys, en_values, de_keys, de_values,
                 lang_start, lang_end, nl)
    plogits = get_logprobs(r, use_sliced)

    # Seed the active beam list with the top-k continuations of the prefix.
    beams = []
    for idx in np.argsort(-plogits)[:beam_size]:
        beams.append((list(prefix) + [int(idx)], float(plogits[int(idx)]),
                      r["de_keys"], r["de_values"]))

    # Completed (EOS-terminated) hypotheses live on a separate list so a short
    # EOS beam cannot out-score a longer, still-in-progress one purely because it
    # has fewer (negative) logprob terms to sum. Without this separation the
    # small model collapses to an empty hypothesis (see ticket 29 finding).
    finished = []

    for step in range(max_len):
        active = []
        for tokens, score, dk, dv in beams:
            if tokens[-1] == EOS:
                finished.append((tokens, score))
                continue
            active.append((tokens, score, dk, dv))

        if not active:
            break

        all_cands = []
        for tokens, score, dk, dv in active:
            r = decode_step(tokens, len(tokens) - 1, dk, dv)
            logits = get_logprobs(r, use_sliced)
            for idx in np.argsort(-logits)[:beam_size]:
                all_cands.append((tokens + [int(idx)], score + float(logits[int(idx)]),
                                  r["de_keys"], r["de_values"]))
        all_cands.sort(key=lambda x: -x[1])
        beams = all_cands[:beam_size]

    if not finished and beams:
        finished = [(t, s) for t, s, _, _ in beams]

    # DakeQQ's small decoder gives <eos> an abnormally high probability right
    # after the prefix, so both the empty prefix-only hypothesis AND short
    # one-token-then-eos hypotheses out-score longer real transcripts under raw
    # cumulative-logprob comparison. Two fixes (see ticket 29 finding):
    #   1. Drop prefix-only (content-less) hypotheses — an empty transcript is
    #      never a valid dictation output.
    #   2. Select by LENGTH-NORMALIZED score (cumulative / #generated content
    #      tokens), the standard ASR beam-search scoring, so the full sentence
    #      isn't penalised for being longer than an early-eos truncation.
    pref_len = len(prefix)
    meaningful = [(t, s) for t, s in finished if len(t) > pref_len]
    if not meaningful:
        meaningful = [(t, s) for t, s in beams if len(t) > pref_len]

    if meaningful:
        # Normalize by number of generated content tokens; prefix tokens don't
        # count toward the transcript length.
        def norm_score(item):
            tokens, score = item
            gen = max(1, len(tokens) - pref_len)
            return score / gen
        return max(meaningful, key=norm_score)[0]
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

    # Number of model layers = number of `en_key_*` encoder outputs / decoder
    # `in_de_key_*` inputs (base: 6, small: 12). Derived, not hardcoded.
    nl = sum(1 for o in enc.get_outputs() if o.name.startswith("en_key_"))

    import soundfile as sf
    audio_f, sr = sf.read(args.wav, dtype="float32")
    if audio_f.ndim > 1:
        audio_f = audio_f[:, 0]
    audio_i16 = (audio_f * 32768).astype(np.int16)

    print(f"\n=== {os.path.basename(args.wav)} ===")
    for use_sliced in (False, True):
        label = "FULL-logits" if not use_sliced else "LANG-SLICED"
        print(f"\n--- beam search with {label} ---")
        tokens = beam_search(enc, dec, audio_i16, LANG_UR, REGION_PK, token_list, nl,
                             beam_size=args.beam, use_sliced=use_sliced)
        text, content = decode_text(tokens, sp, token_list)
        print(f"  RESULT [{label}]: {text}")
        print(f"  content tokens: {content}")


if __name__ == "__main__":
    main()
