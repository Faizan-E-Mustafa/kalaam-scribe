"""Standalone Dolphin inference: torch encoder + exported decoder.onnx + beam search.

This is the WORKING laptop pipeline established in ticket 28. The encoder stays
in torch (it could not be ONNX-exported — see README.md); only the decoder runs
through onnxruntime. Decoding is plain `attention` beam search, which is the
method that respects a pinned language/region (attention_rescoring and greedy
override ur/PK and flip output to Devanagari/garbage).

CRITICAL (double-CMVN bug): Dolphin's encoder applies global_cmvn internally
(model.py:1743). Feed RAW log-mel features (no manual mean/std normalization).
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

DEFAULT_MODEL_CACHE = os.path.expanduser("~/.cache/dolphin")
REPO_SAMPLE = str(
    Path(__file__).resolve().parents[3]
    / "tools/roman-urdu/samples/rec.wav"
)

SOS = 39999
EOS = 40000
NOTIMESTAMP = 324  # <notimestamp>

LANG_IDS = {
    "ur": 136, "hi": 50, "bn": 18, "en": 13, "zh": 39, "ar": 5,
    "ja": 77, "ko": 92, "ta": 166, "te": 172, "ml": 111,
    "mr": 113, "gu": 56, "kn": 86, "pa": 124, "or": 120,
}
REGION_IDS = {
    "IN": 69, "PK": 269, "CN": 26, "JP": 76, "KR": 91,
    "BD": 16, "LK": 320, "NP": 259, "MM": 254, "TH": 335,
    "SA": 300, "AE": 348, "EG": 52, "NG": 261, "ZA": 363,
    "IR": 179, "TR": 344,
}


def load_model_sessions(decoder_onnx, model_size="base", model_cache=DEFAULT_MODEL_CACHE,
                        torch_threads=2):
    """Return (torch model (for encoder), onnxruntime decoder session)."""
    os.environ.setdefault("AUDIO_BACKEND", "soundfile")
    os.environ.setdefault("MODELSCOPE_CACHE", os.path.expanduser("~/.cache/modelscope"))
    torch.set_num_threads(torch_threads)
    import dolphin
    model = dolphin.load_model(model_size, model_dir=os.path.join(model_cache, model_size),
                               device="cpu")
    model.eval()

    import onnxruntime as ort
    opts = ort.SessionOptions()
    opts.inter_op_num_threads = torch_threads
    opts.intra_op_num_threads = torch_threads
    dec = ort.InferenceSession(decoder_onnx, opts, providers=["CPUExecutionProvider"])
    return model, dec


def load_tokenizer(model_dir):
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor()
    sp.load(os.path.join(model_dir, "bpe.model"))
    with open(os.path.join(model_dir, "units.txt")) as f:
        token_list = [ln.strip().split(" ")[0] for ln in f]
    return sp, token_list


def extract_features(wav_1d, sr=16000):
    """Raw log-mel features (NO CMVN) -> (feats [1,T,80], feats_len [1])."""
    import torchaudio.functional as F_mel
    wav_t = torch.from_numpy(wav_1d).float().unsqueeze(0)
    n_fft, hop, win = 512, 160, 400
    window = torch.hann_window(win)
    spec = torch.stft(wav_t, n_fft=n_fft, hop_length=hop, win_length=win,
                      window=window, return_complex=True)
    power = spec.real ** 2 + spec.imag ** 2
    mel_fb = F_mel.melscale_fbanks(n_freqs=n_fft // 2 + 1, n_mels=80,
                                   sample_rate=sr, f_min=0.0, f_max=sr / 2.0)
    feats = torch.matmul(mel_fb.T, power.squeeze(0))
    feats = torch.clamp(feats, min=1e-10).log().T.unsqueeze(0)
    feats_len = torch.tensor([feats.shape[1]], dtype=torch.long)
    return feats, feats_len


def run_encoder(model, feats, feats_len):
    with torch.no_grad():
        enc_out, enc_mask = model.encoder(feats, feats_len)
    return enc_out.numpy().astype(np.float32), enc_mask.squeeze(1).numpy()


def predict_prefix(dec_sess, enc_out, enc_mask, base_prefix):
    """Predict <task> token then append <notimestamp>. Returns full prefix."""
    tgt = np.array([base_prefix], dtype=np.int64)
    logits = dec_sess.run(None, {
        "memory": enc_out,
        "memory_mask": enc_mask.astype(bool),
        "tgt": tgt,
    })[0]
    task_id = int(np.argmax(logits[0, -1, :]))
    return base_prefix + [task_id, NOTIMESTAMP]


def beam_search(dec_sess, enc_out, enc_mask, prefix, beam_size=4, max_len=100):
    """Single-pass attention beam search. Returns content tokens after prefix."""
    beams = [(list(prefix), 0.0)]
    for _ in range(max_len):
        all_cands = []
        for tokens, score in beams:
            if tokens[-1] == EOS:
                all_cands.append((tokens, score))
                continue
            tgt = np.array([tokens], dtype=np.int64)
            logits = dec_sess.run(None, {
                "memory": enc_out,
                "memory_mask": enc_mask.astype(bool),
                "tgt": tgt,
            })[0]
            logprobs = logits[0, -1, :]
            topk = np.argsort(-logprobs)[:beam_size]
            for idx in topk:
                all_cands.append((tokens + [int(idx)], score + float(logprobs[idx])))
        all_cands.sort(key=lambda x: -x[1])
        beams = all_cands[:beam_size]
        if all(t[-1] == EOS for t, _ in beams):
            break
    best = beams[0][0][len(prefix):]
    if best and best[-1] == EOS:
        best = best[:-1]
    return best


def detokenize(token_ids, sp, token_list):
    content = [t for t in token_ids if t not in (SOS, EOS, 0, NOTIMESTAMP)]
    symbols = [token_list[t] for t in content if t < len(token_list)]
    return sp.DecodePieces(symbols)


def transcribe(wav_path, decoder_onnx, model_size="base", model_cache=DEFAULT_MODEL_CACHE,
               lang=None, region=None, beam_size=4):
    audio, sr = sf.read(wav_path, dtype="float32")
    if audio.ndim > 1:
        audio = audio[:, 0]

    model, dec = load_model_sessions(decoder_onnx, model_size, model_cache)
    sp, token_list = load_tokenizer(os.path.join(model_cache, model_size))

    feats, feats_len = extract_features(audio, sr)
    enc_out, enc_mask = run_encoder(model, feats, feats_len)

    prefix = [SOS]
    if lang and region:
        lang_id, region_id = LANG_IDS.get(lang), REGION_IDS.get(region)
        if lang_id is not None and region_id is not None:
            prefix += [lang_id, region_id]
    print(f"Base prefix: {prefix}")

    prefix = predict_prefix(dec, enc_out, enc_mask, prefix)
    print(f"Full prefix: {prefix}")

    content = beam_search(dec, enc_out, enc_mask, prefix, beam_size=beam_size)
    print(f"Content tokens: {content}")

    return detokenize(content, sp, token_list)


def main():
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser()
    ap.add_argument("wav", nargs="?", default=REPO_SAMPLE)
    ap.add_argument("--model", default="base", choices=["base", "small"])
    ap.add_argument("--lang", default="ur")
    ap.add_argument("--region", default="PK")
    ap.add_argument("--beam", type=int, default=4)
    ap.add_argument("--decoder", default=str(here.parent / "artifacts/decoder.onnx"),
                    help="path to exported decoder.onnx")
    ap.add_argument("--model-cache", default=DEFAULT_MODEL_CACHE)
    args = ap.parse_args()

    result = transcribe(
        args.wav, args.decoder, args.model, args.model_cache,
        lang=args.lang, region=args.region, beam_size=args.beam,
    )
    print(f"\nResult: {result}")


if __name__ == "__main__":
    main()
