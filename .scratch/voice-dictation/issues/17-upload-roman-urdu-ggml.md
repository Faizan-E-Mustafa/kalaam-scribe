# 17: Upload Roman-Urdu GGML models to HuggingFace

**What to build:** Publish the two converted Roman-Urdu GGML files
(`ggml-model-f16.bin` and `ggml-model-q4_0.bin`) to a public HuggingFace repo so
the Android app can download them at runtime instead of requiring a manual copy.

**Blocked by:** 16 — Convert Roman-Urdu to GGML (f16 + q4_0)

**Status:** done

## Verified

Repo: https://huggingface.co/femustafa/voicedictation-models (public)

- [x] [human] Generate HF Write access token (Settings → Access Tokens)
- [x] Repo created public (actually via `hf repos create`; wizard also supports it)
- [x] `ggml-model-q4_0.bin` uploaded and downloadable (HTTP 200, 145,458,032 bytes)
- [x] `ggml-model-f16.bin` uploaded and downloadable (HTTP 200, 487,601,984 bytes)
- [x] File names match exactly what the app catalog references
- [x] `README.md` added (model card: source, files, `-l auto` usage)

## Approach notes

The repo must be **public**: the Android app's plain `HttpURLConnection`
downloader sends no auth headers, so a private repo would break the runtime
download (ticket 18). The model derives from the MIT-licensed public fine-tune,
so hosting it carries no proprietary risk.

Upload was driven by the human via the `hf` CLI (`hf repos create` + `hf upload`
after `hf auth login`); `tools/roman-urdu/hf-upload.sh` is an alternative wizard
that captures an HF Write token into `.env`, creates the public repo, and uploads
via `huggingface_hub`.
