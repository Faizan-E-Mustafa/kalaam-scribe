# 17: Upload Roman-Urdu GGML models to HuggingFace

**What to build:** Publish the two converted Roman-Urdu GGML files
(`ggml-model-f16.bin` and `ggml-model-q4_0.bin`) to a public HuggingFace repo so
the Android app can download them at runtime instead of requiring a manual copy.

**Blocked by:** 16 — Convert Roman-Urdu to GGML (f16 + q4_0)

**Status:** ready-for-agent (needs human for the HF token step)

## Approach

The repo must be **public**: the Android app's plain `HttpURLConnection`
downloader sends no auth headers, so a private repo would break the runtime
download (ticket 18). The model derives from the MIT-licensed public fine-tune,
so hosting it carries no proprietary risk.

Upload is driven by `tools/roman-urdu/hf-upload.sh`, a wizard that:
1. Captures the user's HF **Write** access token (`HF_TOKEN` in `.env`).
2. Creates the public repo `<user>/voicedictation-models` (idempotent; resolves
   the real username from the token, not a hardcoded one).
3. Uploads both `.bin` files as Git LFS (auto-detected for `.bin`).

The human's only manual step is generating the HF token; everything else is
automated with `huggingface_hub` (already in the `validate_stt` venv).

- [ ] [human] Generate HF Write access token (Settings → Access Tokens)
- [ ] `tools/roman-urdu/hf-upload.sh` creates the public repo
- [ ] `ggml-model-q4_0.bin` uploaded and downloadable (HTTP 200)
- [ ] `ggml-model-f16.bin` uploaded and downloadable (HTTP 200)
- [ ] File names match exactly what the app catalog references
