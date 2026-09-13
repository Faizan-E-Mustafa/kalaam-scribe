# Deploy Roman-Urdu model to the phone

Validated dev-machine artifacts are staged on Windows at:

    C:\Users\femustafa\Downloads\dictate-deploy\
        ggml-model-q4_0.bin   (139 MB, small q4_0 Roman-Urdu model)
        dictate.sh            (updated: MODEL + -l auto)

## 1. Get the files onto the phone

Two files go onto the phone. Use any route you like (file manager, cloud
upload + phone download, email, etc.). The model is 139 MB.

Target locations on the phone:

- Model:   `~/whisper.cpp/models/ggml-model-q4_0.bin`
- Script:  `$PREFIX/bin/dictate`  (overwrite the existing one)

The `dictate.sh` already self-sets `PATH`/`HOME`, so you can move it into
place with any editor/copy; remember to keep it executable
(`chmod +x $PREFIX/bin/dictate`) and with Unix line endings.

## 2. If copying via Termux from the phone's storage

On the phone, after you've landed the files somewhere in shared storage
(e.g. `~/storage/downloads/dictate-deploy/`):

    termux-setup-storage        # grant storage access once
    cp ~/storage/downloads/dictate-deploy/ggml-model-q4_0.bin ~/whisper.cpp/models/
    cp ~/storage/downloads/dictate-deploy/dictate.sh $PREFIX/bin/dictate
    chmod +x $PREFIX/bin/dictate

## 3. Verify the model loads on the phone

    whisper-cli -m ~/whisper.cpp/models/ggml-model-q4_0.bin -f <some.wav> -l auto

It must report `type = 3 (small)` and `CPU total size ≈ 145 MB (q4_0)`.

## 4. Full dictation test (Roman Urdu)

Run the ticket-05 flow with a Roman-Urdu message:

    dictate start      # record, tap Stop in the notification
    # then long-press-paste into WhatsApp

The transcript must be Roman/Latin script (NOT Urdu script). Because we
use `-l auto`, do NOT pass `-l ur`.

## Notes

- The small model is slower than the old `base.en` on the Exynos 9610
  (CPU-only, 4 GB RAM). If latency is unacceptable, options: accept it,
  re-quantize smaller (q5/q4), or keep `base.en` for English and load the
  RU model only for Roman-Urdu messages.
- Re-run `tools/bench/bench.sh` for a timing/RTF snapshot of the small
  model on the phone.
