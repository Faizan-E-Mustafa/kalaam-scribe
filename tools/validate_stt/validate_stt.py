import argparse
import sys
import urllib.request
from pathlib import Path

from faster_whisper import WhisperModel

MODEL_SIZE = "base.en"
KNOWN_SAMPLE_URL = (
    "https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples"
    "/resolve/main/mary_had_lamb.mp3"
)
KNOWN_SAMPLE_PATH = Path(__file__).parent / "samples" / "mary_had_lamb.mp3"


def ensure_sample(audio_path: Path) -> Path:
    if audio_path:
        if not audio_path.exists():
            sys.exit(f"Audio file not found: {audio_path}")
        return audio_path
    KNOWN_SAMPLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not KNOWN_SAMPLE_PATH.exists():
        print(f"Downloading known sample -> {KNOWN_SAMPLE_PATH}")
        urllib.request.urlretrieve(KNOWN_SAMPLE_URL, KNOWN_SAMPLE_PATH)
    return KNOWN_SAMPLE_PATH


def transcribe(audio_path: Path, model_size: str) -> None:
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    segments, info = model.transcribe(
        str(audio_path),
        beam_size=5,
        vad_filter=True,
        language="en",
    )
    print(f"Detected language: {info.language} (p={info.language_probability:.2f})")
    text = "".join(segment.text for segment in segments).strip()
    print("TRANSCRIPT:")
    print(text or "<empty>")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate faster-whisper base.en on the dev machine."
    )
    parser.add_argument("audio", nargs="?", type=Path, help="Path to an audio file")
    parser.add_argument(
        "--model", default=MODEL_SIZE, help=f"Whisper model size (default {MODEL_SIZE})"
    )
    args = parser.parse_args()

    audio = ensure_sample(args.audio)
    transcribe(audio, args.model)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
