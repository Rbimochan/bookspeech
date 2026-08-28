"""Standalone manual QA: synthesize one paragraph end-to-end and save it.

Usage:
    cd backend
    uv run python ../scripts/tts_smoke_test.py "Some text to speak." --voice af_heart
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from app.tts.kokoro_engine import KokoroEngine  # noqa: E402

DEFAULT_TEXT = (
    "The quick brown fox jumps over the lazy dog, while the old clock on the "
    "wall ticks steadily past midnight."
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("text", nargs="?", default=DEFAULT_TEXT)
    parser.add_argument("--voice", default="af_heart")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--model-path", default="models/kokoro-v1.0.onnx")
    parser.add_argument("--voices-path", default="models/voices-v1.0.bin")
    parser.add_argument("--out", default="/tmp/bookspeech_smoke_test.wav")
    args = parser.parse_args()

    print(f"Loading Kokoro from {args.model_path} / {args.voices_path} ...")
    engine = KokoroEngine(args.model_path, args.voices_path)

    print(f"Voices available: {len(engine.list_voices())}")
    print(f"Synthesizing with voice={args.voice} speed={args.speed} ...")
    start = time.time()
    path = engine.synthesize_chunk_to_file(args.text, args.voice, args.speed, args.out)
    elapsed = time.time() - start

    print(f"Wrote {path} in {elapsed:.2f}s")
    print(f"Play it with: afplay {path}   (macOS)  or  aplay {path}  (Linux)")


if __name__ == "__main__":
    main()
