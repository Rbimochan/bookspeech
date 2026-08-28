"""Measure Kokoro synthesis throughput on this machine and print the
characters/sec figure to plug into app/estimate.py's DEFAULT_CHARS_PER_SEC.

Usage:
    cd backend
    uv run python ../scripts/benchmark_synthesis.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from app.chunker import chunk_chapter  # noqa: E402
from app.tts.kokoro_engine import KokoroEngine  # noqa: E402

MODEL_DIR = Path(__file__).parent.parent / "backend" / "models"

BENCHMARK_TEXT = (
    "It was the best of times, it was the worst of times, it was the age of "
    "wisdom, it was the age of foolishness, it was the epoch of belief, it "
    "was the epoch of incredulity, it was the season of Light, it was the "
    "season of Darkness, it was the spring of hope, it was the winter of "
    "despair. We had everything before us, we had nothing before us, we "
    "were all going direct to Heaven, we were all going direct the other "
    "way — in short, the period was so far like the present period, that "
    "some of its noisiest authorities insisted on its being received, for "
    "good or for evil, in the superlative degree of comparison only."
) * 4  # ~1000 words, representative chapter-sized sample


def main() -> None:
    engine = KokoroEngine(MODEL_DIR / "kokoro-v1.0.onnx", MODEL_DIR / "voices-v1.0.bin")
    voice = engine.list_voices()[0].id
    chunks = chunk_chapter(0, BENCHMARK_TEXT)
    total_chars = sum(len(c.text) for c in chunks)

    print(f"Synthesizing {total_chars} characters across {len(chunks)} chunks with voice={voice} ...")
    start = time.time()
    for chunk in chunks:
        engine.synthesize(chunk.text, voice=voice, speed=1.0)
    elapsed = time.time() - start

    chars_per_sec = total_chars / elapsed
    words_per_1000 = 1000 / (total_chars / len(BENCHMARK_TEXT.split()))

    print(f"\nElapsed: {elapsed:.1f}s for {total_chars} chars ({len(BENCHMARK_TEXT.split())} words)")
    print(f"Throughput: {chars_per_sec:.1f} chars/sec")
    print(f"           ~{elapsed / (len(BENCHMARK_TEXT.split()) / 1000):.1f}s per 1000 words")
    print(f"\nUpdate DEFAULT_CHARS_PER_SEC in backend/app/estimate.py to {chars_per_sec:.0f} for this machine.")


if __name__ == "__main__":
    main()
