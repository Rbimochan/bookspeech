"""Final smoke test: convert one full-length real novel start to finish via
the actual pipeline (no truncation), then report duration/chapter count so a
human can spot-check first/last/middle chapters for quality.

Usage:
    cd backend
    uv run python ../scripts/smoke_test_full_novel.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from app.epub_parser import parse_epub  # noqa: E402
from app.audio_assembler import verify_m4b  # noqa: E402
from app.pipeline import run_full_pipeline  # noqa: E402
from app.tts.kokoro_engine import KokoroEngine  # noqa: E402

FIXTURES = Path(__file__).parent.parent / "tests" / "fixtures"
MODEL_DIR = Path(__file__).parent.parent / "backend" / "models"
OUT_DIR = Path("/tmp/bookspeech_smoke")


def main() -> None:
    epub_path = FIXTURES / "84.epub"  # Frankenstein — full novel, 24 chapters + letters
    book = parse_epub(epub_path, cover_out_dir=OUT_DIR)
    total_chars = sum(len(c.text) for c in book.chapters)
    print(f"Book: {book.title} by {book.author}")
    print(f"Chapters: {len(book.chapters)}, total chars: {total_chars}")

    engine = KokoroEngine(MODEL_DIR / "kokoro-v1.0.onnx", MODEL_DIR / "voices-v1.0.bin")
    voice = engine.list_voices()[0].id

    def on_progress(status, pct, chapter):
        print(f"[{time.strftime('%H:%M:%S')}] {status} {pct:.1f}%  {chapter or ''}")

    import app.config as config_module

    config_module.settings.storage_dir = OUT_DIR
    job_id = "smoke-test-full-novel"

    start = time.time()
    out_path = run_full_pipeline(job_id, book, voice, 1.0, engine, on_progress=on_progress)
    elapsed = time.time() - start

    info = verify_m4b(out_path)
    print(f"\nDone in {elapsed / 60:.1f} minutes.")
    print(f"Output: {out_path}")
    print(f"Verified: title={info['title']!r} chapters={info['chapter_count']} length={info['length_sec'] / 60:.1f}min cover={info['has_cover']}")
    print(f"\nListen check: afplay {out_path}")


if __name__ == "__main__":
    main()
