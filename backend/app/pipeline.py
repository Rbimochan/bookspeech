"""Wires parse -> clean -> chunk -> synthesize -> assemble -> encode into one
blocking, per-job pipeline run. Kept synchronous on purpose — the job queue
runs it in a thread executor so it doesn't block the event loop.
"""

from pathlib import Path
from typing import Callable

from app.audio_assembler import (
    assemble_book,
    assemble_chapter,
    cleanup_intermediates,
    encode_m4b,
    set_audiobook_media_kind,
    write_ffmetadata,
)
from app.chunker import chunk_chapter
from app.config import settings
from app.logging_config import job_logger
from app.models import Book
from app.text_cleaner import clean_text
from app.tts.base import TTSEngine

ProgressCallback = Callable[[str, float, str | None], None]  # (status, progress_pct, current_chapter)


def run_full_pipeline(
    job_id: str,
    book: Book,
    voice: str,
    speed: float,
    engine: TTSEngine,
    on_progress: ProgressCallback | None = None,
    keep_intermediates: bool = False,
) -> Path:
    logger = job_logger(job_id)
    job_dir = settings.uploads_dir / job_id
    chunks_dir = job_dir / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)

    def report(status: str, progress_pct: float, current_chapter: str | None = None) -> None:
        logger.info("status=%s progress=%.1f%% chapter=%s", status, progress_pct, current_chapter)
        if on_progress:
            on_progress(status, progress_pct, current_chapter)

    report("cleaning", 0.0)
    all_intermediates: list[Path] = []
    chapter_wav_paths: list[Path] = []
    chapter_titles: list[str] = []

    total_chapters = len(book.chapters)
    for chapter in book.chapters:
        report("synthesizing", (chapter.index / total_chapters) * 80, chapter.title)

        chapter_wav = chunks_dir / f"chapter{chapter.index}.wav"
        if chapter_wav.exists():
            # Resuming a previously failed run: this chapter was already
            # fully synthesized and assembled last time, so reuse it rather
            # than re-running (possibly expensive) synthesis.
            logger.info("Reusing existing chapter wav from a prior attempt: %s", chapter_wav)
            chapter_wav_paths.append(chapter_wav)
            chapter_titles.append(chapter.title)
            continue

        cleaned = clean_text(chapter.text)
        chunks = chunk_chapter(chapter.index, cleaned.text)

        chunk_paths = []
        for chunk in chunks:
            out = chunks_dir / f"ch{chapter.index}_chunk{chunk.chunk_index}.wav"
            if not out.exists():
                engine.synthesize_chunk_to_file(chunk.text, voice, speed, out)
            chunk_paths.append(out)

        assemble_chapter(chunk_paths, chapter_wav)
        chapter_wav_paths.append(chapter_wav)
        chapter_titles.append(chapter.title)
        all_intermediates.extend(chunk_paths)

    report("assembling", 85.0)
    book_wav = job_dir / "book.wav"
    book_wav_path, timings = assemble_book(chapter_wav_paths, chapter_titles, book_wav)
    all_intermediates.extend(chapter_wav_paths)
    all_intermediates.append(book_wav_path)

    meta_path = write_ffmetadata(timings, job_dir / "meta.txt", book=book)

    report("assembling", 95.0)
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = settings.output_dir / f"{job_id}.m4b"
    encode_m4b(book_wav_path, meta_path, out_path, book, cover_path=book.cover_path)
    set_audiobook_media_kind(out_path)

    cleanup_intermediates(all_intermediates, keep=keep_intermediates)
    meta_path.unlink(missing_ok=True)

    report("done", 100.0)
    return out_path
