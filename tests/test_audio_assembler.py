from pathlib import Path

import numpy as np
import pytest

from app.audio_assembler import (
    assemble_book,
    assemble_chapter,
    cleanup_intermediates,
    encode_m4b,
    set_audiobook_media_kind,
    verify_m4b,
    write_ffmetadata,
)
from app.audio_io import read_wav, write_wav
from app.models import Book, Chapter

_MODEL_DIR = Path(__file__).parent.parent / "backend" / "models"
_HAS_KOKORO = (_MODEL_DIR / "kokoro-v1.0.onnx").exists()


def _make_tone_wav(path: Path, freq: float, duration_sec: float, sample_rate: int = 24000) -> Path:
    t = np.linspace(0, duration_sec, int(sample_rate * duration_sec), endpoint=False)
    audio = (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    write_wav(path, audio, sample_rate)
    return path


def test_assemble_chapter_concatenates_with_silence(tmp_path):
    chunk1 = _make_tone_wav(tmp_path / "c0.wav", 440, 0.5)
    chunk2 = _make_tone_wav(tmp_path / "c1.wav", 880, 0.5)
    out = assemble_chapter([chunk1, chunk2], tmp_path / "chapter0.wav", silence_ms=200)

    audio, sr = read_wav(out)
    duration = len(audio) / sr
    # 0.5 + 0.2 (silence) + 0.5 = 1.2s, allow small ffmpeg rounding tolerance
    assert 1.1 <= duration <= 1.3


def test_assemble_chapter_single_chunk_no_silence(tmp_path):
    chunk1 = _make_tone_wav(tmp_path / "c0.wav", 440, 0.3)
    out = assemble_chapter([chunk1], tmp_path / "chapter0.wav")
    audio, sr = read_wav(out)
    assert abs(len(audio) / sr - 0.3) < 0.05


def test_assemble_book_tracks_chapter_timings(tmp_path):
    ch0 = _make_tone_wav(tmp_path / "ch0.wav", 440, 1.0)
    ch1 = _make_tone_wav(tmp_path / "ch1.wav", 880, 2.0)
    out, timings = assemble_book([ch0, ch1], ["Chapter One", "Chapter Two"], tmp_path / "book.wav")

    assert len(timings) == 2
    assert timings[0].start_sec == 0.0
    assert abs(timings[0].end_sec - 1.0) < 0.05
    assert abs(timings[1].start_sec - 1.0) < 0.05
    assert abs(timings[1].end_sec - 3.0) < 0.05

    audio, sr = read_wav(out)
    assert abs(len(audio) / sr - 3.0) < 0.1


def test_write_ffmetadata_format(tmp_path):
    from app.audio_assembler import ChapterTiming

    timings = [
        ChapterTiming(index=0, title="Chapter One", start_sec=0.0, end_sec=10.0),
        ChapterTiming(index=1, title="Chapter Two", start_sec=10.0, end_sec=25.5),
    ]
    book = Book(title="Test Book", author="Test Author", cover_path=None, language="en", chapters=[])
    out = write_ffmetadata(timings, tmp_path / "meta.txt", book=book)

    content = out.read_text()
    assert content.startswith(";FFMETADATA1")
    assert "title=Test Book" in content
    assert "[CHAPTER]" in content
    assert "START=0" in content
    assert "END=10000" in content
    assert "START=10000" in content
    assert "END=25500" in content
    assert content.count("[CHAPTER]") == 2


def test_full_pipeline_produces_valid_m4b(tmp_path):
    """End-to-end: two chapter wavs -> ffmetadata -> encoded, verified .m4b."""
    ch0 = _make_tone_wav(tmp_path / "ch0.wav", 440, 1.0)
    ch1 = _make_tone_wav(tmp_path / "ch1.wav", 880, 1.5)
    book_wav, timings = assemble_book([ch0, ch1], ["Intro", "Chapter One"], tmp_path / "book.wav")

    book = Book(title="Assembled Test Book", author="A. Test", cover_path=None, language="en", chapters=[])
    meta_path = write_ffmetadata(timings, tmp_path / "meta.txt", book=book)

    m4b_path = encode_m4b(book_wav, meta_path, tmp_path / "out.m4b", book)
    set_audiobook_media_kind(m4b_path)

    info = verify_m4b(m4b_path)
    assert info["title"] == "Assembled Test Book"
    assert info["artist"] == "A. Test"
    assert info["chapter_count"] == 2
    assert abs(info["length_sec"] - 2.5) < 0.3


def test_cleanup_intermediates_deletes_files(tmp_path):
    files = [_make_tone_wav(tmp_path / f"c{i}.wav", 440, 0.1) for i in range(3)]
    cleanup_intermediates(files, keep=False)
    assert all(not f.exists() for f in files)


def test_cleanup_intermediates_keeps_when_requested(tmp_path):
    files = [_make_tone_wav(tmp_path / f"c{i}.wav", 440, 0.1) for i in range(3)]
    cleanup_intermediates(files, keep=True)
    assert all(f.exists() for f in files)


@pytest.mark.skipif(not _HAS_KOKORO, reason="Kokoro model weights not downloaded; see README.md")
def test_end_to_end_epub_to_m4b(tmp_path):
    """Full pipeline against a real epub: parse -> clean -> chunk -> real
    Kokoro TTS -> assemble -> encode -> verify a playable .m4b."""
    from app.chunker import chunk_chapter
    from app.epub_parser import parse_epub
    from app.text_cleaner import clean_text
    from app.tts.kokoro_engine import KokoroEngine

    fixtures = Path(__file__).parent / "fixtures"
    book_data = parse_epub(fixtures / "11.epub", cover_out_dir=tmp_path)
    chapters = book_data.chapters[:2]

    engine = KokoroEngine(_MODEL_DIR / "kokoro-v1.0.onnx", _MODEL_DIR / "voices-v1.0.bin")
    voice = engine.list_voices()[0].id

    chapter_wav_paths = []
    for chapter in chapters:
        cleaned = clean_text(chapter.text[:400]).text  # keep synthesis fast for a test
        chunks = chunk_chapter(chapter.index, cleaned, max_chars=300)
        chunk_paths = []
        for chunk in chunks[:2]:  # cap chunks per chapter to keep the test fast
            out = tmp_path / f"ch{chapter.index}_chunk{chunk.chunk_index}.wav"
            engine.synthesize_chunk_to_file(chunk.text, voice, 1.0, out)
            chunk_paths.append(out)
        chapter_wav = assemble_chapter(chunk_paths, tmp_path / f"chapter{chapter.index}.wav")
        chapter_wav_paths.append(chapter_wav)

    book_wav, timings = assemble_book(chapter_wav_paths, [c.title for c in chapters], tmp_path / "book.wav")
    meta_path = write_ffmetadata(timings, tmp_path / "meta.txt", book=book_data)
    m4b_path = encode_m4b(book_wav, meta_path, tmp_path / "final.m4b", book_data, cover_path=book_data.cover_path)
    set_audiobook_media_kind(m4b_path)

    info = verify_m4b(m4b_path)
    assert info["chapter_count"] == 2
    assert info["length_sec"] > 0
    assert info["has_cover"]
