"""Turn synthesized chunk wavs into one chaptered, metadata-rich .m4b file.

Pipeline: chunk wavs -> per-chapter wav (with silence gaps) -> full-book wav
-> AAC-encoded .m4b with embedded chapter markers, book metadata, and cover
art. Uses ffmpeg's concat demuxer throughout so large books are streamed
rather than loaded fully into memory.
"""

import logging
import subprocess
import wave
from dataclasses import dataclass
from pathlib import Path

from app.models import Book

logger = logging.getLogger(__name__)

SILENCE_MS_BETWEEN_CHUNKS = 300
SAMPLE_RATE = 24000  # Kokoro's fixed output rate; see app/tts/kokoro_engine.py


@dataclass
class ChapterTiming:
    index: int
    title: str
    start_sec: float
    end_sec: float


def _run_ffmpeg(args: list[str]) -> None:
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", *args]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {' '.join(cmd)}\n{result.stderr}")


def _wav_duration_sec(path: str | Path) -> float:
    with wave.open(str(path), "rb") as f:
        return f.getnframes() / f.getframerate()


def _write_silence_wav(path: Path, duration_ms: int, sample_rate: int = SAMPLE_RATE) -> None:
    _run_ffmpeg(
        [
            "-f", "lavfi",
            "-i", f"anullsrc=r={sample_rate}:cl=mono",
            "-t", f"{duration_ms / 1000:.3f}",
            "-ac", "1",
            "-ar", str(sample_rate),
            str(path),
        ]
    )


def _concat_wavs(wav_paths: list[Path], out_path: Path) -> None:
    """Concatenate wavs with ffmpeg's concat demuxer (streams, no full load into memory)."""
    list_path = out_path.with_suffix(".concat.txt")
    with open(list_path, "w") as f:
        for p in wav_paths:
            f.write(f"file '{Path(p).resolve()}'\n")
    _run_ffmpeg(["-f", "concat", "-safe", "0", "-i", str(list_path), "-c", "copy", str(out_path)])
    list_path.unlink(missing_ok=True)


def assemble_chapter(
    chunk_paths: list[Path],
    out_path: str | Path,
    silence_ms: int = SILENCE_MS_BETWEEN_CHUNKS,
) -> Path:
    """Concatenate a chapter's chunk wavs in order, with a small silence gap
    between them, into a single chapter wav."""
    out_path = Path(out_path)
    if not chunk_paths:
        raise ValueError("assemble_chapter requires at least one chunk")

    if len(chunk_paths) == 1 or silence_ms <= 0:
        interleaved = list(chunk_paths)
    else:
        silence_path = out_path.parent / "_silence.wav"
        _write_silence_wav(silence_path, silence_ms)
        interleaved = []
        for i, p in enumerate(chunk_paths):
            interleaved.append(Path(p))
            if i != len(chunk_paths) - 1:
                interleaved.append(silence_path)

    _concat_wavs(interleaved, out_path)
    return out_path


def assemble_book(chapter_wav_paths: list[Path], chapter_titles: list[str], out_path: str | Path) -> tuple[Path, list[ChapterTiming]]:
    """Concatenate all chapter wavs into one full-book wav, tracking each
    chapter's start/end timestamp as it goes."""
    out_path = Path(out_path)
    timings: list[ChapterTiming] = []
    cursor = 0.0
    for i, (path, title) in enumerate(zip(chapter_wav_paths, chapter_titles)):
        duration = _wav_duration_sec(path)
        timings.append(ChapterTiming(index=i, title=title, start_sec=cursor, end_sec=cursor + duration))
        cursor += duration

    _concat_wavs(chapter_wav_paths, out_path)
    return out_path, timings


def write_ffmetadata(timings: list[ChapterTiming], out_path: str | Path, book: Book | None = None) -> Path:
    """Write an ffmpeg ;FFMETADATA1 file encoding chapter markers (and book
    tags, if provided) for muxing into the final .m4b."""
    out_path = Path(out_path)
    lines = [";FFMETADATA1"]
    if book is not None:
        lines.append(f"title={book.title}")
        lines.append(f"artist={book.author}")
        lines.append("album_artist=BookSpeech / Kokoro")
        lines.append("genre=Audiobook")

    for t in timings:
        lines.append("")
        lines.append("[CHAPTER]")
        lines.append("TIMEBASE=1/1000")
        lines.append(f"START={int(t.start_sec * 1000)}")
        lines.append(f"END={int(t.end_sec * 1000)}")
        lines.append(f"title={t.title}")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


def encode_m4b(
    wav_path: str | Path,
    ffmetadata_path: str | Path,
    out_path: str | Path,
    book: Book,
    cover_path: str | Path | None = None,
) -> Path:
    """Encode the full-book wav to AAC in an .m4b container, muxing in the
    chapter metadata, book tags, and cover art, with audiobook-specific tags
    so apps like BookPlayer/Apple Books recognize it correctly."""
    out_path = Path(out_path)
    args = [
        "-i", str(wav_path),
        "-i", str(ffmetadata_path),
    ]
    if cover_path and Path(cover_path).exists():
        args += ["-i", str(cover_path)]

    args += ["-map_metadata", "1"]

    if cover_path and Path(cover_path).exists():
        args += ["-map", "0:a", "-map", "2:v", "-disposition:v", "attached_pic"]
    else:
        args += ["-map", "0:a"]

    args += [
        "-c:a", "aac",
        "-b:a", "64k",
    ]
    if cover_path and Path(cover_path).exists():
        args += ["-c:v", "copy"]
    args += [
        "-metadata", f"title={book.title}",
        "-metadata", f"artist={book.author}",
        "-metadata", f"album={book.title}",
        "-metadata", f"album_artist=BookSpeech / Kokoro",
        "-metadata", "genre=Audiobook",
        "-metadata:s:v", "title=Cover",
        "-metadata:s:v", "comment=Cover (front)",
        "-f", "mp4",
        "-brand", "M4B ",
        str(out_path),
    ]
    _run_ffmpeg(args)
    return out_path


def set_audiobook_media_kind(m4b_path: str | Path) -> None:
    """Set the iTunes 'stik' atom to Audiobook (2) so players like Apple
    Books/BookPlayer file it under audiobooks rather than music."""
    from mutagen.mp4 import MP4

    audio = MP4(str(m4b_path))
    audio["stik"] = [2]
    audio.save()


def verify_m4b(m4b_path: str | Path) -> dict:
    """Parse the generated .m4b back out and assert chapters/metadata are present."""
    from mutagen.mp4 import MP4

    audio = MP4(str(m4b_path))
    chapters = list(audio.chapters) if audio.chapters else []

    info = {
        "title": audio.get("\xa9nam", [None])[0],
        "artist": audio.get("\xa9ART", [None])[0],
        "length_sec": audio.info.length,
        "chapter_count": len(chapters),
        "has_cover": bool(audio.get("covr")),
    }
    if not info["title"] or not info["length_sec"]:
        raise ValueError(f"Generated m4b failed verification: {info}")
    return info


def cleanup_intermediates(paths: list[Path], keep: bool = False) -> None:
    """Delete intermediate chunk/chapter wav files after successful assembly,
    unless keep=True (useful for debugging a bad conversion)."""
    if keep:
        return
    for p in paths:
        Path(p).unlink(missing_ok=True)
