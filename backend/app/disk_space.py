"""Disk space estimation and pre-flight checks, so a multi-hour job doesn't
fail two hours in because the disk filled up."""

import shutil
from pathlib import Path

# Rough heuristic: Kokoro at 24kHz mono 16-bit PCM is ~48KB/sec of audio,
# and a typical novel speaks at ~150 words/min -> ~2.5 words/sec, so budget
# generously per character of source text to cover intermediate wav files
# (which dwarf the final AAC-encoded .m4b) plus a safety margin.
BYTES_PER_SOURCE_CHAR = 15


def estimate_job_bytes_needed(total_chapter_chars: int) -> int:
    return total_chapter_chars * BYTES_PER_SOURCE_CHAR


def check_disk_space(path: str | Path, required_bytes: int) -> tuple[bool, str | None]:
    """Returns (ok, detail). detail is a human-readable message when ok is False."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(path)
    if usage.free < required_bytes:
        needed_mb = required_bytes / (1024 * 1024)
        free_mb = usage.free / (1024 * 1024)
        return False, f"Not enough disk space: need ~{needed_mb:.0f}MB, only {free_mb:.0f}MB free at {path}"
    return True, None
