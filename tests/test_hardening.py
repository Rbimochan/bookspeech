from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import db
from app.config import settings
from app.disk_space import check_disk_space, estimate_job_bytes_needed
from app.epub_parser import is_drm_protected
from app.estimate import estimate_conversion_seconds

_MODEL_DIR = Path(__file__).parent.parent / "backend" / "models"
_HAS_KOKORO = (_MODEL_DIR / "kokoro-v1.0.onnx").exists()
FIXTURES = Path(__file__).parent / "fixtures"


# ---------------------------------------------------------------------------
# DRM detection
# ---------------------------------------------------------------------------

def test_drm_protected_epub_detected():
    assert is_drm_protected(FIXTURES / "drm_samples" / "drm_protected.epub") is True


def test_font_obfuscation_only_not_flagged_as_drm():
    assert is_drm_protected(FIXTURES / "drm_samples" / "font_obfuscated.epub") is False


def test_regular_epub_not_flagged_as_drm():
    assert is_drm_protected(FIXTURES / "11.epub") is False


def test_corrupt_file_not_flagged_as_drm():
    # is_drm_protected should fail safe (False) rather than raise on garbage input.
    bad = FIXTURES / "_not_a_zip.epub"
    bad.write_bytes(b"not a zip file at all")
    try:
        assert is_drm_protected(bad) is False
    finally:
        bad.unlink()


# ---------------------------------------------------------------------------
# Disk space checks
# ---------------------------------------------------------------------------

def test_disk_space_check_passes_with_plenty_of_room(tmp_path):
    ok, detail = check_disk_space(tmp_path, required_bytes=1024)
    assert ok is True
    assert detail is None


def test_disk_space_check_fails_when_insufficient(tmp_path):
    ok, detail = check_disk_space(tmp_path, required_bytes=10**18)  # 1 exabyte
    assert ok is False
    assert "Not enough disk space" in detail


def test_estimate_job_bytes_scales_with_text_length():
    small = estimate_job_bytes_needed(1000)
    large = estimate_job_bytes_needed(100000)
    assert large > small


# ---------------------------------------------------------------------------
# Time estimate
# ---------------------------------------------------------------------------

def test_estimate_conversion_seconds_scales_with_length():
    short = estimate_conversion_seconds(1000)
    long = estimate_conversion_seconds(100000)
    assert long > short
    assert short > 0


# ---------------------------------------------------------------------------
# API-level: upload rejection + disk space + retry
# ---------------------------------------------------------------------------

@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "storage_dir", tmp_path)
    monkeypatch.setattr(settings, "db_path", tmp_path / "test.db")
    from app.main import app

    with TestClient(app) as c:
        yield c


def test_upload_rejects_empty_file(client):
    resp = client.post("/books/upload", files={"file": ("empty.epub", b"", "application/epub+zip")})
    assert resp.status_code == 400


def test_upload_rejects_drm_protected_epub(client):
    with open(FIXTURES / "drm_samples" / "drm_protected.epub", "rb") as f:
        resp = client.post("/books/upload", files={"file": ("drm.epub", f, "application/epub+zip")})
    assert resp.status_code == 400
    assert "DRM" in resp.json()["detail"]


def test_upload_insufficient_disk_space_returns_507(client, monkeypatch):
    def fake_check(*args, **kwargs):
        return False, "Not enough disk space: pretend the disk is full"

    import app.routers.books as books_module

    monkeypatch.setattr(books_module, "check_disk_space", fake_check)

    with open(FIXTURES / "11.epub", "rb") as f:
        resp = client.post("/books/upload", files={"file": ("11.epub", f, "application/epub+zip")})
    assert resp.status_code == 507


@pytest.mark.skipif(not _HAS_KOKORO, reason="Kokoro model weights not downloaded")
def test_retry_accepts_queued_status(client, monkeypatch):
    """Any non-"done" status is retryable, not just "failed" — this also
    covers a job orphaned by a backend restart (left "queued"/"synthesizing"
    in the DB with no live asyncio task backing it)."""
    import app.job_queue as job_queue_module

    monkeypatch.setattr(job_queue_module.JobQueue, "submit", lambda self, *a, **kw: None)

    with open(FIXTURES / "11.epub", "rb") as f:
        upload = client.post("/books/upload", files={"file": ("11.epub", f, "application/epub+zip")})
    book_id = upload.json()["book_id"]
    voice = client.get("/voices").json()[0]["id"]
    job_resp = client.post("/jobs", json={"book_id": book_id, "voice": voice, "speed": 1.0})
    job_id = job_resp.json()["job_id"]

    retry_resp = client.post(f"/jobs/{job_id}/retry")
    assert retry_resp.status_code == 200


@pytest.mark.skipif(not _HAS_KOKORO, reason="Kokoro model weights not downloaded")
def test_retry_rejects_done_status(client, monkeypatch):
    import app.job_queue as job_queue_module

    monkeypatch.setattr(job_queue_module.JobQueue, "submit", lambda self, *a, **kw: None)

    with open(FIXTURES / "11.epub", "rb") as f:
        upload = client.post("/books/upload", files={"file": ("11.epub", f, "application/epub+zip")})
    book_id = upload.json()["book_id"]
    voice = client.get("/voices").json()[0]["id"]
    job_resp = client.post("/jobs", json={"book_id": book_id, "voice": voice, "speed": 1.0})
    job_id = job_resp.json()["job_id"]

    with db.get_conn() as conn:
        db.update_job(conn, job_id, status="done", output_path="/tmp/fake.m4b")

    retry_resp = client.post(f"/jobs/{job_id}/retry")
    assert retry_resp.status_code == 409


def test_retry_nonexistent_job_returns_404(client):
    resp = client.post("/jobs/does-not-exist/retry")
    assert resp.status_code == 404


@pytest.mark.skipif(not _HAS_KOKORO, reason="Kokoro model weights not downloaded")
def test_resume_reuses_existing_chapter_wav(tmp_path, monkeypatch):
    """A chapter wav already on disk from a prior failed attempt should be
    reused rather than re-synthesized."""
    import time

    from app.epub_parser import parse_epub
    from app.pipeline import run_full_pipeline
    from app.tts.kokoro_engine import KokoroEngine

    monkeypatch.setattr(settings, "storage_dir", tmp_path)

    book = parse_epub(FIXTURES / "11.epub", cover_out_dir=tmp_path)
    book.chapters = book.chapters[:1]  # single chapter, keeps this fast
    book.chapters[0].text = book.chapters[0].text[:200]

    engine = KokoroEngine(_MODEL_DIR / "kokoro-v1.0.onnx", _MODEL_DIR / "voices-v1.0.bin")
    voice = engine.list_voices()[0].id
    job_id = "resume-test-job"

    # Simulate a prior attempt that already finished synthesizing chapter 0
    # before the process died (e.g. crashed during assembly of a later
    # chapter) — pre-seed the chapter wav a real run would have produced.
    job_dir = settings.uploads_dir / job_id
    chunks_dir = job_dir / "chunks"
    chunks_dir.mkdir(parents=True)
    fake_chapter_wav = chunks_dir / "chapter0.wav"

    from app.audio_io import write_wav
    import numpy as np

    write_wav(fake_chapter_wav, np.zeros(2400, dtype=np.float32), 24000)

    out_path = run_full_pipeline(job_id, book, voice, 1.0, engine)
    assert out_path.exists()

    # The key assertion: synthesis for chapter 0 was skipped because its wav
    # already existed, logged via the "Reusing existing chapter wav" line.
    log_path = settings.logs_dir / f"{job_id}.log"
    assert "Reusing existing chapter wav" in log_path.read_text()
