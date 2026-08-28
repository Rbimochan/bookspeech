import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import settings

_MODEL_DIR = Path(__file__).parent.parent / "backend" / "models"
_HAS_KOKORO = (_MODEL_DIR / "kokoro-v1.0.onnx").exists()
FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def client(tmp_path, monkeypatch):
    # Isolate storage + db per test so tests don't collide/pollute the real dev db.
    monkeypatch.setattr(settings, "storage_dir", tmp_path)
    monkeypatch.setattr(settings, "db_path", tmp_path / "test.db")
    from app.main import app

    with TestClient(app) as c:
        yield c


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_upload_book(client):
    with open(FIXTURES / "11.epub", "rb") as f:
        resp = client.post("/books/upload", files={"file": ("11.epub", f, "application/epub+zip")})
    assert resp.status_code == 200
    data = resp.json()
    assert "alice" in data["title"].lower()
    assert data["chapter_count"] > 0
    assert "book_id" in data


def test_upload_rejects_non_epub(client):
    resp = client.post("/books/upload", files={"file": ("notabook.txt", b"hello", "text/plain")})
    assert resp.status_code == 400


def test_upload_rejects_corrupt_epub(client):
    resp = client.post("/books/upload", files={"file": ("bad.epub", b"not a real epub", "application/epub+zip")})
    assert resp.status_code == 400


def test_upload_pdf_book(client):
    with open(FIXTURES / "sample.pdf", "rb") as f:
        resp = client.post("/books/upload", files={"file": ("sample.pdf", f, "application/pdf")})
    assert resp.status_code == 200
    data = resp.json()
    assert data["title"] == "The Test Novel"
    assert data["author"] == "Jane Testwriter"
    assert data["chapter_count"] == 3


def test_upload_rejects_encrypted_pdf(client):
    with open(FIXTURES / "encrypted.pdf", "rb") as f:
        resp = client.post("/books/upload", files={"file": ("encrypted.pdf", f, "application/pdf")})
    assert resp.status_code == 400
    assert "password-protected" in resp.json()["detail"]


def test_upload_rejects_corrupt_pdf(client):
    resp = client.post("/books/upload", files={"file": ("bad.pdf", b"not a real pdf", "application/pdf")})
    assert resp.status_code == 400


@pytest.mark.skipif(not _HAS_KOKORO, reason="Kokoro model weights not downloaded")
def test_voices_list(client):
    resp = client.get("/voices")
    assert resp.status_code == 200
    voices = resp.json()
    assert len(voices) > 0
    assert "id" in voices[0]


@pytest.mark.skipif(not _HAS_KOKORO, reason="Kokoro model weights not downloaded")
def test_job_not_found(client):
    resp = client.get("/jobs/does-not-exist")
    assert resp.status_code == 404


@pytest.mark.skipif(not _HAS_KOKORO, reason="Kokoro model weights not downloaded")
def test_create_job_requires_valid_book(client):
    resp = client.post("/jobs", json={"book_id": "nope", "voice": "af_heart", "speed": 1.0})
    assert resp.status_code == 404


@pytest.mark.skipif(not _HAS_KOKORO, reason="Kokoro model weights not downloaded")
def test_full_job_lifecycle_and_download(client, monkeypatch):
    # Keep the real end-to-end run fast: truncate every chapter's text to a
    # couple sentences before synthesis so the test doesn't take minutes.
    import app.pipeline as pipeline_module

    original_clean_text = pipeline_module.clean_text

    def truncated_clean_text(text, *args, **kwargs):
        return original_clean_text(text[:150], *args, **kwargs)

    monkeypatch.setattr(pipeline_module, "clean_text", truncated_clean_text)

    with open(FIXTURES / "11.epub", "rb") as f:
        upload = client.post("/books/upload", files={"file": ("11.epub", f, "application/epub+zip")})
    book_id = upload.json()["book_id"]

    voice = client.get("/voices").json()[0]["id"]
    job_resp = client.post("/jobs", json={"book_id": book_id, "voice": voice, "speed": 1.0})
    assert job_resp.status_code == 200
    job_id = job_resp.json()["job_id"]

    deadline = time.time() + 120
    status = None
    while time.time() < deadline:
        status_resp = client.get(f"/jobs/{job_id}")
        status = status_resp.json()["status"]
        if status in ("done", "failed"):
            break
        time.sleep(1)

    assert status == "done", f"job did not complete in time (last status={status})"

    download = client.get(f"/jobs/{job_id}/download")
    assert download.status_code == 200
    assert download.headers["content-type"] == "audio/mp4"
    assert len(download.content) > 1000

    library = client.get("/library")
    assert library.status_code == 200
    assert any(j["id"] == job_id for j in library.json())
