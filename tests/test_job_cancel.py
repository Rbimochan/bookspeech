from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import settings

_MODEL_DIR = Path(__file__).parent.parent / "backend" / "models"
_HAS_KOKORO = (_MODEL_DIR / "kokoro-v1.0.onnx").exists()
FIXTURES = Path(__file__).parent / "fixtures"

pytestmark = pytest.mark.skipif(not _HAS_KOKORO, reason="Kokoro model weights not downloaded")


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "storage_dir", tmp_path)
    monkeypatch.setattr(settings, "db_path", tmp_path / "test.db")
    from app.main import app

    with TestClient(app) as c:
        yield c


def _upload_and_create_job(client, monkeypatch, block_pipeline=True):
    import app.job_queue as job_queue_module

    if block_pipeline:
        # Stand in for the real (slow) pipeline with something that just
        # blocks, so we can reliably catch the job mid-flight to cancel it —
        # patched at the pipeline-call level (not the whole _run method) so
        # _run's own semaphore/cancellation-handling logic still runs for real.
        import time

        def fake_pipeline(*args, **kwargs):
            time.sleep(2)

        monkeypatch.setattr(job_queue_module, "run_full_pipeline", fake_pipeline)

    with open(FIXTURES / "11.epub", "rb") as f:
        upload = client.post("/books/upload", files={"file": ("11.epub", f, "application/epub+zip")})
    book_id = upload.json()["book_id"]
    voice = client.get("/voices").json()[0]["id"]
    job_resp = client.post("/jobs", json={"book_id": book_id, "voice": voice, "speed": 1.0})
    return job_resp.json()["job_id"]


def test_cancel_queued_job_succeeds(client, monkeypatch):
    job_id = _upload_and_create_job(client, monkeypatch)
    resp = client.post(f"/jobs/{job_id}/cancel")
    assert resp.status_code == 200
    assert resp.json()["status"] == "cancelling"


def test_cancelled_job_eventually_reaches_cancelled_status(client, monkeypatch):
    import time

    job_id = _upload_and_create_job(client, monkeypatch)
    client.post(f"/jobs/{job_id}/cancel")

    deadline = time.time() + 5
    status = None
    while time.time() < deadline:
        status = client.get(f"/jobs/{job_id}").json()["status"]
        if status == "cancelled":
            break
        time.sleep(0.05)
    assert status == "cancelled"


def test_cancel_nonexistent_job_returns_404(client):
    resp = client.post("/jobs/does-not-exist/cancel")
    assert resp.status_code == 404


def test_cancel_already_done_job_returns_409(client, monkeypatch):
    from app import db

    job_id = _upload_and_create_job(client, monkeypatch)
    with db.get_conn() as conn:
        db.update_job(conn, job_id, status="done", output_path="/tmp/fake.m4b")

    resp = client.post(f"/jobs/{job_id}/cancel")
    assert resp.status_code == 409


def test_cancelled_job_can_be_retried(client, monkeypatch):
    import time

    job_id = _upload_and_create_job(client, monkeypatch)
    client.post(f"/jobs/{job_id}/cancel")

    deadline = time.time() + 5
    while time.time() < deadline:
        if client.get(f"/jobs/{job_id}").json()["status"] == "cancelled":
            break
        time.sleep(0.05)

    # Stub submit for the retry so it doesn't actually re-run synthesis.
    import app.job_queue as job_queue_module

    monkeypatch.setattr(job_queue_module.JobQueue, "submit", lambda self, *a, **kw: None)
    resp = client.post(f"/jobs/{job_id}/retry")
    assert resp.status_code == 200
