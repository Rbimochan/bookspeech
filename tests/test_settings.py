import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import settings

_MODEL_DIR = Path(__file__).parent.parent / "backend" / "models"
_HAS_KOKORO = (_MODEL_DIR / "kokoro-v1.0.onnx").exists()


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "storage_dir", tmp_path)
    monkeypatch.setattr(settings, "db_path", tmp_path / "test.db")
    from app.main import app

    with TestClient(app) as c:
        yield c


def test_get_settings_returns_defaults(client):
    resp = client.get("/settings")
    assert resp.status_code == 200
    data = resp.json()
    assert "default_voice" in data
    assert "max_concurrent_jobs" in data
    assert "output_dir" in data


def test_update_default_voice(client):
    resp = client.put("/settings", json={"default_voice": "am_adam"})
    assert resp.status_code == 200
    assert resp.json()["default_voice"] == "am_adam"

    resp2 = client.get("/settings")
    assert resp2.json()["default_voice"] == "am_adam"


def test_update_persists_to_disk(client, tmp_path):
    client.put("/settings", json={"default_voice": "bf_emma", "max_concurrent_jobs": 3})
    settings_file = tmp_path / "settings.json"
    assert settings_file.exists()
    data = json.loads(settings_file.read_text())
    assert data["default_voice"] == "bf_emma"
    assert data["max_concurrent_jobs"] == 3


def test_max_concurrent_jobs_floor_is_one(client):
    resp = client.put("/settings", json={"max_concurrent_jobs": 0})
    assert resp.json()["max_concurrent_jobs"] == 1


def test_persisted_settings_applied_on_next_startup(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "storage_dir", tmp_path)
    monkeypatch.setattr(settings, "db_path", tmp_path / "test.db")

    (tmp_path).mkdir(parents=True, exist_ok=True)
    (tmp_path / "settings.json").write_text(json.dumps({"default_voice": "zm_yunxi"}))

    from app.main import app

    with TestClient(app) as c:
        resp = c.get("/settings")
        assert resp.json()["default_voice"] == "zm_yunxi"
