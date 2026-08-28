from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import settings

_MODEL_DIR = Path(__file__).parent.parent / "backend" / "models"
_HAS_KOKORO = (_MODEL_DIR / "kokoro-v1.0.onnx").exists()

pytestmark = pytest.mark.skipif(not _HAS_KOKORO, reason="Kokoro model weights not downloaded")


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "storage_dir", tmp_path)
    monkeypatch.setattr(settings, "db_path", tmp_path / "test.db")
    from app.main import app

    with TestClient(app) as c:
        yield c


def test_preview_default_text(client):
    voice = client.get("/voices").json()[0]["id"]
    resp = client.get(f"/voices/{voice}/preview")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "audio/wav"
    assert len(resp.content) > 1000


def test_preview_is_cached(client, tmp_path):
    voice = client.get("/voices").json()[0]["id"]
    resp1 = client.get(f"/voices/{voice}/preview")
    cache_dir = tmp_path / "cache" / "voice_previews"
    cached_files = list(cache_dir.glob("*.wav"))
    assert len(cached_files) == 1
    mtime_before = cached_files[0].stat().st_mtime

    resp2 = client.get(f"/voices/{voice}/preview")
    assert resp1.content == resp2.content
    assert cached_files[0].stat().st_mtime == mtime_before  # not regenerated


def test_preview_custom_text_produces_different_cache_entry(client, tmp_path):
    voice = client.get("/voices").json()[0]["id"]
    client.get(f"/voices/{voice}/preview")
    client.get(f"/voices/{voice}/preview", params={"text": "A completely different custom sentence."})

    cache_dir = tmp_path / "cache" / "voice_previews"
    assert len(list(cache_dir.glob("*.wav"))) == 2


def test_preview_different_speed_produces_different_cache_entry(client, tmp_path):
    voice = client.get("/voices").json()[0]["id"]
    client.get(f"/voices/{voice}/preview", params={"speed": 1.0})
    client.get(f"/voices/{voice}/preview", params={"speed": 1.3})

    cache_dir = tmp_path / "cache" / "voice_previews"
    assert len(list(cache_dir.glob("*.wav"))) == 2


def test_preview_invalid_voice_returns_400(client):
    resp = client.get("/voices/not-a-real-voice/preview")
    assert resp.status_code == 400


def test_preview_engine_unavailable_returns_503(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "storage_dir", tmp_path)
    monkeypatch.setattr(settings, "db_path", tmp_path / "test2.db")

    # Force the "model weights missing" path regardless of local dev state.
    import app.main as main_module

    original_lifespan = main_module.lifespan

    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def fake_lifespan(app):
        from app import db

        db.init_db()
        app.state.engine = None
        app.state.job_queue = None
        yield

    monkeypatch.setattr(main_module, "lifespan", fake_lifespan)
    main_module.app.router.lifespan_context = fake_lifespan

    with TestClient(main_module.app) as c:
        resp = c.get("/voices/af_heart/preview")
        assert resp.status_code == 503
