"""Runtime-editable defaults (default voice, max concurrent jobs), persisted
to storage/settings.json and merged over the .env-derived Settings object at
startup. Output/storage paths are shown but not editable here — changing
them safely requires restarting the process anyway.
"""

import json

from fastapi import APIRouter, Request
from pydantic import BaseModel

from app.config import settings

router = APIRouter()


class SettingsUpdate(BaseModel):
    default_voice: str | None = None
    max_concurrent_jobs: int | None = None


def _settings_file():
    return settings.storage_dir / "settings.json"


def load_persisted_settings() -> dict:
    path = _settings_file()
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def apply_persisted_settings() -> None:
    for key, value in load_persisted_settings().items():
        if hasattr(settings, key):
            setattr(settings, key, value)


@router.get("/settings")
async def get_settings():
    return {
        "default_voice": settings.default_voice,
        "default_speed": settings.default_speed,
        "max_concurrent_jobs": settings.max_concurrent_jobs,
        "output_dir": str(settings.output_dir),
        "storage_dir": str(settings.storage_dir),
    }


@router.put("/settings")
async def update_settings(update: SettingsUpdate, request: Request):
    persisted = load_persisted_settings()

    if update.default_voice is not None:
        settings.default_voice = update.default_voice
        persisted["default_voice"] = update.default_voice

    if update.max_concurrent_jobs is not None:
        if update.max_concurrent_jobs < 1:
            update.max_concurrent_jobs = 1
        settings.max_concurrent_jobs = update.max_concurrent_jobs
        persisted["max_concurrent_jobs"] = update.max_concurrent_jobs
        job_queue = request.app.state.job_queue
        if job_queue is not None:
            job_queue.set_max_concurrent_jobs(update.max_concurrent_jobs)

    _settings_file().parent.mkdir(parents=True, exist_ok=True)
    _settings_file().write_text(json.dumps(persisted, indent=2))

    return await get_settings()
