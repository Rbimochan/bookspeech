import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import db
from app.config import settings
from app.job_queue import JobQueue
from app.logging_config import setup_logging
from app.routers import books, jobs, library, settings_router, voices

setup_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init_db()
    settings_router.apply_persisted_settings()

    model_dir = Path(__file__).parent.parent / "models"
    model_path = model_dir / "kokoro-v1.0.onnx"
    voices_path = model_dir / "voices-v1.0.bin"

    if model_path.exists() and voices_path.exists():
        from app.tts.kokoro_engine import KokoroEngine

        app.state.engine = KokoroEngine(model_path, voices_path)
    else:
        logger.warning(
            "Kokoro model weights not found at %s; /voices and job synthesis will be unavailable "
            "until they're downloaded (see README.md).",
            model_dir,
        )
        app.state.engine = None

    app.state.job_queue = JobQueue(app.state.engine) if app.state.engine else None
    yield


app = FastAPI(title="BookSpeech", lifespan=lifespan)

# Local single-user tool: the frontend is a separate static server (Plan 8),
# so allow it freely rather than hardcoding a port that may vary.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(books.router)
app.include_router(jobs.router)
app.include_router(voices.router)
app.include_router(library.router)
app.include_router(settings_router.router)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
