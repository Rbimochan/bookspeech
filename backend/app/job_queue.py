"""Lightweight asyncio background worker with a max-concurrency semaphore.

v1 scale: an in-process asyncio queue is enough for a local single-user tool.
If this ever needs to scale across processes/machines, swap this module for
rq/celery — nothing above it (routers, pipeline) needs to change.
"""

import asyncio
import logging

from app import db
from app.config import settings
from app.models import Book
from app.pipeline import run_full_pipeline
from app.tts.base import TTSEngine

logger = logging.getLogger(__name__)


class JobQueue:
    def __init__(self, engine: TTSEngine, max_concurrent_jobs: int | None = None):
        self._engine = engine
        self._semaphore = asyncio.Semaphore(max_concurrent_jobs or settings.max_concurrent_jobs)
        self._subscribers: dict[str, list[asyncio.Queue]] = {}
        self._tasks: dict[str, asyncio.Task] = {}

    def set_max_concurrent_jobs(self, max_concurrent_jobs: int) -> None:
        # Replacing the semaphore only affects jobs queued from this point
        # on — already-running jobs keep whatever permit they hold. Good
        # enough for a local single-user tool changing this setting live.
        self._semaphore = asyncio.Semaphore(max_concurrent_jobs)

    def subscribe(self, job_id: str) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue()
        self._subscribers.setdefault(job_id, []).append(q)
        return q

    def _publish(self, job_id: str, event: dict) -> None:
        for q in self._subscribers.get(job_id, []):
            q.put_nowait(event)

    def submit(self, job_id: str, book: Book, voice: str, speed: float) -> None:
        task = asyncio.create_task(self._run(job_id, book, voice, speed))
        self._tasks[job_id] = task
        task.add_done_callback(lambda t, jid=job_id: self._tasks.pop(jid, None))

    def cancel(self, job_id: str) -> bool:
        """Cancel a job. Reliable while it's still waiting on the concurrency
        semaphore (queued); once synthesis has actually started, the
        underlying work runs in a thread-pool executor that Python can't
        forcibly interrupt, so cancelling stops us from *awaiting* it and
        marks the job cancelled immediately, but the orphaned thread finishes
        its current chunk in the background before exiting (output discarded)."""
        task = self._tasks.get(job_id)
        if task is None or task.done():
            return False
        task.cancel()
        return True

    async def _run(self, job_id: str, book: Book, voice: str, speed: float) -> None:
        try:
            async with self._semaphore:
                loop = asyncio.get_running_loop()

                def on_progress(status: str, progress_pct: float, current_chapter: str | None) -> None:
                    with db.get_conn() as conn:
                        db.update_job(conn, job_id, status=status, progress_pct=progress_pct, current_chapter=current_chapter)
                    loop.call_soon_threadsafe(
                        self._publish, job_id, {"status": status, "progress_pct": progress_pct, "current_chapter": current_chapter}
                    )

                try:
                    out_path = await loop.run_in_executor(
                        None, run_full_pipeline, job_id, book, voice, speed, self._engine, on_progress
                    )
                    with db.get_conn() as conn:
                        db.update_job(conn, job_id, status="done", progress_pct=100.0, output_path=str(out_path))
                    self._publish(job_id, {"status": "done", "progress_pct": 100.0, "output_path": str(out_path)})
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    logger.exception("Job %s failed", job_id)
                    with db.get_conn() as conn:
                        db.update_job(conn, job_id, status="failed", error_msg=str(e))
                    self._publish(job_id, {"status": "failed", "error_msg": str(e)})
        except asyncio.CancelledError:
            with db.get_conn() as conn:
                db.update_job(conn, job_id, status="cancelled", error_msg="Cancelled by user")
            self._publish(job_id, {"status": "cancelled", "error_msg": "Cancelled by user"})
        finally:
            self._publish(job_id, {"event": "close"})
