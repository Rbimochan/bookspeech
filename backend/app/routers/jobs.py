import asyncio
import json
import uuid

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from app import db
from app.config import settings
from app.disk_space import check_disk_space, estimate_job_bytes_needed
from app.estimate import estimate_conversion_seconds
from app.models import Book, Chapter

router = APIRouter()


class JobCreateRequest(BaseModel):
    book_id: str
    voice: str
    speed: float = 1.0


def _book_from_row(book_row: dict) -> Book:
    return Book(
        title=book_row["title"],
        author=book_row["author"],
        cover_path=book_row["cover_path"],
        language=book_row["language"],
        chapters=[Chapter(**c) for c in book_row["chapters"]],
    )


@router.post("/jobs")
async def create_job(req: JobCreateRequest, request: Request):
    if request.app.state.job_queue is None:
        raise HTTPException(status_code=503, detail="TTS engine not loaded (Kokoro model weights missing)")
    with db.get_conn() as conn:
        book_row = db.get_book(conn, req.book_id)
        if book_row is None:
            raise HTTPException(status_code=404, detail="Book not found")

        total_chars = sum(len(c["text"]) for c in book_row["chapters"])
        ok, detail = check_disk_space(settings.output_dir, estimate_job_bytes_needed(total_chars))
        if not ok:
            raise HTTPException(status_code=507, detail=detail)

        job_id = str(uuid.uuid4())
        db.insert_job(
            conn,
            {
                "id": job_id,
                "book_id": req.book_id,
                "book_title": book_row["title"],
                "voice": req.voice,
                "speed": req.speed,
                "status": "queued",
            },
        )

    book = _book_from_row(book_row)
    request.app.state.job_queue.submit(job_id, book, req.voice, req.speed)
    return {
        "job_id": job_id,
        "status": "queued",
        "estimated_seconds": estimate_conversion_seconds(total_chars),
    }


@router.post("/jobs/{job_id}/cancel")
async def cancel_job(job_id: str, request: Request):
    if request.app.state.job_queue is None:
        raise HTTPException(status_code=503, detail="TTS engine not loaded (Kokoro model weights missing)")
    with db.get_conn() as conn:
        job = db.get_job(conn, job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")
        if job["status"] in ("done", "failed", "cancelled"):
            raise HTTPException(status_code=409, detail=f"Job already finished (status={job['status']})")

    cancelled = request.app.state.job_queue.cancel(job_id)
    if not cancelled:
        raise HTTPException(status_code=409, detail="Job could not be cancelled (already finishing)")
    return {"job_id": job_id, "status": "cancelling"}


@router.post("/jobs/{job_id}/retry")
async def retry_job(job_id: str, request: Request):
    """Resume a failed job from wherever it left off — per-chapter/chunk wav
    files from the failed attempt are reused rather than re-synthesized."""
    if request.app.state.job_queue is None:
        raise HTTPException(status_code=503, detail="TTS engine not loaded (Kokoro model weights missing)")
    with db.get_conn() as conn:
        job = db.get_job(conn, job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")
        # Anything other than "done" can be (re)started — this also covers a
        # job left "queued"/"synthesizing" in the DB because the backend
        # process restarted (killing the in-memory asyncio task) without the
        # job itself ever reaching a terminal state. Resuming is always safe
        # since already-synthesized chapters are reused from disk either way.
        if job["status"] == "done":
            raise HTTPException(status_code=409, detail="Job already finished")

        book_row = db.get_book(conn, job["book_id"])
        if book_row is None:
            raise HTTPException(status_code=404, detail="Original book no longer available")

        db.update_job(conn, job_id, status="queued", error_msg=None)

    book = _book_from_row(book_row)
    request.app.state.job_queue.submit(job_id, book, job["voice"], job["speed"])
    return {"job_id": job_id, "status": "queued"}


@router.get("/jobs/{job_id}")
async def get_job(job_id: str):
    with db.get_conn() as conn:
        job = db.get_job(conn, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@router.get("/jobs/{job_id}/stream")
async def stream_job(job_id: str, request: Request):
    with db.get_conn() as conn:
        job = db.get_job(conn, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    queue = request.app.state.job_queue.subscribe(job_id)

    async def event_stream():
        yield f"data: {json.dumps({'status': job['status'], 'progress_pct': job['progress_pct']})}\n\n"
        if job["status"] in ("done", "failed", "cancelled"):
            return
        while True:
            event = await queue.get()
            if event.get("event") == "close":
                break
            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.get("/jobs/{job_id}/download")
async def download_job(job_id: str):
    with db.get_conn() as conn:
        job = db.get_job(conn, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if job["status"] != "done" or not job["output_path"]:
        raise HTTPException(status_code=409, detail=f"Job is not finished (status={job['status']})")

    filename = f"{job['book_title']}.m4b"
    return FileResponse(job["output_path"], media_type="audio/mp4", filename=filename)
