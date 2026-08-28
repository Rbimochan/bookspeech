from fastapi import APIRouter

from app import db

router = APIRouter()


@router.get("/library")
async def list_library():
    with db.get_conn() as conn:
        jobs = db.list_jobs(conn)
    return jobs
