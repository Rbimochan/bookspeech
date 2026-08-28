"""SQLite persistence for books and jobs — no external DB dependency for v1."""

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path

from app.config import settings

_SCHEMA = """
CREATE TABLE IF NOT EXISTS books (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    author TEXT NOT NULL,
    cover_path TEXT,
    language TEXT,
    chapter_count INTEGER NOT NULL,
    source_path TEXT NOT NULL,
    chapters_json TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    book_id TEXT NOT NULL REFERENCES books(id),
    book_title TEXT NOT NULL,
    voice TEXT NOT NULL,
    speed REAL NOT NULL DEFAULT 1.0,
    status TEXT NOT NULL DEFAULT 'queued',
    progress_pct REAL NOT NULL DEFAULT 0,
    current_chapter TEXT,
    error_msg TEXT,
    output_path TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);
"""


def _connect() -> sqlite3.Connection:
    Path(settings.db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(settings.db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


@contextmanager
def get_conn():
    conn = _connect()
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with get_conn() as conn:
        conn.executescript(_SCHEMA)


def insert_book(conn: sqlite3.Connection, book: dict) -> None:
    conn.execute(
        """INSERT INTO books (id, title, author, cover_path, language, chapter_count, source_path, chapters_json)
           VALUES (:id, :title, :author, :cover_path, :language, :chapter_count, :source_path, :chapters_json)""",
        {**book, "chapters_json": json.dumps(book["chapters"])},
    )


def get_book(conn: sqlite3.Connection, book_id: str) -> dict | None:
    row = conn.execute("SELECT * FROM books WHERE id = ?", (book_id,)).fetchone()
    if row is None:
        return None
    data = dict(row)
    data["chapters"] = json.loads(data.pop("chapters_json"))
    return data


def insert_job(conn: sqlite3.Connection, job: dict) -> None:
    conn.execute(
        """INSERT INTO jobs (id, book_id, book_title, voice, speed, status)
           VALUES (:id, :book_id, :book_title, :voice, :speed, :status)""",
        job,
    )


def update_job(conn: sqlite3.Connection, job_id: str, **fields) -> None:
    if not fields:
        return
    set_clause = ", ".join(f"{k} = ?" for k in fields) + ", updated_at = datetime('now')"
    conn.execute(f"UPDATE jobs SET {set_clause} WHERE id = ?", (*fields.values(), job_id))


def get_job(conn: sqlite3.Connection, job_id: str) -> dict | None:
    row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
    return dict(row) if row else None


def list_jobs(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute("SELECT * FROM jobs ORDER BY created_at DESC").fetchall()
    return [dict(r) for r in rows]
