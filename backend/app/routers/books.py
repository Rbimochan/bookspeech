import uuid

from fastapi import APIRouter, HTTPException, UploadFile

from app import db
from app.config import settings
from app.disk_space import check_disk_space
from app.epub_parser import is_drm_protected, parse_epub
from app.html_parser import parse_html
from app.pdf_parser import PdfEncryptedError, is_pdf_encrypted, parse_pdf

router = APIRouter()

_MAX_UPLOAD_BYTES = 200 * 1024 * 1024  # 200MB
_SUPPORTED_EXTENSIONS = (".epub", ".pdf", ".html", ".htm")


@router.post("/books/upload")
async def upload_book(file: UploadFile):
    filename_lower = file.filename.lower()
    if not filename_lower.endswith(_SUPPORTED_EXTENSIONS):
        raise HTTPException(status_code=400, detail="Only .epub, .pdf, and .html files are supported")
    is_pdf = filename_lower.endswith(".pdf")
    is_html = filename_lower.endswith((".html", ".htm"))

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")
    if len(content) > _MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File too large (max 200MB)")

    ok, detail = check_disk_space(settings.uploads_dir, required_bytes=len(content) * 3)
    if not ok:
        raise HTTPException(status_code=507, detail=detail)

    book_id = str(uuid.uuid4())
    book_dir = settings.uploads_dir / book_id
    book_dir.mkdir(parents=True, exist_ok=True)
    source_path = book_dir / ("book.pdf" if is_pdf else "book.html" if is_html else "book.epub")
    source_path.write_bytes(content)

    if is_pdf:
        if is_pdf_encrypted(source_path):
            source_path.unlink(missing_ok=True)
            book_dir.rmdir()
            raise HTTPException(
                status_code=400,
                detail="This PDF is password-protected and cannot be converted. Remove the password first before uploading.",
            )
        try:
            book = parse_pdf(source_path, cover_out_dir=book_dir)
        except PdfEncryptedError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Could not parse PDF: {e}") from e
    elif is_html:
        try:
            book = parse_html(source_path, cover_out_dir=book_dir)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Could not parse HTML: {e}") from e
    else:
        if is_drm_protected(source_path):
            source_path.unlink(missing_ok=True)
            book_dir.rmdir()
            raise HTTPException(
                status_code=400,
                detail="This epub is DRM-protected and cannot be converted. Remove DRM first (e.g. with Calibre + a DeDRM plugin) before uploading.",
            )
        try:
            book = parse_epub(source_path, cover_out_dir=book_dir)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Could not parse epub: {e}") from e

    if not book.chapters:
        raise HTTPException(status_code=400, detail="No chapters could be extracted from this file")

    with db.get_conn() as conn:
        db.insert_book(
            conn,
            {
                "id": book_id,
                "title": book.title,
                "author": book.author,
                "cover_path": book.cover_path,
                "language": book.language,
                "chapter_count": len(book.chapters),
                "source_path": str(source_path),
                "chapters": [c.model_dump() for c in book.chapters],
            },
        )

    return {
        "book_id": book_id,
        "title": book.title,
        "author": book.author,
        "chapter_count": len(book.chapters),
        "cover_path": book.cover_path,
    }
