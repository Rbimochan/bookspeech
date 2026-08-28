"""Extract chapter text and metadata from a PDF file.

PDFs carry no structural chapter markup the way epubs do (no spine, no
<h1>/<h2>), so chapter detection is heuristic: scan extracted text for
heading-shaped lines ("Chapter 3", "PART TWO", roman numerals, ...). When no
headings are found at all — scanned books, poetry, anything without a
consistent heading style — fall back to grouping a fixed number of pages per
"chapter" so a long PDF still gets reasonable chapter markers in the
audiobook rather than one multi-hour blob.

No cover image is extracted for PDFs (rendering a page to an image needs a
poppler/pymupdf dependency this project doesn't otherwise need) — cover_path
is always None for PDF-sourced books.
"""

import logging
import re
from pathlib import Path

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from app.models import Book, Chapter

logger = logging.getLogger(__name__)

# A heading-shaped line: "Chapter 3", "CHAPTER III", "Part Two:  Logic".
# Requires chapter/part to be immediately followed by a number or roman
# numeral — a bare "chapter|part|book" prefix matched ANY sentence that
# happened to start with that word after PDF line-wrapping (e.g. "part of
# the path." or "book progresses." from the middle of a paragraph), which
# swallowed huge stretches of real content under a bogus "heading".
_HEADING_RE = re.compile(
    r"^\s*(chapter|part)\s+(\d{1,3}|[ivxlcdm]{1,7})\b[:.\-–—]?\s{0,3}.{0,40}$"
    # Spelled-out chapter numbers ("Part Two: Logic") don't match the digit/
    # roman-numeral form above — require a colon right after the label word
    # instead, since ordinary prose essentially never has "chapter <word>:".
    r"|^\s*(chapter|part)\s+\w+\s*:\s*.{0,40}$"
    r"|^\s*(prologue|epilogue|introduction|preface|afterword)\s*$",
    re.IGNORECASE,
)

# Fallback chaptering when no headings are detected at all.
_PAGES_PER_FALLBACK_CHAPTER = 15

# Hard ceiling on a single chapter's size regardless of how it was produced
# (heading-detected or page-grouped). Protects against pathological cases —
# a book whose real headings don't extract cleanly, a PDF with unusual
# layout, OCR artifacts — where one "chapter" would otherwise balloon to
# tens of thousands of words and take the better part of an hour to
# synthesize before the user sees any progress on the next one.
_MAX_CHAPTER_CHARS = 40_000


class PdfEncryptedError(ValueError):
    """Raised when a PDF is password-protected and cannot be read."""


def is_pdf_encrypted(pdf_path: str | Path) -> bool:
    try:
        reader = PdfReader(str(pdf_path))
        return reader.is_encrypted
    except PdfReadError:
        return False


def _extract_pages(pdf_path: str | Path) -> list[str]:
    reader = PdfReader(str(pdf_path))
    pages = []
    for page in reader.pages:
        try:
            pages.append(page.extract_text() or "")
        except Exception as e:
            logger.warning("Failed to extract text from a PDF page: %s", e)
            pages.append("")
    return pages


def _split_on_headings(pages: list[str]) -> list[Chapter] | None:
    """Split page text into chapters at heading-shaped lines. Returns None if
    no headings were found anywhere, so the caller can fall back."""
    full_text = "\n".join(pages)
    lines = full_text.split("\n")

    heading_indices = [i for i, line in enumerate(lines) if _HEADING_RE.match(line)]
    if not heading_indices:
        return None

    chapters = []
    for idx, start in enumerate(heading_indices):
        end = heading_indices[idx + 1] if idx + 1 < len(heading_indices) else len(lines)
        title = lines[start].strip()
        body_lines = lines[start + 1 : end]
        text = "\n".join(line for line in body_lines if line.strip())
        if text.strip():
            chapters.append(Chapter(index=len(chapters), title=title, text=text))
    return chapters or None


def _split_by_page_groups(pages: list[str]) -> list[Chapter]:
    chapters = []
    for i in range(0, len(pages), _PAGES_PER_FALLBACK_CHAPTER):
        group = pages[i : i + _PAGES_PER_FALLBACK_CHAPTER]
        text = "\n\n".join(p for p in group if p.strip())
        if text.strip():
            chapters.append(
                Chapter(index=len(chapters), title=f"Chapter {len(chapters) + 1}", text=text)
            )
    return chapters


def _split_text_to_max_chars(text: str, max_chars: int) -> list[str]:
    """Split text into pieces no longer than max_chars, preferring paragraph
    breaks, then line breaks, then whitespace — and a hard slice only as a
    last resort. Some PDFs don't preserve blank-line paragraph breaks at all
    (extract_text() just runs everything together with single newlines), so
    a paragraph-only splitter can find zero break points and silently fail
    to cap anything — this tries progressively finer boundaries until it
    actually finds one, so the cap is enforced unconditionally."""
    if len(text) <= max_chars:
        return [text] if text.strip() else []

    for sep in ("\n\n", "\n", " "):
        parts = text.split(sep)
        if len(parts) <= 1:
            continue
        pieces: list[str] = []
        current = ""
        for part in parts:
            candidate = f"{current}{sep}{part}" if current else part
            if len(candidate) > max_chars and current:
                pieces.extend(_split_text_to_max_chars(current, max_chars))
                current = part
            else:
                current = candidate
        if current:
            pieces.extend(_split_text_to_max_chars(current, max_chars))
        return pieces

    # No separator found at all (one unbroken run) — slice rather than loop forever.
    return [text[i : i + max_chars] for i in range(0, len(text), max_chars)]


def _cap_oversized_chapters(chapters: list[Chapter]) -> list[Chapter]:
    result: list[Chapter] = []
    for chapter in chapters:
        if len(chapter.text) <= _MAX_CHAPTER_CHARS:
            result.append(Chapter(index=len(result), title=chapter.title, text=chapter.text))
            continue

        pieces = _split_text_to_max_chars(chapter.text, _MAX_CHAPTER_CHARS)
        for i, piece in enumerate(pieces):
            title = chapter.title if len(pieces) == 1 else f"{chapter.title} (part {i + 1})"
            result.append(Chapter(index=len(result), title=title, text=piece))
    return result


def parse_pdf(pdf_path: str | Path, cover_out_dir: str | Path) -> Book:
    pdf_path = Path(pdf_path)
    reader = PdfReader(str(pdf_path))
    if reader.is_encrypted:
        raise PdfEncryptedError(f"{pdf_path.name} is password-protected")

    meta = reader.metadata or {}
    title = (meta.title or "").strip() or pdf_path.stem
    author = (meta.author or "").strip() or "Unknown"

    pages = _extract_pages(pdf_path)
    chapters = _split_on_headings(pages) or _split_by_page_groups(pages)
    chapters = _cap_oversized_chapters(chapters)

    return Book(title=title, author=author, cover_path=None, language=None, chapters=chapters)
