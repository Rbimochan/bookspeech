"""Extract clean, ordered chapter text and metadata from a standalone .html file."""

import logging
import re
from pathlib import Path

from bs4 import BeautifulSoup

from app.epub_parser import _extract_paragraphs
from app.models import Book, Chapter

logger = logging.getLogger(__name__)

# Headings that split the document into chapters. Anything before the first
# one (or the whole document, if there are none) becomes a single chapter.
_HEADING_TAGS = ("h1", "h2", "h3")

_BARE_NUMBER_HEADING_RE = re.compile(r"^(\d{1,3}|[IVXLCDM]{1,7})\.?$", re.IGNORECASE)
_BARE_CHAPTER_LABEL_RE = re.compile(r"^chapter\s+(\d{1,3}|[IVXLCDM]{1,7})\.?$", re.IGNORECASE)

# A chapter with fewer than this many characters of text is treated as
# boilerplate rather than real content.
_MIN_CHAPTER_CHARS = 200


def _heading_title(heading, index: int) -> str:
    text = heading.get_text(strip=True)
    if not text or _BARE_NUMBER_HEADING_RE.match(text) or _BARE_CHAPTER_LABEL_RE.match(text):
        return f"Chapter {index}"
    return text


def parse_html(html_path: str | Path, cover_out_dir: str | Path) -> Book:
    html_path = Path(html_path)
    raw = html_path.read_bytes()
    soup = BeautifulSoup(raw, "html.parser")

    title_tag = soup.find("title")
    title = (title_tag.get_text(strip=True) if title_tag else None) or html_path.stem

    author_meta = soup.find("meta", attrs={"name": re.compile("^author$", re.IGNORECASE)})
    author = (author_meta.get("content", "").strip() if author_meta else "") or "Unknown"

    body = soup.body or soup

    # Walk the body's block-level descendants once, in document order,
    # splitting into sections wherever a heading tag appears.
    blocks = body.find_all(["h1", "h2", "h3", "p", "div"])
    sections: list[tuple[object | None, list[str]]] = [(None, [])]
    for tag in blocks:
        if tag.name in _HEADING_TAGS:
            sections.append((tag, []))
            continue
        # Skip container divs that just wrap other block elements already visited.
        if tag.name == "div" and tag.find(["p", "div"]):
            continue
        text = tag.get_text(separator=" ", strip=True)
        if text:
            sections[-1][1].append(text)

    chapters: list[Chapter] = []

    preamble_heading, preamble_paragraphs = sections[0]
    preamble_text = "\n\n".join(preamble_paragraphs)
    if len(preamble_text) >= _MIN_CHAPTER_CHARS:
        chapters.append(Chapter(index=0, title="Preface", text=preamble_text))

    for heading, paragraphs in sections[1:]:
        text = "\n\n".join(paragraphs)
        if not text.strip():
            continue
        chapter_title = _heading_title(heading, len(chapters) + 1)
        chapters.append(Chapter(index=len(chapters), title=chapter_title, text=text))

    if not chapters:
        # No headings produced usable sections (e.g. a single untitled
        # document) — fall back to treating the whole body as one chapter.
        paragraphs = _extract_paragraphs(body)
        text = "\n\n".join(paragraphs)
        if text.strip():
            chapters.append(Chapter(index=0, title=title, text=text))

    return Book(title=title, author=author, cover_path=None, language=None, chapters=chapters)
