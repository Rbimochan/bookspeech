"""Extract clean, ordered chapter text and metadata from an epub file."""

import logging
import re
import zipfile
from pathlib import Path

import ebooklib
from bs4 import BeautifulSoup
from ebooklib import epub

from app.models import Book, Chapter

logger = logging.getLogger(__name__)


class DrmProtectedError(ValueError):
    """Raised when an epub is DRM-protected and cannot be converted."""


def is_drm_protected(epub_path: str | Path) -> bool:
    """Adobe ADEPT / other DRM schemes register an encryption.xml that
    doesn't just cover obfuscated fonts — presence of META-INF/encryption.xml
    with a non-font cipher reference means the content itself is encrypted."""
    try:
        with zipfile.ZipFile(epub_path) as zf:
            if "META-INF/encryption.xml" not in zf.namelist():
                return False
            content = zf.read("META-INF/encryption.xml").decode("utf-8", errors="ignore")
    except (zipfile.BadZipFile, KeyError):
        return False

    # IDPF/Adobe font obfuscation also registers an encryption.xml, but only
    # to scramble embedded font files — that's not DRM on the book content.
    # Real DRM (Adobe ADEPT, etc.) uses a different EncryptionMethod Algorithm.
    _FONT_OBFUSCATION_ALGORITHMS = (
        "http://www.idpf.org/2008/embedding",
        "http://ns.adobe.com/pdf/enc#RC",
    )
    content_lower = content.lower()
    return not any(alg.lower() in content_lower for alg in _FONT_OBFUSCATION_ALGORITHMS)

# Spine items whose filename/id suggests non-content front/back matter.
_SKIP_ID_PATTERNS = re.compile(
    r"(cover|titlepage|title-page|toc|nav|copyright|colophon|dedication|"
    r"pg-header|pg-footer|wrap0000)",
    re.IGNORECASE,
)
# A spine item with fewer than this many characters of text is treated as
# boilerplate (title pages, blank separators) rather than real chapter content.
_MIN_CHAPTER_CHARS = 200

# Heading text that marks a table-of-contents or index page rather than real
# content, even when the spine item's id/filename doesn't give it away.
# Narrating a list of section titles/page numbers or an alphabetical index of
# terms produces useless, disorienting audio.
_SKIP_TITLE_PATTERNS = re.compile(
    r"^((table of )?contents( in detail)?|index)$", re.IGNORECASE
)

# A heading that's just a bare chapter number ("1.", "IV", "12") — some
# epubs put this in its own spine item, separate from the file with the
# actual chapter body/title. Not a real title worth showing in the
# audiobook's chapter list.
_BARE_NUMBER_HEADING_RE = re.compile(r"^(\d{1,3}|[IVXLCDM]{1,7})\.?$", re.IGNORECASE)
_BARE_CHAPTER_LABEL_RE = re.compile(r"^chapter\s+(\d{1,3}|[IVXLCDM]{1,7})\.?$", re.IGNORECASE)

# Some publisher epub templates (e.g. Wiley's "For Dummies" series) never use
# real <h1>/<h2>/<h3> tags — chapter titles are just a styled <p>. The spine
# filename itself is often the only structural signal ("ch01.html"), so it's
# used as a fallback trigger to start a new chapter when no heading exists.
_FILENAME_CHAPTER_RE = re.compile(r"\bch(?:a?p?ter)?[\s_-]?\d{1,3}\b", re.IGNORECASE)

# When falling back to filename-based chapter detection, look for a title in
# one of the first few short <p> tags rather than the full body text.
_MAX_PSEUDO_HEADING_LEN = 80
_MAX_PSEUDO_HEADING_SCAN = 6


def _find_pseudo_heading(soup: BeautifulSoup) -> str | None:
    """Best-effort chapter title when there's no real heading tag: the first
    short early paragraph that isn't just a bare chapter number/label."""
    candidates = soup.find_all("p", limit=_MAX_PSEUDO_HEADING_SCAN)
    for tag in candidates:
        text = tag.get_text(strip=True)
        if not text or len(text) > _MAX_PSEUDO_HEADING_LEN:
            continue
        if _BARE_NUMBER_HEADING_RE.match(text) or _BARE_CHAPTER_LABEL_RE.match(text):
            continue
        return text
    return None


def _get_metadata_value(book: epub.EpubBook, namespace: str, name: str) -> str | None:
    values = book.get_metadata(namespace, name)
    if not values:
        return None
    return values[0][0] or None


def _extract_cover(book: epub.EpubBook, out_dir: Path) -> str | None:
    cover_item = None
    for item in book.get_items_of_type(ebooklib.ITEM_COVER):
        cover_item = item
        break
    if cover_item is None:
        for item in book.get_items_of_type(ebooklib.ITEM_IMAGE):
            if "cover" in item.get_name().lower():
                cover_item = item
                break
    if cover_item is None:
        return None

    ext = Path(cover_item.get_name()).suffix or ".jpg"
    out_path = out_dir / f"cover{ext}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(cover_item.get_content())
    return str(out_path)


def _strip_code_blocks(soup: BeautifulSoup) -> None:
    """Remove <pre>/<code> blocks in place — code listings read aloud as
    prose (variable names, punctuation, indentation) produce useless audio."""
    for tag in soup.find_all(["pre", "code"]):
        tag.decompose()


def _extract_paragraphs(soup: BeautifulSoup) -> list[str]:
    _strip_code_blocks(soup)
    paragraphs = []
    for tag in soup.find_all(["p", "div"]):
        # Skip container divs that just wrap other block elements we'll visit directly.
        if tag.find(["p", "div"]):
            continue
        text = tag.get_text(separator=" ", strip=True)
        if text:
            paragraphs.append(text)
    if not paragraphs:
        # Fallback: some chapters have no <p> tags at all, just loose text nodes.
        text = soup.get_text(separator="\n", strip=True)
        paragraphs = [line for line in text.splitlines() if line.strip()]
    return paragraphs


def _chapter_title(soup: BeautifulSoup, fallback: str) -> str:
    heading = soup.find(["h1", "h2", "h3"])
    if heading:
        title = heading.get_text(strip=True)
        if title:
            return title
    return fallback


def parse_epub(epub_path: str | Path, cover_out_dir: str | Path) -> Book:
    epub_path = Path(epub_path)
    book = epub.read_epub(str(epub_path), options={"ignore_ncx": True})

    title = _get_metadata_value(book, "DC", "title") or epub_path.stem
    author = _get_metadata_value(book, "DC", "creator") or "Unknown"
    language = _get_metadata_value(book, "DC", "language")
    cover_path = _extract_cover(book, Path(cover_out_dir))

    chapters: list[Chapter] = []
    for spine_id, _linear in book.spine:
        item = book.get_item_with_id(spine_id)
        if item is None or item.get_type() != ebooklib.ITEM_DOCUMENT:
            continue
        if _SKIP_ID_PATTERNS.search(item.get_name()) or _SKIP_ID_PATTERNS.search(spine_id):
            continue

        soup = BeautifulSoup(item.get_content(), "html.parser")

        heading = soup.find(["h1", "h2", "h3"])
        if heading and _SKIP_TITLE_PATTERNS.match(heading.get_text(strip=True)):
            logger.info("Skipping table-of-contents spine item %s", item.get_name())
            continue

        paragraphs = _extract_paragraphs(soup)
        text = "\n\n".join(paragraphs)
        has_heading = heading is not None

        # Some publisher templates never use real heading tags at all — the
        # spine filename ("ch01.html") is the only structural signal that
        # this item is its own chapter, not a continuation of the last one.
        filename_suggests_chapter = bool(
            _FILENAME_CHAPTER_RE.search(item.get_name()) or _FILENAME_CHAPTER_RE.search(spine_id)
        )

        if not has_heading:
            if len(text) < _MIN_CHAPTER_CHARS:
                # Likely boilerplate (half-title, blank page) — skip rather
                # than creating a near-empty "chapter".
                logger.info("Skipping short spine item %s (%d chars)", item.get_name(), len(text))
                continue
            if chapters and not filename_suggests_chapter:
                # No heading of its own: treat as a continuation of the
                # previous chapter (epubs frequently split one chapter
                # across multiple files).
                prev = chapters[-1]
                prev.text = f"{prev.text}\n\n{text}"
                continue
            # No heading, but either no prior chapter to attach to, or the
            # filename itself marks this as a new chapter — fall through
            # and start one, titled from a pseudo-heading paragraph if we
            # can find one.

        # A spine item WITH a heading always starts a new chapter, even if
        # its own text is tiny — some epubs put the chapter number in its
        # own file ("<h1>1.</h1>", a few bytes) and the actual body in the
        # next, headingless, file. Treating the number-only stub as
        # boilerplate (the old behavior) meant the real body merged into
        # whatever chapter came before it instead of starting a new one —
        # e.g. an entire "Chapter 1" silently vanished into "INTRODUCTION".

        heading_text = heading.get_text(strip=True) if heading else ""
        if heading and _BARE_NUMBER_HEADING_RE.match(heading_text):
            # A heading that's just "1." or "IV" carries no real title —
            # use a positional fallback instead of putting "1." in the
            # audiobook's chapter list.
            chapter_title = f"Chapter {len(chapters) + 1}"
        elif heading:
            chapter_title = _chapter_title(soup, fallback=f"Chapter {len(chapters) + 1}")
        else:
            # No real heading tag anywhere in this item (the filename-based
            # path above) — try to find a short early paragraph that reads
            # as a title (e.g. Wiley "For Dummies" epubs style chapter
            # titles as a plain <p>, never a heading tag).
            chapter_title = _find_pseudo_heading(soup) or f"Chapter {len(chapters) + 1}"
        chapters.append(Chapter(index=len(chapters), title=chapter_title, text=text))

    return Book(
        title=title,
        author=author,
        cover_path=cover_path,
        language=language,
        chapters=chapters,
    )
