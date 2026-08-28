from pathlib import Path

import pytest

from app.pdf_parser import PdfEncryptedError, is_pdf_encrypted, parse_pdf

FIXTURES = Path(__file__).parent / "fixtures"


def test_parses_title_and_author_from_metadata():
    book = parse_pdf(FIXTURES / "sample.pdf", cover_out_dir="/tmp")
    assert book.title == "The Test Novel"
    assert book.author == "Jane Testwriter"


def test_splits_on_heading_shaped_lines():
    book = parse_pdf(FIXTURES / "sample.pdf", cover_out_dir="/tmp")
    assert len(book.chapters) == 3
    assert book.chapters[0].title == "Chapter 1"
    assert book.chapters[1].title == "Chapter 2"
    assert book.chapters[2].title == "Chapter 3"


def test_chapter_text_contains_expected_content():
    book = parse_pdf(FIXTURES / "sample.pdf", cover_out_dir="/tmp")
    assert "first paragraph" in book.chapters[0].text
    assert "second chapter" in book.chapters[1].text
    assert "final chapter" in book.chapters[2].text


def test_chapters_have_sequential_index():
    book = parse_pdf(FIXTURES / "sample.pdf", cover_out_dir="/tmp")
    for i, chapter in enumerate(book.chapters):
        assert chapter.index == i


def test_cover_path_is_none_for_pdf():
    book = parse_pdf(FIXTURES / "sample.pdf", cover_out_dir="/tmp")
    assert book.cover_path is None


def test_no_headings_falls_back_to_page_grouping():
    book = parse_pdf(FIXTURES / "no_headings.pdf", cover_out_dir="/tmp")
    # 20 pages / 15 pages-per-fallback-chapter -> 2 chapters
    assert len(book.chapters) == 2
    assert book.chapters[0].title == "Chapter 1"
    assert book.chapters[1].title == "Chapter 2"
    assert all(len(c.text) > 0 for c in book.chapters)


def test_encrypted_pdf_detected():
    assert is_pdf_encrypted(FIXTURES / "encrypted.pdf") is True


def test_regular_pdf_not_flagged_as_encrypted():
    assert is_pdf_encrypted(FIXTURES / "sample.pdf") is False


def test_parse_encrypted_pdf_raises():
    with pytest.raises(PdfEncryptedError):
        parse_pdf(FIXTURES / "encrypted.pdf", cover_out_dir="/tmp")


# ---------------------------------------------------------------------------
# Regression: the old heading regex matched ANY line starting with
# "chapter"/"part"/"book" regardless of what followed, so ordinary sentences
# like "part of the path." or "book progresses." (from PDF line-wrapping
# splitting mid-sentence) were misdetected as chapter headings. That swallowed
# huge stretches of real content under a bogus heading — one real book
# produced a 202,631-character "chapter" that took ~34 minutes to synthesize.
# ---------------------------------------------------------------------------

def test_heading_regex_ignores_prose_starting_with_trigger_words():
    from app.pdf_parser import _HEADING_RE

    assert not _HEADING_RE.match("part of the path.")
    assert not _HEADING_RE.match("book progresses.")
    assert not _HEADING_RE.match("chapters like this one are long")
    assert not _HEADING_RE.match("partly cloudy skies were seen")


def test_heading_regex_matches_real_chapter_headings():
    from app.pdf_parser import _HEADING_RE

    assert _HEADING_RE.match("Chapter 7")
    assert _HEADING_RE.match("Chapter 12.")
    assert _HEADING_RE.match("CHAPTER III")
    assert _HEADING_RE.match("Part Two: Logic")
    assert _HEADING_RE.match("Prologue")


def test_no_chapter_ever_exceeds_the_hard_size_cap():
    """Even a heading-detection failure that lumps a huge stretch of a real
    book together must not produce a single monster chapter — it should be
    split at paragraph boundaries so no chapter takes an unreasonable amount
    of synthesis time before the user sees the next progress update."""
    from app.pdf_parser import _MAX_CHAPTER_CHARS, _cap_oversized_chapters
    from app.models import Chapter

    huge_text = "\n\n".join(f"This is paragraph number {i} of a very long chapter." for i in range(3000))
    assert len(huge_text) > _MAX_CHAPTER_CHARS * 2  # actually pathological, like the real case

    capped = _cap_oversized_chapters([Chapter(index=0, title="Chapter 5", text=huge_text)])

    assert len(capped) > 1
    assert all(len(c.text) <= _MAX_CHAPTER_CHARS for c in capped)
    # Sequential indices and no lost/duplicated text.
    for i, c in enumerate(capped):
        assert c.index == i
    assert "".join(c.text for c in capped).replace("\n\n", "") in huge_text.replace("\n\n", "")


def test_small_chapters_are_left_untouched_by_the_cap():
    from app.pdf_parser import _cap_oversized_chapters
    from app.models import Chapter

    chapters = [Chapter(index=0, title="Chapter 1", text="Short chapter text.")]
    result = _cap_oversized_chapters(chapters)
    assert len(result) == 1
    assert result[0].title == "Chapter 1"
    assert result[0].text == "Short chapter text."
