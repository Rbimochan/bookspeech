from pathlib import Path

import pytest

from app.epub_parser import parse_epub

FIXTURES = Path(__file__).parent / "fixtures"
SAMPLE_EPUBS = sorted(FIXTURES.glob("*.epub"))


@pytest.fixture(params=SAMPLE_EPUBS, ids=[p.name for p in SAMPLE_EPUBS])
def sample_epub(request, tmp_path):
    return parse_epub(request.param, cover_out_dir=tmp_path)


def test_has_title_and_author(sample_epub):
    assert sample_epub.title.strip()
    assert sample_epub.author.strip()


def test_has_chapters(sample_epub):
    assert len(sample_epub.chapters) >= 3


def test_chapters_have_text_and_sequential_index(sample_epub):
    for i, chapter in enumerate(sample_epub.chapters):
        assert chapter.index == i
        assert chapter.title.strip()
        assert len(chapter.text) > 0
        assert "\n\n" in chapter.text or len(chapter.text.split()) < 50


def test_cover_extracted_when_present(sample_epub):
    if sample_epub.cover_path:
        assert Path(sample_epub.cover_path).exists()
        assert Path(sample_epub.cover_path).stat().st_size > 0


def test_no_html_tags_leak_into_text(sample_epub):
    for chapter in sample_epub.chapters:
        assert "<" not in chapter.text or ">" not in chapter.text


def test_alice_in_wonderland_specifics():
    book = parse_epub(FIXTURES / "11.epub", cover_out_dir=Path("/tmp/bookspeech_test_covers"))
    assert "alice" in book.title.lower()
    assert len(book.chapters) == 12
    assert "rabbit" in book.chapters[0].text.lower()


# ---------------------------------------------------------------------------
# Regression: some epubs put a chapter's number in its own tiny spine item
# ("<h1>2.</h1>", a few bytes) separate from the file with the actual body
# text (which has no heading of its own). The old logic skipped the
# number-only stub as boilerplate (too short), which meant the headingless
# body that followed merged into whatever chapter came *before* it instead
# of starting a new one — an entire real chapter silently vanished into the
# previous chapter's text. Found via a real book ("80/20 Running") where
# Chapters 2-4 disappeared into a 208,877-character "INTRODUCTION".
# ---------------------------------------------------------------------------

def test_chapter_number_in_separate_file_does_not_swallow_the_next_chapter():
    book = parse_epub(FIXTURES / "split_heading_book.epub", cover_out_dir="/tmp")
    assert len(book.chapters) == 3
    assert book.chapters[0].title == "Chapter One"
    assert "chapter two" in book.chapters[1].text.lower()
    assert "split across files" in book.chapters[1].text.lower()
    assert book.chapters[2].title == "Chapter Three"
    # Chapter one's text must NOT have absorbed chapter two's content.
    assert "chapter two" not in book.chapters[0].text.lower()


def test_bare_number_heading_gets_a_positional_fallback_title():
    book = parse_epub(FIXTURES / "split_heading_book.epub", cover_out_dir="/tmp")
    # The "2." heading shouldn't leak into the audiobook's chapter list as
    # a literal "2." — it should read as "Chapter 2".
    assert book.chapters[1].title == "Chapter 2"


# ---------------------------------------------------------------------------
# Regression: some publisher epub templates (Wiley's "For Dummies" series)
# never use real <h1>/<h2>/<h3> tags at all — chapter titles are just a
# styled <p>. With no heading tag anywhere, the whole book collapsed into
# one giant "chapter" (a real book produced a single 434,448-character
# chapter). The spine filename ("ch01.xhtml") is used as a fallback signal
# to start a new chapter, with the title pulled from the first short early
# paragraph that isn't a bare chapter number/label.
# ---------------------------------------------------------------------------

def test_epub_with_no_heading_tags_at_all_still_splits_by_chapter():
    book = parse_epub(FIXTURES / "no_heading_tags_book.epub", cover_out_dir="/tmp")
    assert len(book.chapters) == 3
    assert book.chapters[0].title == "Guitar Theory in a Nutshell"
    assert book.chapters[1].title == "Tuning Up and Getting Started"
    assert book.chapters[2].title == "Chords and Progressions"
    assert "guitar theory" in book.chapters[0].text.lower()
    assert "tune your guitar" in book.chapters[1].text.lower()
    # No cross-contamination between chapters.
    assert "tune your guitar" not in book.chapters[0].text.lower()
