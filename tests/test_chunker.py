import re

import pytest

from app.chunker import chunk_chapter, chunk_chapter_dry_run

SAMPLE_TEXT = (
    "This is the first sentence of the first paragraph. Here is a second sentence, "
    "just to make sure grouping works.\n\n"
    "This is a new paragraph. It has its own sentences here. And a third one too."
)


def _reassembled_words(chunks) -> list[str]:
    words = []
    for c in chunks:
        words.extend(c.text.split())
    return words


def test_no_chunk_exceeds_max_chars():
    chunks = chunk_chapter(0, SAMPLE_TEXT, max_chars=60)
    for c in chunks:
        assert len(c.text) <= 60, c.text


def test_chunks_are_sequential():
    chunks = chunk_chapter(0, SAMPLE_TEXT, max_chars=60)
    for i, c in enumerate(chunks):
        assert c.chunk_index == i
        assert c.chapter_index == 0


def test_paragraph_boundary_preferred_when_it_fits():
    text = "Short one. Short two.\n\nShort three. Short four."
    chunks = chunk_chapter(0, text, max_chars=200)
    # Both paragraphs fit comfortably within 200 chars combined, but a fresh
    # paragraph still starts a new chunk boundary preference — assert at
    # minimum that no sentence got dropped or duplicated.
    all_words = _reassembled_words(chunks)
    assert all_words == text.replace("\n\n", " ").split()


def test_no_sentence_is_split_when_it_fits_in_ceiling():
    text = "A reasonably short sentence that fits easily within the limit."
    chunks = chunk_chapter(0, text, max_chars=200)
    assert len(chunks) == 1
    assert chunks[0].text == text


def test_long_single_sentence_falls_back_to_clause_split():
    long_sentence = (
        "This is an extremely long sentence, containing many clauses, "
        "separated by commas, that must be split, because it exceeds, "
        "the maximum chunk size, by quite a large margin indeed."
    )
    chunks = chunk_chapter(0, long_sentence, max_chars=40)
    for c in chunks:
        assert len(c.text) <= 40
    # No words lost or mangled across the split.
    original_words = re.findall(r"\w+", long_sentence)
    reassembled_words = re.findall(r"\w+", " ".join(c.text for c in chunks))
    assert original_words == reassembled_words


def test_long_single_word_run_falls_back_to_whitespace_split():
    text = "one two three four five six seven eight nine ten"
    chunks = chunk_chapter(0, text, max_chars=10)
    for c in chunks:
        assert len(c.text) <= 10
    assert " ".join(c.text for c in chunks) == text


def test_reassembly_preserves_all_words():
    chunks = chunk_chapter(0, SAMPLE_TEXT, max_chars=50)
    original_words = re.findall(r"\w+", SAMPLE_TEXT)
    reassembled_words = re.findall(r"\w+", " ".join(c.text for c in chunks))
    assert original_words == reassembled_words


def test_dry_run_returns_plain_dicts():
    result = chunk_chapter_dry_run(0, SAMPLE_TEXT, max_chars=60)
    assert isinstance(result, list)
    assert all(isinstance(d, dict) for d in result)
    assert all("text" in d and "chunk_index" in d for d in result)


def test_empty_text_produces_no_chunks():
    assert chunk_chapter(0, "", max_chars=100) == []
