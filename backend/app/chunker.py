"""Split cleaned chapter text into TTS-safe chunks without breaking mid-sentence.

Sentences are grouped up to a per-engine character ceiling, preferring
paragraph boundaries for better prosody continuity. A single sentence that
exceeds the ceiling on its own is split on clause boundaries (commas/semicolons)
as a fallback, never mid-word.
"""

import nltk

from app.models import Chunk

# Empirically-safe default ceiling for Kokoro-82M (kokoro-onnx handles
# multi-sentence input well up to a few hundred characters before prosody
# degrades). XTTS-v2 will use a different ceiling — pass engine_max_chars
# explicitly rather than relying on this default when engine != kokoro.
DEFAULT_MAX_CHARS = 400

_CLAUSE_SPLIT_CHARS = (";", ":", ",")


def _ensure_punkt() -> None:
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab", quiet=True)


def _split_long_sentence(sentence: str, max_chars: int) -> list[str]:
    """Fallback for a single sentence longer than max_chars: split on clause
    boundaries, then on whitespace as a last resort. Never splits mid-word."""
    if len(sentence) <= max_chars:
        return [sentence]

    for sep in _CLAUSE_SPLIT_CHARS:
        parts = [p.strip() for p in sentence.split(sep) if p.strip()]
        if len(parts) > 1:
            pieces = []
            for i, part in enumerate(parts):
                piece = part if i == len(parts) - 1 else part + sep
                pieces.extend(_split_long_sentence(piece, max_chars))
            return pieces

    words = sentence.split(" ")
    pieces = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if len(candidate) > max_chars and current:
            pieces.append(current)
            current = word
        else:
            current = candidate
    if current:
        pieces.append(current)
    return pieces


def _paragraph_sentences(text: str) -> list[list[str]]:
    """Return sentences grouped by paragraph, preserving paragraph boundaries."""
    _ensure_punkt()
    paragraphs = [p for p in text.split("\n\n") if p.strip()]
    return [nltk.sent_tokenize(p) for p in paragraphs]


def chunk_chapter(chapter_index: int, text: str, max_chars: int = DEFAULT_MAX_CHARS) -> list[Chunk]:
    paragraphs = _paragraph_sentences(text)

    chunks: list[Chunk] = []
    current_text = ""
    current_start_para = 0
    current_end_para = 0
    chunk_idx = 0

    def flush() -> None:
        nonlocal current_text, chunk_idx
        if current_text:
            chunks.append(
                Chunk(
                    chapter_index=chapter_index,
                    chunk_index=chunk_idx,
                    text=current_text,
                    paragraph_start=current_start_para,
                    paragraph_end=current_end_para,
                )
            )
            chunk_idx += 1
        current_text = ""

    for para_idx, sentences in enumerate(paragraphs):
        for raw_sentence in sentences:
            for sentence in _split_long_sentence(raw_sentence, max_chars):
                candidate = f"{current_text} {sentence}".strip() if current_text else sentence
                if len(candidate) <= max_chars:
                    if not current_text:
                        current_start_para = para_idx
                    current_text = candidate
                    current_end_para = para_idx
                else:
                    flush()
                    current_start_para = para_idx
                    current_end_para = para_idx
                    current_text = sentence

    flush()
    return chunks


def chunk_chapter_dry_run(chapter_index: int, text: str, max_chars: int = DEFAULT_MAX_CHARS) -> list[dict]:
    """Return chunk boundaries as plain dicts (no TTS call) for QA before burning compute."""
    return [c.model_dump() for c in chunk_chapter(chapter_index, text, max_chars)]
