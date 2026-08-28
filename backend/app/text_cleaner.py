"""Turn raw extracted chapter text into TTS-ready text.

A pluggable pipeline of ordered normalization rules. Each rule is a plain
`str -> str` function; add new ones to DEFAULT_RULES without touching the
core `clean_text` driver.
"""

import html
import re
from dataclasses import dataclass, field

from num2words import num2words

# ---------------------------------------------------------------------------
# Individual rules (ordered — order matters, see DEFAULT_RULES at the bottom)
# ---------------------------------------------------------------------------

_TAG_RE = re.compile(r"<[^>]+>")
_FOOTNOTE_MARKER_RE = re.compile(r"(?<=[a-zA-Z.,!?\"'])\s*(\[\d{1,3}\]|\(\d{1,3}\)|[¹²³⁰-⁹]+)(?=\s|$)")
_HYPHEN_LINEBREAK_RE = re.compile(r"(\w)-\n(\w)")
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")
_MULTI_SPACE_RE = re.compile(r"[ \t]{2,}")
_TRAILING_SPACE_RE = re.compile(r"[ \t]+\n")

_CURRENCY_RE = re.compile(r"([$£€])\s?(\d[\d,]*(?:\.\d+)?)")
_DECIMAL_RE = re.compile(r"\b(\d+)\.(\d+)\b")
_ORDINAL_RE = re.compile(r"\b(\d+)(st|nd|rd|th)\b", re.IGNORECASE)
_PLAIN_NUMBER_RE = re.compile(r"(?<![\w.])\d{1,9}(?![\w.])")
_YEAR_RANGE_RE = re.compile(r"\b(1[0-9]{3}|20[0-9]{2})\b")

_ABBREVIATIONS = {
    "Mr.": "Mister",
    "Mrs.": "Missus",
    "Ms.": "Miz",
    "Dr.": "Doctor",
    "Prof.": "Professor",
    "St.": "Saint",
    "Jr.": "Junior",
    "Sr.": "Senior",
    "vs.": "versus",
    "etc.": "et cetera",
    "e.g.": "for example",
    "i.e.": "that is",
    "approx.": "approximately",
    "no.": "number",
    "No.": "Number",
}
_ABBREVIATION_RE = re.compile(
    r"(?<!\w)(" + "|".join(re.escape(k) for k in sorted(_ABBREVIATIONS, key=len, reverse=True)) + r")(?!\w)"
)

_CURLY_QUOTES = str.maketrans({
    "‘": "'", "’": "'",
    "“": '"', "”": '"',
})
_EM_DASH_RE = re.compile(r"\s*[–—]\s*")

# Heuristic: a run of 4+ consecutive non-ASCII letters, unlikely to be an
# English word, flagged for the caller rather than mangled by TTS.
_FOREIGN_PHRASE_RE = re.compile(r"[^\x00-\x7F]{4,}")

_TABLE_MARKER_RE = re.compile(r"\|.*\|")


def strip_html(text: str) -> str:
    return html.unescape(_TAG_RE.sub("", text))


def remove_footnote_markers(text: str) -> str:
    return _FOOTNOTE_MARKER_RE.sub("", text)


def fix_hyphenated_linebreaks(text: str) -> str:
    return _HYPHEN_LINEBREAK_RE.sub(r"\1\2", text)


_SPACE_BEFORE_PUNCT_RE = re.compile(r"\s+([.,!?;:])")


def fix_spacing_before_punctuation(text: str) -> str:
    return _SPACE_BEFORE_PUNCT_RE.sub(r"\1", text)


def normalize_whitespace(text: str) -> str:
    text = _TRAILING_SPACE_RE.sub("\n", text)
    text = _MULTI_SPACE_RE.sub(" ", text)
    text = _MULTI_NEWLINE_RE.sub("\n\n", text)
    return text.strip()


def normalize_quotes_and_dashes(text: str) -> str:
    text = text.translate(_CURLY_QUOTES)
    # Keep a real em-dash (spaced, so it doesn't glue neighboring words) rather
    # than " -- " — Kokoro's phonemizer reads a literal double-hyphen as words
    # instead of pausing on it, which was killing the natural mid-sentence
    # beat a dash is supposed to give the narration.
    text = _EM_DASH_RE.sub(" — ", text)
    return text


def expand_abbreviations(text: str) -> str:
    return _ABBREVIATION_RE.sub(lambda m: _ABBREVIATIONS[m.group(1)], text)


def normalize_numbers(text: str) -> str:
    def currency_repl(m: re.Match) -> str:
        symbol, amount = m.group(1), m.group(2).replace(",", "")
        unit = {"$": "dollars", "£": "pounds", "€": "euros"}[symbol]
        value = float(amount) if "." in amount else int(amount)
        return f"{num2words(value)} {unit}"

    def decimal_repl(m: re.Match) -> str:
        return f"{num2words(int(m.group(1)))} point {' '.join(num2words(int(d)) for d in m.group(2))}"

    def ordinal_repl(m: re.Match) -> str:
        return num2words(int(m.group(1)), to="ordinal")

    def year_repl(m: re.Match) -> str:
        return num2words(int(m.group(1)), to="year")

    def plain_number_repl(m: re.Match) -> str:
        return num2words(int(m.group(0)))

    text = _CURRENCY_RE.sub(currency_repl, text)
    text = _DECIMAL_RE.sub(decimal_repl, text)
    text = _ORDINAL_RE.sub(ordinal_repl, text)
    text = _YEAR_RANGE_RE.sub(year_repl, text)
    text = _PLAIN_NUMBER_RE.sub(plain_number_repl, text)
    return text


def flag_tables_and_lists(text: str) -> str:
    """Replace markdown-style table rows with a spoken placeholder."""
    if _TABLE_MARKER_RE.search(text):
        lines = text.splitlines()
        out = []
        for line in lines:
            if _TABLE_MARKER_RE.search(line):
                if not out or out[-1] != "[table content skipped]":
                    out.append("[table content skipped]")
            else:
                out.append(line)
        text = "\n".join(out)
    return text


DEFAULT_RULES = [
    strip_html,
    remove_footnote_markers,
    fix_hyphenated_linebreaks,
    normalize_quotes_and_dashes,
    flag_tables_and_lists,
    normalize_whitespace,
    expand_abbreviations,
    normalize_numbers,
    fix_spacing_before_punctuation,
]


@dataclass
class CleanResult:
    text: str
    foreign_phrases: list[str] = field(default_factory=list)


def detect_foreign_phrases(text: str) -> list[str]:
    return _FOREIGN_PHRASE_RE.findall(text)


def clean_text(text: str, rules: list = None) -> CleanResult:
    rules = DEFAULT_RULES if rules is None else rules
    foreign_phrases = detect_foreign_phrases(text)
    for rule in rules:
        text = rule(text)
    return CleanResult(text=text, foreign_phrases=foreign_phrases)
