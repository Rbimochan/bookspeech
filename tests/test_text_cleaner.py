from app.text_cleaner import clean_text, detect_foreign_phrases


def test_strip_html_and_entities():
    result = clean_text("<p>Hello &amp; welcome</p>")
    assert result.text == "Hello & welcome"


def test_remove_footnote_markers():
    result = clean_text("This is a claim[12] worth noting.")
    assert "[12]" not in result.text
    assert "This is a claim worth noting." in result.text


def test_fix_hyphenated_linebreak():
    result = clean_text("This is an exam-\nple of a split word.")
    assert "example" in result.text
    assert "exam-\nple" not in result.text


def test_collapse_whitespace():
    result = clean_text("Paragraph one.\n\n\n\n\nParagraph two.    With  spaces.")
    assert "\n\n\n" not in result.text
    assert "  " not in result.text.replace("\n\n", "")


def test_number_normalization_cardinal():
    result = clean_text("I have 42 apples.")
    assert "forty-two" in result.text


def test_number_normalization_currency():
    result = clean_text("It costs $19.")
    assert "nineteen dollars" in result.text


def test_number_normalization_ordinal():
    result = clean_text("This is his 3rd attempt.")
    assert "third" in result.text.lower()


def test_number_normalization_year():
    result = clean_text("It happened in 1999.")
    assert "nineteen" in result.text.lower()


def test_abbreviation_expansion():
    result = clean_text("Dr. Smith met Mr. Jones, e.g. at noon.")
    assert "Doctor Smith" in result.text
    assert "Mister Jones" in result.text
    assert "for example" in result.text


def test_curly_quotes_and_em_dash():
    result = clean_text("She said “hello” — then left.")
    assert '"hello"' in result.text
    # A real em-dash is kept (spaced, not glued to neighbors) rather than
    # collapsed to " -- ", since the TTS engine actually pauses on "—" but
    # reads a literal double-hyphen as words.
    assert " — " in result.text
    assert "--" not in result.text


def test_nested_quotes_preserved():
    result = clean_text("He said, “She told me ‘no’ yesterday.”")
    assert result.text.count('"') == 2
    assert "'no'" in result.text


def test_foreign_phrase_detection():
    phrases = detect_foreign_phrases("The phrase c'est la vie is French, but 早稲田大学 is Japanese.")
    assert any("早稲田" in p for p in phrases)


def test_table_flagged_as_skipped():
    result = clean_text("Intro text.\n| a | b |\n| 1 | 2 |\nOutro text.")
    assert "[table content skipped]" in result.text
    assert "| a | b |" not in result.text


def test_pipeline_is_idempotent_on_clean_text():
    once = clean_text("A simple clean sentence with no issues.").text
    twice = clean_text(once).text
    assert once == twice


def test_custom_rules_subset():
    from app.text_cleaner import strip_html, normalize_whitespace

    result = clean_text("<b>Hi</b>   there", rules=[strip_html, normalize_whitespace])
    assert result.text == "Hi there"
