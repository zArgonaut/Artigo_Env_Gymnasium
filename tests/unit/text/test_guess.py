import pytest

from artigo_env.text.guess import GuessKind, ProcessedGuess, clean_guess, process_guess
from artigo_env.text.stopwords import StopwordSpec, load_stopwords

pytestmark = pytest.mark.fast

STOP = load_stopwords(StopwordSpec())


def go(raw: str, **kwargs) -> ProcessedGuess:
    return process_guess(raw, stopwords=STOP, **kwargs)


@pytest.mark.req("REQ-TXT-008")
def test_hyphen_is_removed_not_split():
    result = go("Guarda-Chuva")
    assert (result.kind, result.form) == (GuessKind.LEXICAL, "guardachuva")


@pytest.mark.req("REQ-TXT-008")
@pytest.mark.parametrize("raw", ["  ", "-", "!?", "", "٣"])
def test_empty_after_cleaning_is_invalid(raw):
    result = go(raw)
    assert (result.kind, result.reason) == (GuessKind.INVALID, "empty")


@pytest.mark.req("REQ-TXT-008")
def test_numeric_guess_and_hash_prefix():
    assert go("1500").kind is GuessKind.NUMERIC
    hashed = go("#1")
    assert (hashed.kind, hashed.form) == (GuessKind.NUMERIC, "1")
    assert go("f1").kind is GuessKind.LEXICAL


@pytest.mark.req("REQ-TXT-008")
def test_stopword_is_checked_on_lowercase_form():
    assert go("De").kind is GuessKind.STOPWORD
    assert go("historia").form != go("história").form
    assert go("ÁGUA").form == "água"


@pytest.mark.req("REQ-TXT-008")
def test_clean_guess():
    assert clean_guess("Guarda-Chuva!") == "guardachuva"
    assert clean_guess("a_b 1") == "ab1"


@pytest.mark.req("REQ-TXT-009")
def test_separator_reject():
    for raw in ("guarda-chuva", "#1", "a b"):
        result = go(raw, separator="reject")
        assert (result.kind, result.reason) == (GuessKind.INVALID, "separator")
    assert go(" casa ", separator="reject").kind is GuessKind.LEXICAL
    assert go("-", separator="reject").reason == "empty"


@pytest.mark.req("REQ-TXT-009")
def test_unknown_separator_mode():
    with pytest.raises(ValueError):
        go("casa", separator="split")


@pytest.mark.req("REQ-TXT-008")
def test_numbers_disabled_makes_digits_lexical():
    assert go("1500", numbers_enabled=False).kind is GuessKind.LEXICAL
