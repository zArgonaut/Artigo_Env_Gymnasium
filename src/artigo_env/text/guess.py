"""Guess processing (SDD 05 §3, DEC-038, REQ-TXT-008, REQ-TXT-009).

Order: lowercase, keep only letters and ASCII digits, then classify. Lemmatization of lexical
guesses is left to the caller.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import regex

from artigo_env.text.normalize import lower
from artigo_env.text.stopwords import StopwordSet

_FOREIGN = regex.compile(r"[^\p{L}0-9]")
_DIGITS = regex.compile(r"[0-9]+")


class GuessKind(Enum):
    LEXICAL = "lexical"
    NUMERIC = "numeric"
    STOPWORD = "stopword"
    INVALID = "invalid"


@dataclass(frozen=True, slots=True)
class ProcessedGuess:
    """A classified guess; ``form`` is the cleaned lowercase text."""

    kind: GuessKind
    raw: str
    form: str
    reason: str | None = None


def clean_guess(raw: str) -> str:
    """Lowercase ``raw`` and remove every non-letter, non-ASCII-digit character (not a split)."""
    return _FOREIGN.sub("", lower(raw))


def process_guess(
    raw: str,
    *,
    stopwords: StopwordSet,
    separator: str = "strip",
    numbers_enabled: bool = True,
) -> ProcessedGuess:
    """Classify ``raw`` following REQ-TXT-008.

    With ``separator == "reject"`` a guess is invalid when cleaning removes any character other
    than whitespace at its ends (REQ-TXT-009).
    """
    if separator not in ("strip", "reject"):
        raise ValueError(f"unknown guess separator mode: {separator!r}")
    lowered = lower(raw)
    form = _FOREIGN.sub("", lowered)
    if not form:
        return ProcessedGuess(GuessKind.INVALID, raw, form, "empty")
    if separator == "reject" and _FOREIGN.search(lowered.strip()) is not None:
        return ProcessedGuess(GuessKind.INVALID, raw, form, "separator")
    if numbers_enabled and _DIGITS.fullmatch(form) is not None:
        return ProcessedGuess(GuessKind.NUMERIC, raw, form)
    if form in stopwords:
        return ProcessedGuess(GuessKind.STOPWORD, raw, form)
    return ProcessedGuess(GuessKind.LEXICAL, raw, form)
