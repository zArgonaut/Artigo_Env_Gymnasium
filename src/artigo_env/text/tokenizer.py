"""Tokenizer that reproduces the game's rule (SDD 05 §2, DEC-037).

A token is a maximal run of Unicode letters and ASCII digits. Everything else (hyphen,
apostrophe, punctuation, whitespace, combining marks, underscore, non-ASCII digits) separates
tokens and is never hidden.
"""

from __future__ import annotations

from dataclasses import dataclass

import regex

from artigo_env.text.normalize import normalize

# ``\p{L}`` includes the ordinal indicators "º" and "ª" (REQ-TXT-005). The ``re`` module's
# ``\w`` must not be used: it also accepts "_" and non-ASCII digits (REQ-TXT-004).
_TOKEN = regex.compile(r"[\p{L}0-9]+")

# With ``hyphen="join"`` an ASCII hyphen directly between two letters does not split a token
# (DEC-043); chains such as "sul-rio-grandense" join fully, digits never join across a hyphen.
_TOKEN_JOIN = regex.compile(r"[\p{L}0-9]+(?:(?<=\p{L})-(?=\p{L})[\p{L}0-9]+)*")

# Section header as emitted by TextExtracts: "== Title ==", any level from 2 up, alone on a line.
_HEADER = regex.compile(r"^(={2,})[^\S\n]*\S.*?[^\S\n]*\1[^\S\n]*$")


@dataclass(frozen=True, slots=True)
class Token:
    """A token with its original spelling and its offsets in the normalized text."""

    text: str
    start: int
    end: int
    header: bool = False


def is_header_line(line: str) -> bool:
    """Whether ``line`` (without its newline) is a section header (REQ-TXT-007)."""
    return _HEADER.match(line.rstrip("\r")) is not None


def tokenize(text: str, hyphen: str = "split") -> list[Token]:
    """Tokenize ``text``; offsets refer to ``normalize(text)`` (REQ-TXT-003, REQ-TXT-017).

    ``hyphen`` is ``"split"`` (default, the game's rule) or ``"join"`` (letter-hyphen-letter is
    a single token); any other value raises ``ValueError``.
    """
    if hyphen == "split":
        pattern = _TOKEN
    elif hyphen == "join":
        pattern = _TOKEN_JOIN
    else:
        raise ValueError(f"unknown hyphen mode: {hyphen!r} (expected 'split' or 'join')")
    text = normalize(text)
    tokens: list[Token] = []
    line_start = 0
    for line in text.split("\n"):
        header = is_header_line(line)
        for match in pattern.finditer(line):
            tokens.append(
                Token(match.group(), line_start + match.start(), line_start + match.end(), header)
            )
        line_start += len(line) + 1
    return tokens
