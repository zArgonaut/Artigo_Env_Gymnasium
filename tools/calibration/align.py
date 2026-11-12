"""Alignment of DOM token elements with our tokenizer's tokens (SDD 03 §7.2, REQ-CAL-002).

A misalignment is a tokenization divergence to be recorded, never an exception.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from artigo_env.text.tokenizer import Token, tokenize
from tools.calibration.dom import DomToken

WINDOW = 5
AGREE = 3
_FORM_PREVIEW = 12


@dataclass(frozen=True)
class Divergence:
    kind: Literal["count", "length", "form", "header"]
    dom_index: int | None
    token_index: int | None
    detail: str


@dataclass(frozen=True)
class Alignment:
    pairs: tuple[tuple[int, int], ...]
    divergences: tuple[Divergence, ...]


def article_tokens(title: str, text: str) -> list[Token]:
    """Title tokens first, then body tokens (unified positions, SDD 07 §1)."""
    return [*tokenize(title), *tokenize(text)]


def _preview(value: str | None) -> str:
    value = value or ""
    return repr(value if len(value) <= _FORM_PREVIEW else value[:_FORM_PREVIEW] + "...")


def _mismatch(
    dom: DomToken, token: Token
) -> tuple[Literal["length", "form", "header"], str] | None:
    if dom.length != len(token.text):
        return "length", f"dom length {dom.length}, token length {len(token.text)}"
    if dom.revealed and dom.text is not None and dom.text != token.text:
        return "form", f"dom form {_preview(dom.text)}, token form {_preview(token.text)}"
    if dom.header != token.header:
        return "header", f"dom header {dom.header}, token header {token.header}"
    return None


def _resync(
    dom: Sequence[DomToken], tokens: Sequence[Token], i: int, j: int
) -> tuple[int, int] | None:
    """Smallest skip (a, b) after which up to 3 consecutive lengths agree."""
    n, m = len(dom), len(tokens)
    for total in range(1, 2 * WINDOW + 1):
        for a in range(max(0, total - WINDOW), min(total, WINDOW) + 1):
            b = total - a
            di, tj = i + a, j + b
            if di > n or tj > m:
                continue
            checked = min(AGREE, n - di, m - tj)
            if checked == 0:
                if di == n and tj == m:
                    return a, b
                continue
            if all(dom[di + k].length == len(tokens[tj + k].text) for k in range(checked)):
                return a, b
    return None


def align(dom: Sequence[DomToken], tokens: Sequence[Token]) -> Alignment:
    """Pair DOM elements with tokens in order, recording every divergence found."""
    pairs: list[tuple[int, int]] = []
    divergences: list[Divergence] = []
    i = j = 0
    while i < len(dom) and j < len(tokens):
        problem = _mismatch(dom[i], tokens[j])
        if problem is None:
            pairs.append((i, j))
            i, j = i + 1, j + 1
            continue
        kind, detail = problem
        if kind != "length":
            divergences.append(Divergence(kind, dom[i].index, j, detail))
            pairs.append((i, j))
            i, j = i + 1, j + 1
            continue
        skip = _resync(dom, tokens, i, j)
        if skip is None:
            divergences.append(Divergence("length", dom[i].index, j, detail + "; no resync"))
            i, j = i + 1, j + 1
        elif skip[0] == skip[1]:
            divergences.append(Divergence("length", dom[i].index, j, detail))
            i, j = i + skip[0], j + skip[1]
        else:
            note = f"{detail}; skipped {skip[0]} dom element(s) and {skip[1]} token(s)"
            divergences.append(Divergence("count", dom[i].index, j, note))
            i, j = i + skip[0], j + skip[1]
    if i < len(dom) or j < len(tokens):
        detail = f"{len(dom) - i} dom element(s) and {len(tokens) - j} token(s) left unpaired"
        divergences.append(
            Divergence(
                "count",
                dom[i].index if i < len(dom) else None,
                j if j < len(tokens) else None,
                detail,
            )
        )
    return Alignment(tuple(pairs), tuple(divergences))


def revealed_indices(before: Sequence[DomToken], after: Sequence[DomToken]) -> list[int]:
    """Indices revealed in ``after`` that were not revealed in ``before``."""
    already = {token.index for token in before if token.revealed}
    return [token.index for token in after if token.revealed and token.index not in already]
