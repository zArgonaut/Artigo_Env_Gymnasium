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


def align_by_offset(dom: Sequence[DomToken], tokens: Sequence[Token], n_title: int) -> Alignment:
    """Pair DOM elements with tokens through the character offset in the element id.

    ``tokens`` is the unified list (title first, ``n_title`` title tokens). Every DOM element
    must carry an ``offset``. Elements or tokens without a partner are ``count`` divergences.
    """
    by_offset = {(j < n_title, token.start): j for j, token in enumerate(tokens)}
    pairs: list[tuple[int, int]] = []
    divergences: list[Divergence] = []
    used: set[int] = set()
    for position, element in enumerate(dom):
        j = by_offset.get((element.in_title, element.offset if element.offset is not None else -1))
        if j is None:
            where = "title" if element.in_title else "body"
            detail = f"no token at {where} offset {element.offset}"
            divergences.append(Divergence("count", element.index, None, detail))
            continue
        used.add(j)
        pairs.append((position, j))
        problem = _mismatch(element, tokens[j])
        if problem is not None:
            divergences.append(Divergence(problem[0], element.index, j, problem[1]))
    for j, token in enumerate(tokens):
        if j not in used:
            where = "title" if j < n_title else "body"
            detail = f"no dom element for {where} token at offset {token.start}"
            divergences.append(Divergence("count", None, j, detail))
    return Alignment(tuple(pairs), tuple(divergences))


def align(
    dom: Sequence[DomToken], tokens: Sequence[Token], n_title: int | None = None
) -> Alignment:
    """Pair DOM elements with tokens, recording every divergence found.

    With ``n_title`` and element offsets the pairing is exact (by offset); otherwise elements
    and tokens are paired in order with resynchronisation.
    """
    if n_title is not None and dom and all(element.offset is not None for element in dom):
        return align_by_offset(dom, tokens, n_title)
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
    """Indices fully revealed in ``after`` that were not fully revealed in ``before``."""
    already = {token.index for token in before if token.revealed}
    return [token.index for token in after if token.revealed and token.index not in already]


def partial_indices(before: Sequence[DomToken], after: Sequence[DomToken]) -> list[int]:
    """Indices that gained visible characters without becoming fully revealed."""
    old = {token.index: token.mask for token in before}
    return [
        token.index
        for token in after
        if not token.revealed and token.mask & ~old.get(token.index, 0)
    ]
