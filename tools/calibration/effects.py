"""Token effects of each guess, replayed from the DOM mutation stream.

The recorder reports every mutation of the game container in page order, so the token state
after each guess can be rebuilt without depending on when a snapshot was taken. A mutation of
a token element has the token id as target (``#token-text-N`` / ``#token-title-N``); a change
of its text arrives as a ``childList`` mutation with ``#text:<shown text>`` entries (or as
``characterData``), and the ``guess-match`` class marks the tokens matching the latest counted
guess.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from tools.calibration.dom import DomToken
from tools.recorder import selectors

_TARGET = re.compile(
    rf"^(?:{re.escape(selectors.TOKEN_ID_TITLE_PREFIX)}(?P<title>\d+)"
    rf"|{re.escape(selectors.TOKEN_ID_TEXT_PREFIX)}(?P<text>\d+))$"
)
_TEXT_PREFIX = "#text:"


@dataclass(frozen=True)
class TokenEffects:
    """What one guess did to the tokens (unified indices, see ``DomToken.index``)."""

    revealed: tuple[int, ...]
    partial: tuple[int, ...]
    matched: tuple[int, ...]
    touched: int  # token mutations seen between this submit and the next


def _mask(text: str) -> int:
    return sum(1 << i for i, char in enumerate(text) if char != selectors.HIDDEN_CHAR)


def _target(target: str) -> tuple[bool, int] | None:
    """``(in_title, offset)`` of a token element selector, else ``None``."""
    found = _TARGET.match(target[1:]) if target.startswith("#") else None
    if found is None:
        return None
    in_title = found.group("title") is not None
    return in_title, int(found.group("title" if in_title else "text"))


def replay_effects(
    initial: Sequence[DomToken], events: Sequence[dict[str, Any]], n_guesses: int
) -> list[TokenEffects]:
    """Replay ``events`` from the token state of ``initial`` (the load snapshot).

    The n-th ``input.submit`` opens the n-th guess; its effects are the state change up to the
    next submit. Returns one entry per guess (at most ``n_guesses``).
    """
    where = {(t.in_title, t.offset): t.index for t in initial if t.offset is not None}
    length = {t.index: t.length for t in initial}
    mask = {t.index: t.mask for t in initial}
    matched = {t.index for t in initial if t.matched}

    def full(index: int) -> bool:
        return mask[index] == (1 << length[index]) - 1

    def apply(payload: dict[str, Any]) -> int | None:
        key = _target(str(payload.get("target", "")))
        index = where.get(key) if key is not None else None
        if index is None:
            return None
        kind = payload.get("kind")
        if kind == "attributes" and payload.get("attribute") == "class":
            classes = str(payload.get("new") or "").split()
            (matched.add if selectors.TOKEN_MATCH_CLASS in classes else matched.discard)(index)
        elif kind == "characterData":
            mask[index] = _mask(str(payload.get("new") or ""))
        elif kind == "childList":
            added = [
                str(item)[len(_TEXT_PREFIX) :]
                for item in payload.get("added") or []
                if str(item).startswith(_TEXT_PREFIX)
            ]
            if added:
                mask[index] = _mask("".join(added))
        return index

    results: list[TokenEffects] = []
    start_mask: dict[int, int] = {}
    touched = 0
    open_guess = False

    def close() -> None:
        changed = [i for i, m in mask.items() if m != start_mask.get(i, m)]
        revealed = [i for i in changed if full(i) and start_mask[i] != (1 << length[i]) - 1]
        partial = [i for i in changed if not full(i) and mask[i] & ~start_mask[i]]
        results.append(
            TokenEffects(
                tuple(sorted(revealed)), tuple(sorted(partial)), tuple(sorted(matched)), touched
            )
        )

    for event in events:
        kind = event.get("type")
        if kind == "input.submit":
            if open_guess:
                close()
            if len(results) >= n_guesses:
                break
            start_mask = dict(mask)
            touched = 0
            open_guess = True
        elif kind == "dom.mutation":
            hit = apply(event.get("payload", {}) or {})
            if hit is not None and open_guess:
                touched += 1
    else:
        if open_guess:
            close()
    return results
