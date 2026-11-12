"""What a player sees (SDD 07 §3, REQ-ENG-014)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from artigo_env.engine.types import GuessRecord, Status

HIDDEN_CHAR = "█"


@dataclass(frozen=True, slots=True)
class ScreenToken:
    """One token as displayed.

    ``text`` is set only for fully revealed tokens. A hidden token has ``length`` (``None`` when
    the ruleset hides lengths) and, when numerically revealed in part, ``partial`` with ``█``
    for hidden characters (runs of hidden characters collapse to one ``█`` when lengths are
    hidden). ``flags`` keeps only what is visible on screen.
    """

    text: str | None
    length: int | None
    flags: int
    partial: str | None


@dataclass(frozen=True, slots=True)
class ScreenView:
    title: tuple[ScreenToken, ...]
    body: tuple[ScreenToken, ...]
    history: tuple[GuessRecord, ...]
    n_counted: int
    status: Status


@dataclass(frozen=True, eq=False)
class ScreenArrays:
    """The same information as ``ScreenView``, as arrays over unified positions.

    Meant for building numeric observations without one Python object per token. Nothing here
    goes beyond what ``ScreenView`` shows: ``length`` is 0 for a hidden token whose length the
    ruleset conceals, and ``flags`` keeps only layout bits for hidden tokens.
    """

    visible: npt.NDArray[np.bool_]  # fully revealed
    length: npt.NDArray[np.uint16]  # exposed length in characters (0 = concealed)
    flags: npt.NDArray[np.uint8]  # exposed engine flag bits
    partial: Mapping[int, int]  # position -> revealed-character bits, hidden tokens only
    n_title: int


def render_partial(surface: str, mask: int, show_length: bool) -> str:
    """Revealed characters of ``surface`` per ``mask``; hidden ones as ``█``."""
    out: list[str] = []
    for i, char in enumerate(surface):
        if (mask >> i) & 1:
            out.append(char)
        elif show_length or not out or out[-1] != HIDDEN_CHAR:
            out.append(HIDDEN_CHAR)
    return "".join(out)
