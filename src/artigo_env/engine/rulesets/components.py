"""Registered ruleset levers (SDD 08 §A.1, REQ-RUL-003).

Each lever is a ``Registry`` of small components with a minimal protocol, so a new variant can be
added without touching the engine.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import numpy as np
import numpy.typing as npt

from artigo_env.core.config import ConfigBase
from artigo_env.core.registry import Registry
from artigo_env.engine.types import FLAG_INSIDE_PARENTHESES, TokenSeq
from artigo_env.text.stopwords import STOPWORD_KINDS as _STOPWORD_KIND_NAMES

if TYPE_CHECKING:
    from artigo_env.engine.types import Article


@dataclass(frozen=True)
class HintSpec(ConfigBase):
    """Hint lever configuration: ``kind`` names a ``HINT_RULES`` entry."""

    kind: str = "none"
    after: int = 0

    def __post_init__(self) -> None:
        if isinstance(self.after, bool) or not isinstance(self.after, int) or self.after < 0:
            raise ValueError(f"hint.after must be a non-negative integer, got {self.after!r}")


# --- numbers -------------------------------------------------------------------------------


class NumberRule(Protocol):
    enabled: bool

    def mask(self, surface: str, digits: str) -> int:
        """Bitmask of the characters of ``surface`` revealed by the numeric guess ``digits``."""
        ...


class SubstringNumbers:
    """R6: the digits are searched as a substring; occurrences are non-overlapping, scanned
    from the left (JavaScript ``matchAll`` semantics): ``11`` on ``111`` marks characters 0-1.

    Read from the game client (DEC-108); kept isolated here so it can be changed in one place.
    """

    enabled = True

    def mask(self, surface: str, digits: str) -> int:
        result = 0
        if not digits:
            return result
        width = len(digits)
        start = surface.find(digits)
        while start != -1:
            result |= ((1 << width) - 1) << start
            start = surface.find(digits, start + width)
        return result


class WholeTokenNumbers:
    """Only a token that is exactly the guessed number is revealed."""

    enabled = True

    def mask(self, surface: str, digits: str) -> int:
        return (1 << len(surface)) - 1 if surface == digits else 0


class DisabledNumbers:
    """Digits are ordinary word characters: digit-only guesses are lexical."""

    enabled = False

    def mask(self, surface: str, digits: str) -> int:
        return 0


NUMBER_RULES: Registry[NumberRule] = Registry("number rule")
NUMBER_RULES.register("substring", SubstringNumbers())
NUMBER_RULES.register("whole-token", WholeTokenNumbers())
NUMBER_RULES.register("disabled", DisabledNumbers())


# --- title victory -------------------------------------------------------------------------


class TitleRule(Protocol):
    def required_mask(self, title: TokenSeq) -> npt.NDArray[np.bool_]:
        """Which title tokens must be fully revealed to win."""
        ...


class RequiredTitle:
    def required_mask(self, title: TokenSeq) -> npt.NDArray[np.bool_]:
        required: npt.NDArray[np.bool_] = np.logical_not(title.stop)
        return required


class IgnoredParentheticalTitle:
    def required_mask(self, title: TokenSeq) -> npt.NDArray[np.bool_]:
        inside = (title.flags & FLAG_INSIDE_PARENTHESES) != 0
        required: npt.NDArray[np.bool_] = np.logical_not(np.logical_or(title.stop, inside))
        return required


TITLE_RULES: Registry[TitleRule] = Registry("title rule")
TITLE_RULES.register("required", RequiredTitle())
TITLE_RULES.register("ignored", IgnoredParentheticalTitle())


# --- lengths -------------------------------------------------------------------------------


class LengthRule(Protocol):
    def visible(self, in_title: bool) -> bool: ...


@dataclass(frozen=True)
class _Lengths:
    title: bool
    body: bool

    def visible(self, in_title: bool) -> bool:
        return self.title if in_title else self.body


LENGTH_RULES: Registry[LengthRule] = Registry("length rule")
LENGTH_RULES.register("all", _Lengths(True, True))
LENGTH_RULES.register("title_only", _Lengths(True, False))
LENGTH_RULES.register("none", _Lengths(False, False))


# --- hints ---------------------------------------------------------------------------------


class HintRule(Protocol):
    def reveals(self, article: Article, n_counted: int, spec: HintSpec) -> Mapping[int, int]:
        """Position -> character bits to reveal now (the engine skips revealed positions)."""
        ...


class NoHint:
    def reveals(self, article: Article, n_counted: int, spec: HintSpec) -> Mapping[int, int]:
        return {}


class FirstLetterHint:
    """After ``spec.after`` counted guesses, reveal the first character of every title token."""

    def reveals(self, article: Article, n_counted: int, spec: HintSpec) -> Mapping[int, int]:
        if n_counted < spec.after:
            return {}
        stop = article.title.stop
        return {pos: 1 for pos in range(article.n_title) if not stop[pos]}


HINT_RULES: Registry[HintRule] = Registry("hint rule")
HINT_RULES.register("none", NoHint())
HINT_RULES.register("first-letter", FirstLetterHint())


# --- named modes ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NamedMode:
    """A lever whose behaviour is implemented by the text layer under this ``mode`` name."""

    mode: str


SEPARATOR_RULES: Registry[NamedMode] = Registry("separator rule")
SEPARATOR_RULES.register("strip", NamedMode("strip"))
SEPARATOR_RULES.register("reject", NamedMode("reject"))

HYPHEN_RULES: Registry[NamedMode] = Registry("hyphen rule")
HYPHEN_RULES.register("split", NamedMode("split"))
HYPHEN_RULES.register("join", NamedMode("join"))

STOPWORD_KINDS: Registry[NamedMode] = Registry("stopword kind")
for _kind in _STOPWORD_KIND_NAMES:
    STOPWORD_KINDS.register(_kind, NamedMode(_kind))
