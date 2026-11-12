"""Engine types (SDD 07 §1-§2, DEC-056, DEC-057).

Positions are unified: ``0..T-1`` are title tokens, ``T..T+L-1`` are body tokens. Arrays stored
in these objects are read-only; ``Engine.apply`` always copies.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Literal

import numpy as np
import numpy.typing as npt

FLAG_HEADER = 1
FLAG_PARAGRAPH_BREAK = 2
FLAG_INSIDE_PARENTHESES = 4
FLAG_HAS_DIGIT = 8


def freeze(array: npt.NDArray[np.generic]) -> None:
    """Make ``array`` read-only so accidental in-place mutation raises."""
    array.flags.writeable = False


@dataclass(frozen=True, eq=False)
class TokenSeq:
    """A sequence of tokens in parallel arrays."""

    forms: npt.NDArray[np.uint32]
    lemmas: npt.NDArray[np.int64]  # -1: lemma not in the lexicon
    lengths: npt.NDArray[np.uint16]
    flags: npt.NDArray[np.uint8]
    stop: npt.NDArray[np.bool_]
    surface: tuple[str, ...]

    def __post_init__(self) -> None:
        sizes = {len(a) for a in (self.forms, self.lemmas, self.lengths, self.flags, self.stop)}
        sizes.add(len(self.surface))
        if len(sizes) != 1:
            raise ValueError("TokenSeq arrays must have the same length")
        for array in (self.forms, self.lemmas, self.lengths, self.flags, self.stop):
            freeze(array)

    def __len__(self) -> int:
        return len(self.surface)


@dataclass(frozen=True, eq=False)
class Article:
    """A playable article: title and body token sequences plus lookup indexes."""

    article_ref: str
    title: TokenSeq
    body: TokenSeq
    lemma_index: Mapping[int, npt.NDArray[np.int64]]
    digit_positions: npt.NDArray[np.int64]
    lemmas: npt.NDArray[np.int64] = field(init=False, repr=False)
    lengths: npt.NDArray[np.uint16] = field(init=False, repr=False)
    flags: npt.NDArray[np.uint8] = field(init=False, repr=False)
    stop: npt.NDArray[np.bool_] = field(init=False, repr=False)
    surfaces: tuple[str, ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        pairs: Sequence[tuple[str, npt.NDArray[np.generic]]] = [
            ("lemmas", np.concatenate([self.title.lemmas, self.body.lemmas])),
            ("lengths", np.concatenate([self.title.lengths, self.body.lengths])),
            ("flags", np.concatenate([self.title.flags, self.body.flags])),
            ("stop", np.concatenate([self.title.stop, self.body.stop])),
        ]
        for name, array in pairs:
            freeze(array)
            object.__setattr__(self, name, array)
        object.__setattr__(self, "surfaces", self.title.surface + self.body.surface)
        freeze(self.digit_positions)
        for positions in self.lemma_index.values():
            freeze(positions)

    @property
    def n_title(self) -> int:
        return len(self.title)

    @property
    def n_total(self) -> int:
        return len(self.title) + len(self.body)

    def surface_at(self, pos: int) -> str:
        return self.surfaces[pos]


class OutcomeKind(Enum):
    HIT = "hit"
    MISS = "miss"
    NOOP_STOPWORD = "noop_stopword"  # "não conta"
    NOOP_REPEAT = "noop_repeat"  # "já foi"
    INVALID = "invalid"


@dataclass(frozen=True, slots=True)
class GuessRecord:
    """One entry of the append-only history; ``kind`` is an ``OutcomeKind`` value."""

    key: str  # "l:<lemma_id>", "u:<lemma>", "n:<digits>" or "s:<form>" (stopword)
    form: str  # the guess as typed, after clean-up (what a player sees in the history)
    kind: str
    counted: bool
    count: int


Status = Literal["playing", "won"]


@dataclass(frozen=True, eq=False)
class GameState:
    """Mutable-looking game progress that is never mutated: ``apply`` returns a new one."""

    revealed: npt.NDArray[np.bool_]  # bool[T+L]; stopwords start True
    partial: Mapping[int, int]  # position -> bitmask of revealed characters
    guessed: frozenset[str]
    history: tuple[GuessRecord, ...]
    n_counted: int
    status: Status

    def __post_init__(self) -> None:
        freeze(self.revealed)
        object.__setattr__(self, "partial", MappingProxyType(dict(self.partial)))


@dataclass(frozen=True, slots=True)
class LexicalGuess:
    lemma_id: int  # -1 when the lemma is not in the lexicon
    lemma: str
    form: str


@dataclass(frozen=True, slots=True)
class NumericGuess:
    digits: str


@dataclass(frozen=True, slots=True)
class StopwordGuess:
    form: str


@dataclass(frozen=True, slots=True)
class InvalidGuess:
    raw: str
    reason: str


Guess = LexicalGuess | NumericGuess | StopwordGuess | InvalidGuess


@dataclass(frozen=True, slots=True)
class TitleProgress:
    tokens_revealed: int
    tokens_total: int
    letters_revealed: int
    letters_total: int

    @property
    def token_fraction(self) -> float:
        return self.tokens_revealed / self.tokens_total if self.tokens_total else 1.0

    @property
    def letter_fraction(self) -> float:
        return self.letters_revealed / self.letters_total if self.letters_total else 1.0


@dataclass(frozen=True, eq=False)
class Outcome:
    """Result of one guess.

    ``newly_revealed`` lists positions that became fully revealed. ``newly_partial`` maps a
    position to the character bits newly revealed by this guess (a position that became fully
    revealed through partial reveals appears in both).
    """

    kind: OutcomeKind
    counted: bool
    count: int
    newly_revealed: npt.NDArray[np.int64]
    newly_partial: Mapping[int, int]
    title_progress: TitleProgress
    won: bool
