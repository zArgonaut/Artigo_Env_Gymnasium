"""The game engine: pure rules of one match (SDD 07, DEC-055).

No randomness, no Gymnasium. ``apply`` never mutates its input state.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from artigo_env.core.errors import InvalidActionError
from artigo_env.engine.lexicon import Lexicon
from artigo_env.engine.rulesets.components import (
    HINT_RULES,
    LENGTH_RULES,
    NUMBER_RULES,
    TITLE_RULES,
)
from artigo_env.engine.rulesets.ruleset import Ruleset
from artigo_env.engine.screen import ScreenArrays, ScreenToken, ScreenView, render_partial
from artigo_env.engine.serialize import deserialize_state, serialize_state
from artigo_env.engine.types import (
    FLAG_HEADER,
    FLAG_INSIDE_PARENTHESES,
    FLAG_PARAGRAPH_BREAK,
    Article,
    GameState,
    Guess,
    GuessRecord,
    InvalidGuess,
    LexicalGuess,
    NumericGuess,
    Outcome,
    OutcomeKind,
    StopwordGuess,
    TitleProgress,
    freeze,
)
from artigo_env.text.guess import GuessKind, process_guess
from artigo_env.text.lemmatizers.registry import get_lemmatizer
from artigo_env.text.stopwords import load_stopwords

_VISIBLE_FLAGS = FLAG_HEADER | FLAG_PARAGRAPH_BREAK | FLAG_INSIDE_PARENTHESES
_EMPTY = np.zeros(0, dtype=np.int64)


class Engine:
    """Applies guesses to game states under a ruleset."""

    def __init__(self, ruleset: Ruleset, lexicon: Lexicon) -> None:
        self.ruleset = ruleset
        self.lexicon = lexicon
        self._stopwords = load_stopwords(ruleset.stopwords)
        self._lemmatizer = get_lemmatizer(ruleset.lemmatizer)
        self._numbers = NUMBER_RULES.get(ruleset.numbers)
        self._title_rule = TITLE_RULES.get(ruleset.title_parenthetical)
        self._lengths = LENGTH_RULES.get(ruleset.lengths_visible)
        self._hint = HINT_RULES.get(ruleset.hint.kind)

    # -- game lifecycle -------------------------------------------------------------------

    def new_game(self, article: Article) -> GameState:
        """Initial state: stopwords revealed, nothing guessed."""
        return GameState(
            revealed=article.stop.copy(),
            partial={},
            guessed=frozenset(),
            history=(),
            n_counted=0,
            status="playing",
        )

    # -- guesses --------------------------------------------------------------------------

    def parse_guess(self, raw: str) -> Guess:
        processed = process_guess(
            raw,
            stopwords=self._stopwords,
            separator=self.ruleset.guess_separator,
            numbers_enabled=self._numbers.enabled,
        )
        if processed.kind is GuessKind.INVALID:
            return InvalidGuess(raw, processed.reason or "invalid")
        if processed.kind is GuessKind.NUMERIC:
            return NumericGuess(processed.form)
        if processed.kind is GuessKind.STOPWORD:
            return StopwordGuess(processed.form)
        lemma = self._lemmatizer.lemma(processed.form)
        return LexicalGuess(self.lexicon.lemma_id(lemma), lemma, processed.form)

    def guess_from_lemma(self, lemma_id: int) -> Guess:
        text = self.lexicon.lemma_text(lemma_id)
        return LexicalGuess(lemma_id, text, text)

    def apply(self, state: GameState, article: Article, guess: Guess) -> tuple[GameState, Outcome]:
        """Apply ``guess``; returns a new state and the outcome (``state`` is untouched)."""
        if state.status == "won":
            raise InvalidActionError("the game was already won")
        required = self._required(article)
        if isinstance(guess, InvalidGuess):
            progress = self._progress(article, state.revealed, state.partial, required)
            return state, Outcome(OutcomeKind.INVALID, False, 0, _EMPTY, {}, progress, False)

        revealed = state.revealed.copy()
        partial: dict[int, int] = dict(state.partial)
        newly_partial: dict[int, int] = {}
        newly: list[int] = []
        guessed = state.guessed
        count = 0

        form = guess.digits if isinstance(guess, NumericGuess) else guess.form
        if isinstance(guess, StopwordGuess):
            key = f"s:{guess.form}"
            kind = OutcomeKind.NOOP_STOPWORD
            counted = self.ruleset.noop_cost == 1
        else:
            if isinstance(guess, NumericGuess):
                key = f"n:{guess.digits}"
            elif guess.lemma_id >= 0:
                key = f"l:{guess.lemma_id}"
            else:
                key = f"u:{guess.lemma}"
            if key in state.guessed:
                kind = OutcomeKind.NOOP_REPEAT
                counted = self.ruleset.noop_cost == 1
            else:
                guessed = state.guessed | {key}
                counted = True
                if isinstance(guess, NumericGuess):
                    count = self._reveal_numeric(
                        article, guess.digits, revealed, partial, newly_partial, newly
                    )
                else:
                    positions = article.lemma_index.get(guess.lemma_id)
                    if positions is not None:
                        count = len(positions)
                        fresh = positions[~revealed[positions]]
                        revealed[fresh] = True
                        newly.extend(int(p) for p in fresh)
                kind = OutcomeKind.HIT if count > 0 else OutcomeKind.MISS

        n_counted = state.n_counted + (1 if counted else 0)
        if counted:
            self._apply_hint(article, n_counted, revealed, partial, newly_partial, newly)

        won = bool(revealed[: article.n_title][required].all())
        freeze(revealed)
        new_state = GameState(
            revealed=revealed,
            partial=partial,
            guessed=guessed,
            history=(*state.history, GuessRecord(key, form, kind.value, counted, count)),
            n_counted=n_counted,
            status="won" if won else "playing",
        )
        newly_array = np.array(sorted(newly), dtype=np.int64)
        progress = self._progress(article, revealed, partial, required)
        return new_state, Outcome(kind, counted, count, newly_array, newly_partial, progress, won)

    def _reveal_numeric(
        self,
        article: Article,
        digits: str,
        revealed: npt.NDArray[np.bool_],
        partial: dict[int, int],
        newly_partial: dict[int, int],
        newly: list[int],
    ) -> int:
        count = 0
        for raw_pos in article.digit_positions:
            pos = int(raw_pos)
            surface = article.surface_at(pos)
            mask = self._numbers.mask(surface, digits)
            if not mask:
                continue
            count += 1
            if not revealed[pos]:
                self._add_bits(pos, mask, len(surface), revealed, partial, newly_partial, newly)
        return count

    @staticmethod
    def _add_bits(
        pos: int,
        mask: int,
        length: int,
        revealed: npt.NDArray[np.bool_],
        partial: dict[int, int],
        newly_partial: dict[int, int],
        newly: list[int],
    ) -> None:
        old = partial.get(pos, 0)
        gain = mask & ~old
        if gain:
            partial[pos] = old | mask
            newly_partial[pos] = newly_partial.get(pos, 0) | gain
        if partial.get(pos, 0) == (1 << length) - 1:
            revealed[pos] = True
            newly.append(pos)

    def _apply_hint(
        self,
        article: Article,
        n_counted: int,
        revealed: npt.NDArray[np.bool_],
        partial: dict[int, int],
        newly_partial: dict[int, int],
        newly: list[int],
    ) -> None:
        for pos, mask in self._hint.reveals(article, n_counted, self.ruleset.hint).items():
            if not revealed[pos]:
                length = len(article.surface_at(pos))
                self._add_bits(pos, mask, length, revealed, partial, newly_partial, newly)

    # -- progress and screen --------------------------------------------------------------

    def _required(self, article: Article) -> npt.NDArray[np.bool_]:
        return self._title_rule.required_mask(article.title)

    @staticmethod
    def _progress(
        article: Article,
        revealed: npt.NDArray[np.bool_],
        partial: Any,
        required: npt.NDArray[np.bool_],
    ) -> TitleProgress:
        tokens_total = letters_total = tokens_revealed = letters_revealed = 0
        for pos in np.flatnonzero(required):
            p = int(pos)
            length = int(article.lengths[p])
            tokens_total += 1
            letters_total += length
            if revealed[p]:
                tokens_revealed += 1
                letters_revealed += length
            else:
                letters_revealed += int(partial.get(p, 0)).bit_count()
        return TitleProgress(tokens_revealed, tokens_total, letters_revealed, letters_total)

    def screen(self, state: GameState, article: Article) -> ScreenView:
        """The player's view: nothing about hidden tokens beyond what the ruleset shows."""
        n_title = article.n_title

        def token(pos: int) -> ScreenToken:
            flags = int(article.flags[pos])
            if state.revealed[pos]:
                return ScreenToken(article.surface_at(pos), None, flags, None)
            show = self._lengths.visible(pos < n_title)
            mask = state.partial.get(pos, 0)
            partial = render_partial(article.surface_at(pos), mask, show) if mask else None
            length = int(article.lengths[pos]) if show and not mask else None
            return ScreenToken(None, length, flags & _VISIBLE_FLAGS, partial)

        views = [token(pos) for pos in range(article.n_total)]
        return ScreenView(
            tuple(views[:n_title]),
            tuple(views[n_title:]),
            state.history,
            state.n_counted,
            state.status,
        )

    def screen_arrays(self, state: GameState, article: Article) -> ScreenArrays:
        """``screen`` as arrays: same visibility rules, no per-token objects."""
        n_title = article.n_title
        visible = state.revealed
        shown = np.empty(article.n_total, dtype=np.bool_)
        shown[:n_title] = self._lengths.visible(True)
        shown[n_title:] = self._lengths.visible(False)
        length = np.where(visible | shown, article.lengths, 0).astype(np.uint16)
        flags = np.where(visible, article.flags, article.flags & _VISIBLE_FLAGS).astype(np.uint8)
        partial = {pos: mask for pos, mask in state.partial.items() if not visible[pos]}
        return ScreenArrays(visible, length, flags, partial, n_title)

    # -- serialization --------------------------------------------------------------------

    @staticmethod
    def serialize(state: GameState) -> dict[str, Any]:
        return serialize_state(state)

    @staticmethod
    def deserialize(data: dict[str, Any]) -> GameState:
        return deserialize_state(data)
