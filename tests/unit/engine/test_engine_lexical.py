from __future__ import annotations

import re
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import artigo_env.engine as engine_package
from artigo_env.core.errors import InvalidActionError
from artigo_env.engine import (
    ArticleBuilder,
    Engine,
    InvalidGuess,
    LexicalGuess,
    NumericGuess,
    OutcomeKind,
    StopwordGuess,
)

pytestmark = pytest.mark.fast

TITLE = "Gato preto"
BODY = "O gato preto era dos gatos. Um rato foi visto em 1990 e em 1991, mas o gato dormia."


@pytest.fixture
def game(protocol_ruleset, make_game):
    return make_game(TITLE, BODY, protocol_ruleset)


def play(engine, article, state, *guesses):
    outcome = None
    for raw in guesses:
        state, outcome = engine.apply(state, article, engine.parse_guess(raw))
    return state, outcome


@pytest.mark.req("REQ-ENG-004")
def test_stopwords_start_revealed_and_nothing_else(game):
    _, article, state = game
    assert state.revealed.tolist() == article.stop.tolist()
    assert state.revealed.any() and not state.revealed.all()
    assert state.n_counted == 0 and state.status == "playing"
    assert state.history == () and state.guessed == frozenset() and dict(state.partial) == {}


@pytest.mark.req("REQ-ENG-006")
def test_count_adds_title_and_body(game):
    engine, article, state = game
    state, outcome = play(engine, article, state, "gato")
    assert outcome.kind is OutcomeKind.HIT
    assert outcome.count == 4  # title + gato, gatos, gato in the body
    assert outcome.newly_revealed[0] == 0
    assert len(outcome.newly_revealed) == 4
    assert outcome.title_progress.tokens_revealed == 1
    assert outcome.title_progress.letters_revealed == 4
    assert outcome.title_progress.letters_total == 9
    assert outcome.title_progress.letter_fraction == pytest.approx(4 / 9)


@pytest.mark.req("REQ-ENG-005")
def test_stopword_guess_does_not_count(game):
    engine, article, state = game
    state, outcome = play(engine, article, state, "Um")
    assert outcome.kind is OutcomeKind.NOOP_STOPWORD
    assert not outcome.counted and outcome.count == 0
    assert state.n_counted == 0
    assert state.history[-1].kind == "noop_stopword"
    assert state.history[-1].form == "um"


@pytest.mark.req("REQ-ENG-005")
def test_repeated_lemma_does_not_count(game):
    engine, article, state = game
    state, _ = play(engine, article, state, "gato")
    state, outcome = play(engine, article, state, "Gatos")  # same lemma, another form
    assert outcome.kind is OutcomeKind.NOOP_REPEAT
    assert not outcome.counted and outcome.count == 0
    assert len(outcome.newly_revealed) == 0
    assert state.n_counted == 1
    assert len(state.history) == 2


@pytest.mark.req("REQ-ENG-005")
def test_unknown_word_is_a_counted_miss_and_then_a_repeat(game):
    engine, article, state = game
    guess = engine.parse_guess("zebra")
    assert isinstance(guess, LexicalGuess) and guess.lemma_id == -1
    state, outcome = engine.apply(state, article, guess)
    assert (outcome.kind, outcome.counted, outcome.count) == (OutcomeKind.MISS, True, 0)
    assert "u:zebra" in state.guessed
    state, outcome = play(engine, article, state, "zebras")
    assert outcome.kind is OutcomeKind.NOOP_REPEAT
    assert state.n_counted == 1


@pytest.mark.req("REQ-ENG-005")
def test_noop_cost_one_counts_null_guesses(protocol_ruleset, make_game):
    engine, article, state = make_game(TITLE, BODY, replace(protocol_ruleset, noop_cost=1))
    state, outcome = play(engine, article, state, "de")
    assert outcome.kind is OutcomeKind.NOOP_STOPWORD and outcome.counted
    state, _ = play(engine, article, state, "rato")
    state, outcome = play(engine, article, state, "rato")
    assert outcome.kind is OutcomeKind.NOOP_REPEAT and outcome.counted
    assert state.n_counted == 3


@pytest.mark.req("REQ-ENG-003", "DEC-110")
def test_lemma_shared_with_a_stopword_token_ignores_that_token(game):
    engine, article, state = game
    # "era" and "foi" both have the lemma "ser" in the test table; "foi" is a stopword.
    state, outcome = play(engine, article, state, "ser")
    assert (outcome.kind, outcome.count) == (OutcomeKind.HIT, 1)
    era = article.surfaces.index("era")
    assert outcome.newly_revealed.tolist() == [era]


@pytest.mark.req("REQ-TXT-008")
def test_parse_guess_kinds(game):
    engine, _, _ = game
    assert engine.parse_guess("  ") == InvalidGuess("  ", "empty")
    assert engine.parse_guess("De") == StopwordGuess("de")
    assert engine.parse_guess("#19") == NumericGuess("19")
    lexical = engine.parse_guess("Gatos")
    assert isinstance(lexical, LexicalGuess)
    assert (lexical.lemma, lexical.form) == ("gato", "gatos")
    assert lexical.lemma_id == engine.lexicon.lemma_id("gato") >= 0
    mixed = engine.parse_guess("f1")
    assert isinstance(mixed, LexicalGuess)


@pytest.mark.req("REQ-TXT-008")
def test_invalid_guess_changes_nothing(game):
    engine, article, state = game
    new_state, outcome = engine.apply(state, article, engine.parse_guess("!?"))
    assert new_state is state
    assert outcome.kind is OutcomeKind.INVALID and not outcome.counted
    assert outcome.title_progress.tokens_total == 2


@pytest.mark.req("REQ-TXT-008")
def test_separator_reject_makes_hyphenated_guess_invalid(protocol_ruleset, make_game):
    ruleset = replace(protocol_ruleset, guess_separator="reject")
    engine, article, state = make_game(TITLE, BODY, ruleset)
    guess = engine.parse_guess("gato-preto")
    assert guess == InvalidGuess("gato-preto", "separator")
    _, outcome = engine.apply(state, article, guess)
    assert outcome.kind is OutcomeKind.INVALID


@pytest.mark.req("REQ-ENG-003")
def test_guess_from_lemma_is_the_discrete_interface(game):
    engine, article, state = game
    lemma_id = engine.lexicon.lemma_id("preto")
    guess = engine.guess_from_lemma(lemma_id)
    assert guess == LexicalGuess(lemma_id, "preto", "preto")
    _, outcome = engine.apply(state, article, guess)
    assert (outcome.kind, outcome.count) == (OutcomeKind.HIT, 2)


@pytest.mark.req("REQ-ENG-009")
def test_no_guess_is_accepted_after_victory(game):
    engine, article, state = game
    state, outcome = play(engine, article, state, "gato", "preto")
    assert outcome.won and state.status == "won"
    with pytest.raises(InvalidActionError):
        engine.apply(state, article, engine.parse_guess("rato"))


@pytest.mark.req("REQ-ENG-001")
def test_engine_has_no_randomness_and_no_gymnasium():
    forbidden = re.compile(r"\b(import random|from random|np\.random|numpy\.random|gymnasium)\b")
    root = Path(engine_package.__file__).parent
    sources = sorted(root.rglob("*.py"))
    assert len(sources) >= 10
    for source in sources:
        assert not forbidden.search(source.read_text(encoding="utf-8")), source.name


GUESS_POOL = [
    "gato",
    "gatos",
    "preto",
    "rato",
    "ser",
    "de",
    "um",
    "zebra",
    "1",
    "19",
    "199",
    "90",
    "1990",
    "0",
    "dormia",
    "visto",
    "!?",
    "",
]


def _snapshot(state):
    return (
        state.revealed.copy(),
        dict(state.partial),
        state.guessed,
        state.history,
        state.n_counted,
        state.status,
    )


def _same(a, b) -> bool:
    return np.array_equal(a[0], b[0]) and a[1:] == b[1:]


@pytest.mark.req("REQ-ENG-002")
@settings(max_examples=150, deadline=None)
@given(st.lists(st.sampled_from(GUESS_POOL), min_size=1, max_size=14), st.booleans())
def test_apply_is_pure_and_deterministic(guesses, strict):
    from artigo_env.engine.rulesets import Ruleset

    ruleset = Ruleset(name="prop", lemmatizer="test-protocol", noop_cost=1 if strict else 0)
    article, lexicon = ArticleBuilder(ruleset).from_text(TITLE, BODY)
    engine = Engine(ruleset, lexicon)
    state = engine.new_game(article)
    for raw in guesses:
        if state.status == "won":
            break
        guess = engine.parse_guess(raw)
        before = _snapshot(state)
        first_state, first = engine.apply(state, article, guess)
        second_state, second = engine.apply(state, article, guess)
        assert _same(before, _snapshot(state))  # input untouched
        assert _same(_snapshot(first_state), _snapshot(second_state))
        assert (first.kind, first.counted, first.count, first.won) == (
            second.kind,
            second.counted,
            second.count,
            second.won,
        )
        assert first.newly_revealed.tolist() == second.newly_revealed.tolist()
        assert dict(first.newly_partial) == dict(second.newly_partial)
        # monotonic: nothing is ever hidden again, and counting never goes down
        assert not (state.revealed & ~first_state.revealed).any()
        assert first_state.n_counted - state.n_counted == (1 if first.counted else 0)
        with pytest.raises(ValueError):
            first_state.revealed[0] = False
        state = first_state
