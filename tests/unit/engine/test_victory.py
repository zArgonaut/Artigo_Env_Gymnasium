from __future__ import annotations

from dataclasses import replace

import pytest

from artigo_env.engine import OutcomeKind
from artigo_env.engine.rulesets import HintSpec

pytestmark = pytest.mark.fast


def play(engine, article, state, *guesses):
    outcome = None
    for raw in guesses:
        state, outcome = engine.apply(state, article, engine.parse_guess(raw))
    return state, outcome


@pytest.mark.req("REQ-ENG-009")
def test_victory_needs_every_content_token_of_the_title(protocol_ruleset, make_game):
    engine, article, state = make_game(
        "Rio de Janeiro", "O Rio de Janeiro é uma cidade.", protocol_ruleset
    )
    state, outcome = play(engine, article, state, "rio")
    assert not outcome.won and state.status == "playing"
    assert outcome.title_progress.tokens_total == 2  # "de" is a stopword
    state, outcome = play(engine, article, state, "janeiro")
    assert outcome.won and state.status == "won"
    assert outcome.kind is OutcomeKind.HIT


@pytest.mark.req("REQ-ENG-009")
def test_body_only_guesses_never_win(protocol_ruleset, make_game):
    engine, article, state = make_game("Gato", "O rato e a cidade.", protocol_ruleset)
    state, outcome = play(engine, article, state, "rato", "cidade")
    assert not outcome.won
    assert state.revealed[1:].all() and not state.revealed[0]


@pytest.mark.req("REQ-ENG-009")
def test_numeric_title_is_won_through_substrings(protocol_ruleset, make_game):
    engine, article, state = make_game("1984", "O livro 1984 saiu em 1949.", protocol_ruleset)
    state, outcome = play(engine, article, state, "19")
    assert not outcome.won
    assert outcome.title_progress.letters_revealed == 2
    state, outcome = play(engine, article, state, "84")
    assert outcome.won
    assert outcome.newly_revealed.tolist()[0] == 0
    assert state.revealed[0]


@pytest.mark.req("REQ-ENG-009")
def test_parenthetical_title_tokens_are_required_by_default(protocol_ruleset, make_game):
    title, body = "Mercúrio (planeta)", "Mercúrio é o menor planeta."
    engine, article, state = make_game(title, body, protocol_ruleset)
    state, outcome = play(engine, article, state, "mercúrio")
    assert not outcome.won
    assert outcome.title_progress.tokens_total == 2
    state, outcome = play(engine, article, state, "planeta")
    assert outcome.won


@pytest.mark.req("REQ-ENG-009", "REQ-RUL-003")
def test_parenthetical_title_tokens_can_be_ignored(protocol_ruleset, make_game):
    ruleset = replace(protocol_ruleset, title_parenthetical="ignored")
    engine, article, state = make_game("Mercúrio (planeta)", "Mercúrio é o menor planeta.", ruleset)
    state, outcome = play(engine, article, state, "mercúrio")
    assert outcome.won
    assert outcome.title_progress.tokens_total == 1
    assert not state.revealed[1]  # "planeta" is still hidden, and that is fine


@pytest.mark.req("REQ-RUL-003")
def test_first_letter_hint_after_two_counted_guesses(protocol_ruleset, make_game):
    ruleset = replace(protocol_ruleset, hint=HintSpec(kind="first-letter", after=2))
    engine, article, state = make_game("Gato preto", "Um rato e um pato.", ruleset)
    state, outcome = play(engine, article, state, "de")  # not counted
    state, outcome = play(engine, article, state, "rato")
    assert dict(state.partial) == {}
    state, outcome = play(engine, article, state, "zebra")  # second counted guess
    assert dict(outcome.newly_partial) == {0: 1, 1: 1}
    view = engine.screen(state, article)
    assert [token.partial for token in view.title] == ["G███", "p████"]
    assert outcome.title_progress.letters_revealed == 2
    assert not outcome.won
    state, outcome = play(engine, article, state, "pato")
    assert dict(outcome.newly_partial) == {}  # nothing new to hint


@pytest.mark.req("REQ-RUL-003")
def test_hint_can_complete_a_one_letter_title(protocol_ruleset, make_game):
    ruleset = replace(protocol_ruleset, hint=HintSpec(kind="first-letter", after=1))
    engine, article, state = make_game("X", "A letra X.", ruleset)
    state, outcome = play(engine, article, state, "letra")
    assert outcome.won and state.revealed[0]
    assert outcome.kind is OutcomeKind.HIT
