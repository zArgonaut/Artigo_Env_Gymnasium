"""Permutation test for the screen (SDD 15, REQ-ENG-014).

Two articles share every word the guesses reveal and differ only in words that stay hidden. If
the screens are equal after the same guesses, the screen carries nothing about hidden words
beyond what the ruleset declares visible (lengths, when shown).
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from artigo_env.engine.rulesets import PRESETS, get_ruleset

pytestmark = pytest.mark.fast

# Same layout; hidden words differ but keep their lengths (rato/pato, dorme/corre, 1990/1991).
ARTICLE_A = ("Rato preto", "O rato preto dorme na casa azul.\n\n== Vida ==\nEm 1990 o rato fugiu.")
ARTICLE_B = ("Pato preto", "O pato preto corre na casa anil.\n\n== Vida ==\nEm 1991 o pato fugiu.")
# Hidden words also differ in length (only safe when the ruleset hides lengths).
ARTICLE_C = (
    "Jacaré preto",
    "O jacaré preto vai na casa lilás.\n\n== Vida ==\nEm 19911 o jacaré fugiu.",
)

GUESSES = ["preto", "casa", "de", "19", "zebra", "vida", "preto", "fugiu", "!?"]


def _ruleset(preset: str):
    ruleset = get_ruleset(preset)
    if ruleset.lemmatizer != "identity":
        ruleset = replace(ruleset, lemmatizer="test-protocol")
    return ruleset


def _screens(make_game, texts, ruleset):
    engine, article, state = make_game(*texts, ruleset)
    views = [engine.screen(state, article)]
    for raw in GUESSES:
        state, _ = engine.apply(state, article, engine.parse_guess(raw))
        views.append(engine.screen(state, article))
    assert state.status == "playing"
    return views


@pytest.mark.req("REQ-ENG-014")
@pytest.mark.parametrize("preset", sorted(PRESETS.names()))
def test_screens_do_not_depend_on_hidden_words(make_game, preset):
    ruleset = _ruleset(preset)
    assert _screens(make_game, ARTICLE_A, ruleset) == _screens(make_game, ARTICLE_B, ruleset)


@pytest.mark.req("REQ-ENG-014")
@pytest.mark.parametrize("preset", ["no-lengths", "hard"])
def test_hidden_lengths_do_not_leak_when_the_ruleset_hides_them(make_game, preset):
    ruleset = _ruleset(preset)
    assert _screens(make_game, ARTICLE_A, ruleset) == _screens(make_game, ARTICLE_C, ruleset)


@pytest.mark.req("REQ-ENG-014")
def test_the_test_can_fail(make_game):
    """Sanity check of the method: with lengths visible, different lengths do show."""
    ruleset = _ruleset("canonical")
    assert _screens(make_game, ARTICLE_A, ruleset) != _screens(make_game, ARTICLE_C, ruleset)
