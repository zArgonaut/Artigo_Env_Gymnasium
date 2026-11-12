from __future__ import annotations

from dataclasses import replace

import pytest

from artigo_env.engine.screen import HIDDEN_CHAR, render_partial
from artigo_env.engine.types import FLAG_HAS_DIGIT, FLAG_HEADER, FLAG_INSIDE_PARENTHESES

pytestmark = pytest.mark.fast

TITLE = "Gato (felino)"
BODY = "O gato dorme.\n== Em 1990 ==\nO gato caçava ratos."


def play(engine, article, state, *guesses):
    for raw in guesses:
        state, _ = engine.apply(state, article, engine.parse_guess(raw))
    return state


@pytest.mark.req("REQ-ENG-014")
def test_initial_screen_shows_stopwords_and_lengths_only(protocol_ruleset, make_game):
    engine, article, state = make_game(TITLE, BODY, protocol_ruleset)
    view = engine.screen(state, article)
    assert [(t.text, t.length) for t in view.title] == [(None, 4), (None, 6)]
    assert view.body[0].text == "O" and view.body[0].length is None
    assert (view.body[1].text, view.body[1].length, view.body[1].partial) == (None, 4, None)
    assert view.n_counted == 0 and view.status == "playing" and view.history == ()
    assert len(view.body) == len(article.body)


@pytest.mark.req("REQ-ENG-014")
def test_hidden_tokens_expose_layout_flags_but_not_the_digit_flag(protocol_ruleset, make_game):
    engine, article, state = make_game(TITLE, BODY, protocol_ruleset)
    view = engine.screen(state, article)
    year = view.body[article.body.surface.index("1990")]
    assert year.text is None
    assert year.flags & FLAG_HEADER
    assert not year.flags & FLAG_HAS_DIGIT
    assert view.title[1].flags & FLAG_INSIDE_PARENTHESES
    state = play(engine, article, state, "1990")
    year = engine.screen(state, article).body[article.body.surface.index("1990")]
    assert year.text == "1990" and year.flags & FLAG_HAS_DIGIT


@pytest.mark.req("REQ-ENG-014")
def test_revealed_token_shows_its_original_spelling(protocol_ruleset, make_game):
    engine, article, state = make_game(TITLE, BODY, protocol_ruleset)
    state = play(engine, article, state, "GATO")
    view = engine.screen(state, article)
    assert view.title[0].text == "Gato"
    assert [t.text for t in view.body if t.text and t.text.lower() == "gato"] == ["gato", "gato"]
    assert view.history[0].form == "gato" and view.history[0].count == 3


@pytest.mark.req("REQ-ENG-014", "REQ-RUL-003")
@pytest.mark.parametrize(
    ("mode", "title_length", "body_length"),
    [("all", 4, 4), ("title_only", 4, None), ("none", None, None)],
)
def test_length_visibility_modes(protocol_ruleset, make_game, mode, title_length, body_length):
    ruleset = replace(protocol_ruleset, lengths_visible=mode)
    engine, article, state = make_game(TITLE, BODY, ruleset)
    view = engine.screen(state, article)
    assert view.title[0].length == title_length
    assert view.body[1].length == body_length


@pytest.mark.req("REQ-ENG-014")
def test_partial_with_hidden_lengths_collapses_hidden_runs(protocol_ruleset, make_game):
    ruleset = replace(protocol_ruleset, lengths_visible="none")
    engine, article, state = make_game(TITLE, BODY, ruleset)
    state = play(engine, article, state, "9")
    year = engine.screen(state, article).body[article.body.surface.index("1990")]
    assert year.partial == "█99█"
    assert year.length is None


@pytest.mark.req("REQ-ENG-014")
def test_render_partial():
    assert render_partial("1500", 0b0001, True) == "1███"
    assert render_partial("1500", 0b0001, False) == "1█"
    assert render_partial("1500", 0b1000, False) == "█0"
    assert render_partial("12345", 0b00100, False) == "█3█"
    assert render_partial("12345", 0b00100, True) == "██3██"
    assert HIDDEN_CHAR == "█"


@pytest.mark.req("REQ-ENG-014")
@pytest.mark.parametrize("mode", ["all", "title_only", "none"])
def test_screen_arrays_carry_exactly_what_the_screen_view_shows(protocol_ruleset, make_game, mode):
    ruleset = replace(protocol_ruleset, lengths_visible=mode)
    engine, article, state = make_game(TITLE, BODY, ruleset)
    for raw in ("", "gato", "9", "de", "zebra", "19"):
        if raw:
            state = play(engine, article, state, raw)
        view = engine.screen(state, article)
        arrays = engine.screen_arrays(state, article)
        tokens = (*view.title, *view.body)
        assert arrays.n_title == len(view.title)
        assert len(arrays.visible) == len(arrays.length) == len(arrays.flags) == len(tokens)
        for pos, token in enumerate(tokens):
            assert bool(arrays.visible[pos]) == (token.text is not None)
            assert int(arrays.flags[pos]) == token.flags
            surface = article.surface_at(pos)
            if token.text is not None:
                assert int(arrays.length[pos]) == len(token.text)
                assert pos not in arrays.partial
            elif token.partial is not None:
                shown = int(arrays.length[pos]) > 0
                assert render_partial(surface, arrays.partial[pos], shown) == token.partial
            else:
                assert pos not in arrays.partial
                assert int(arrays.length[pos]) == (token.length or 0)
