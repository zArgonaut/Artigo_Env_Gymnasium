from __future__ import annotations

from dataclasses import replace

import pytest

from artigo_env.engine import LexicalGuess, OutcomeKind
from artigo_env.engine.rulesets.components import NUMBER_RULES

pytestmark = pytest.mark.fast

TITLE = "Ano 1101"
BODY = "Em 1101 e em 111 houve 11 festas; o F1 e o 2º lugar ficaram em 1.500 pontos."


@pytest.fixture
def game(protocol_ruleset, make_game):
    return make_game(TITLE, BODY, protocol_ruleset)


def play(engine, article, state, *guesses):
    outcome = None
    for raw in guesses:
        state, outcome = engine.apply(state, article, engine.parse_guess(raw))
    return state, outcome


def bits(*indexes: int) -> int:
    return sum(1 << i for i in indexes)


@pytest.mark.req("REQ-ENG-008", "DEC-108")
@pytest.mark.parametrize(
    ("surface", "digits", "expected"),
    [
        ("1101", "1", bits(0, 1, 3)),
        ("111", "11", bits(0, 1)),  # non-overlapping, from the left
        ("1111", "11", bits(0, 1, 2, 3)),
        ("1500", "500", bits(1, 2, 3)),
        ("1500", "1500", bits(0, 1, 2, 3)),
        ("1500", "15000", 0),
        ("f1", "1", bits(1)),
        ("2º", "2", bits(0)),
        ("casa", "1", 0),
    ],
)
def test_substring_mask(surface, digits, expected):
    assert NUMBER_RULES.get("substring").mask(surface, digits) == expected


@pytest.mark.req("REQ-ENG-008")
def test_whole_token_and_disabled_masks():
    whole = NUMBER_RULES.get("whole-token")
    assert whole.mask("1500", "1500") == bits(0, 1, 2, 3)
    assert whole.mask("1500", "500") == 0
    disabled = NUMBER_RULES.get("disabled")
    assert disabled.mask("1500", "1500") == 0
    assert not disabled.enabled and whole.enabled


@pytest.mark.req("REQ-ENG-008")
def test_numeric_guess_counts_tokens_and_reveals_partially(game):
    engine, article, state = game
    state, outcome = play(engine, article, state, "11")
    # tokens containing "11": 1101 (title), 1101, 111, 11 -> 4; only "11" is whole
    assert (outcome.kind, outcome.counted, outcome.count) == (OutcomeKind.HIT, True, 4)
    eleven = article.surfaces.index("11")
    assert outcome.newly_revealed.tolist() == [eleven]
    title_token = engine.screen(state, article).title[1]
    assert title_token.text is None
    assert title_token.partial == "11██"
    assert title_token.length is None  # the partial string already carries the length
    assert outcome.title_progress.letters_revealed == 2
    assert outcome.title_progress.tokens_revealed == 0


@pytest.mark.req("REQ-ENG-008")
def test_partial_reveals_accumulate_until_the_token_is_whole(game):
    engine, article, state = game
    state, _ = play(engine, article, state, "11")
    state, outcome = play(engine, article, state, "01")
    # "01" occurs in the two "1101" tokens only; both become whole
    assert outcome.count == 2
    assert sorted(outcome.newly_revealed.tolist()) == [
        pos for pos, surface in enumerate(article.surfaces) if surface == "1101"
    ]
    assert dict(outcome.newly_partial) == {
        pos: bits(2, 3) for pos in outcome.newly_revealed.tolist()
    }
    assert state.revealed[1]
    assert engine.screen(state, article).title[1].text == "1101"


@pytest.mark.req("REQ-ENG-008")
def test_numeric_guess_reaches_mixed_tokens_and_split_numbers(game):
    engine, article, state = game
    state, outcome = play(engine, article, state, "1")
    surfaces = article.surfaces
    f1 = surfaces.index("F1")
    assert outcome.newly_partial[f1] == bits(1)
    assert not state.revealed[f1]
    assert state.revealed[surfaces.index("1")]  # the "1" of "1.500"
    assert not state.revealed[surfaces.index("500")]
    # mixed guess is lexical and reveals the whole token
    guess = engine.parse_guess("f1")
    assert isinstance(guess, LexicalGuess)
    state, outcome = engine.apply(state, article, guess)
    assert (outcome.kind, outcome.count) == (OutcomeKind.HIT, 1)
    assert state.revealed[f1]


@pytest.mark.req("REQ-ENG-005", "REQ-ENG-008")
def test_numeric_repeat_and_miss(game):
    engine, article, state = game
    state, outcome = play(engine, article, state, "7")
    assert (outcome.kind, outcome.counted, outcome.count) == (OutcomeKind.MISS, True, 0)
    state, outcome = play(engine, article, state, "#7")
    assert outcome.kind is OutcomeKind.NOOP_REPEAT and not outcome.counted
    assert state.n_counted == 1
    assert state.history[0].key == "n:7"


@pytest.mark.req("REQ-ENG-008")
def test_numeric_count_includes_already_revealed_tokens(game):
    engine, article, state = game
    state, _ = play(engine, article, state, "11")
    state, outcome = play(engine, article, state, "1")
    # tokens with a "1": 1101 (x2), 111, 11 (already whole), F1, 1 -> 6
    assert outcome.count == 6
    assert article.surfaces.index("11") not in outcome.newly_revealed.tolist()


@pytest.mark.req("REQ-ENG-008", "REQ-RUL-003")
def test_whole_token_ruleset(protocol_ruleset, make_game):
    ruleset = replace(protocol_ruleset, numbers="whole-token")
    engine, article, state = make_game(TITLE, BODY, ruleset)
    state, outcome = play(engine, article, state, "11")
    assert outcome.count == 1
    assert dict(state.partial) == {article.surfaces.index("11"): bits(0, 1)}
    state, outcome = play(engine, article, state, "1101")
    assert outcome.count == 2 and state.revealed[1]


@pytest.mark.req("REQ-ENG-008", "REQ-RUL-003")
def test_disabled_numbers_make_digit_guesses_lexical(protocol_ruleset, make_game):
    ruleset = replace(protocol_ruleset, numbers="disabled")
    engine, article, state = make_game(TITLE, BODY, ruleset)
    guess = engine.parse_guess("1101")
    assert isinstance(guess, LexicalGuess)
    state, outcome = engine.apply(state, article, guess)
    assert (outcome.kind, outcome.count) == (OutcomeKind.HIT, 2)
    assert dict(state.partial) == {}
    state, outcome = play(engine, article, state, "11")
    assert outcome.count == 1  # only the token "11", no substrings
