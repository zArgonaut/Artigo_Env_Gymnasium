"""The 15-guess protocol on a synthetic article (SDD 07 §2.1, SDD 15 §2).

Expected counts were derived by hand from the rules read in the game client (DEC-108, DEC-110)
and the small rule table in ``tests/fixtures/lemma_tables/protocol.json``; they are not copied
from the engine's output.
"""

from __future__ import annotations

import pytest

from artigo_env.engine import OutcomeKind

pytestmark = pytest.mark.fast

HIT, MISS = OutcomeKind.HIT, OutcomeKind.MISS

# (raw guess, kind, counted, count)
PROTOCOL = [
    ("de", OutcomeKind.NOOP_STOPWORD, False, 0),  # 1: stopword, "não conta"
    ("anos", HIT, True, 2),  # 2: anos, ano
    ("primeira", HIT, True, 4),  # 3: primeiro, primeira (x2), primeiros
    ("alta", HIT, True, 3),  # 4: alta (x2), altas; "alto" stays hidden
    ("ter", HIT, True, 3),  # 5: tem, tinha, teve; "tinham" is not in the table
    ("principal", HIT, True, 2),  # 6: principais, principal; not "principalmente"
    ("historia", MISS, True, 0),  # 7: no accent, different word
    ("norte", HIT, True, 2),  # 8: "norte" and the first half of "norte-americano"
    ("brasil", HIT, True, 2),  # 9: title + body; not "brasileiro"
    ("1", HIT, True, 4),  # 10: 1822, 19, 1º (partial) and the "1" of "1.500" (whole)
    ("tornou", HIT, True, 2),  # 11: tornou, tornaram; not "torna"
    ("casinha", HIT, True, 2),  # 12: casinhas, casinha
    ("regiões", HIT, True, 3),  # 13: região, regiões (x2)
    ("sul-americano", MISS, True, 0),  # 14: cleaned to "sulamericano"; counts as an error
    ("1500", MISS, True, 0),  # 15: "1.500" is two tokens, "1" and "500"
]


def _positions(article, word: str) -> list[int]:
    return [pos for pos, surface in enumerate(article.surfaces) if surface.lower() == word]


def _revealed_words(state, article) -> set[str]:
    return {
        article.surface_at(pos).lower() for pos in range(article.n_total) if state.revealed[pos]
    }


@pytest.fixture
def played(protocol_ruleset, protocol_text, make_game):
    engine, article, state = make_game(*protocol_text, protocol_ruleset)
    outcomes = []
    for raw, *_ in PROTOCOL:
        state, outcome = engine.apply(state, article, engine.parse_guess(raw))
        outcomes.append(outcome)
    return engine, article, state, outcomes


@pytest.mark.req("REQ-ENG-003", "REQ-ENG-005", "REQ-ENG-007", "REQ-ENG-008")
def test_protocol_outcomes(played):
    _, _, state, outcomes = played
    got = [(o.kind, o.counted, o.count) for o in outcomes]
    assert got == [(kind, counted, count) for _, kind, counted, count in PROTOCOL]
    assert state.n_counted == 14
    assert state.status == "playing"
    assert [r.form for r in state.history][13] == "sulamericano"


@pytest.mark.req("REQ-ENG-003")
def test_protocol_reveals_every_position_of_the_lemma(played):
    _, article, state, _ = played
    for word in ("anos", "ano", "primeiro", "primeira", "primeiros", "alta", "altas"):
        assert all(state.revealed[p] for p in _positions(article, word)), word
    for word in ("tem", "tinha", "teve", "principais", "principal", "tornou", "tornaram"):
        assert all(state.revealed[p] for p in _positions(article, word)), word
    for word in ("casinhas", "casinha", "região", "regiões", "norte", "brasil"):
        assert all(state.revealed[p] for p in _positions(article, word)), word


@pytest.mark.req("REQ-ENG-003", "REQ-ENG-007")
def test_protocol_does_not_reveal_near_misses(played):
    _, article, state, _ = played
    hidden = {
        "alto",
        "tinham",
        "principalmente",
        "história",
        "americano",
        "brasileiro",
        "torna",
        "sul",
        "500",
    }
    assert hidden.isdisjoint(_revealed_words(state, article))


@pytest.mark.req("REQ-ENG-008")
def test_protocol_numeric_guess_is_partial(played):
    engine, article, state, outcomes = played
    one = outcomes[9]
    (p1822,) = _positions(article, "1822")
    (p19,) = _positions(article, "19")
    (p1o,) = _positions(article, "1º")
    (p1,) = _positions(article, "1")
    assert one.newly_revealed.tolist() == [p1]
    assert dict(one.newly_partial) == {p1822: 1, p19: 1, p1o: 1, p1: 1}
    assert not state.revealed[p1822] and not state.revealed[p19] and not state.revealed[p1o]
    view = engine.screen(state, article)
    body = view.body
    offset = article.n_title
    assert body[p1822 - offset].partial == "1███"
    assert body[p19 - offset].partial == "1█"
    assert body[p1o - offset].partial == "1█"
    assert body[p1 - offset].text == "1"


@pytest.mark.req("REQ-ENG-008", "REQ-ENG-009")
def test_protocol_continues_to_victory(played):
    engine, article, state, _ = played
    state, outcome = engine.apply(state, article, engine.parse_guess("500"))
    assert (outcome.kind, outcome.count) == (HIT, 1)
    assert outcome.title_progress.tokens_revealed == 1
    assert outcome.title_progress.tokens_total == 2
    state, outcome = engine.apply(state, article, engine.parse_guess("História"))
    assert (outcome.kind, outcome.count, outcome.won) == (HIT, 3, True)
    assert state.status == "won"
    assert state.n_counted == 16
    assert outcome.title_progress.token_fraction == 1.0
