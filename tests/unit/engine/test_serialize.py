from __future__ import annotations

import json

import numpy as np
import pytest

from artigo_env.engine import Engine

pytestmark = pytest.mark.fast

TITLE = "Gato de 1990"
BODY = "O gato preto nasceu em 1990 e em 1991 os gatos dormiam."


def play(engine, article, state, *guesses):
    for raw in guesses:
        state, _ = engine.apply(state, article, engine.parse_guess(raw))
    return state


def assert_same_state(a, b) -> None:
    assert np.array_equal(a.revealed, b.revealed)
    assert a.revealed.dtype == b.revealed.dtype == np.bool_
    assert dict(a.partial) == dict(b.partial)
    assert a.guessed == b.guessed
    assert a.history == b.history
    assert (a.n_counted, a.status) == (b.n_counted, b.status)


@pytest.mark.req("REQ-ENG-011")
def test_round_trip_in_the_middle_of_a_game(protocol_ruleset, make_game):
    engine, article, state = make_game(TITLE, BODY, protocol_ruleset)
    state = play(engine, article, state, "preto", "19", "de", "zebra", "preto")
    data = json.loads(json.dumps(Engine.serialize(state)))
    restored = Engine.deserialize(data)
    assert_same_state(state, restored)
    # the restored state keeps playing exactly like the original
    continued = play(engine, article, state, "gato", "90")
    resumed = play(engine, article, restored, "gato", "90")
    assert_same_state(continued, resumed)
    assert resumed.status == "won"
    assert_same_state(resumed, Engine.deserialize(Engine.serialize(resumed)))


@pytest.mark.req("REQ-ENG-011")
def test_round_trip_of_a_new_game_and_odd_sizes(protocol_ruleset, make_game):
    for body in ("Um gato.", "Um gato e nove ratos de casa.", BODY):
        engine, article, state = make_game(TITLE, body, protocol_ruleset)
        restored = Engine.deserialize(Engine.serialize(state))
        assert_same_state(state, restored)
        assert len(restored.revealed) == article.n_total


@pytest.mark.req("REQ-ENG-011")
def test_serialized_form_is_plain_json(protocol_ruleset, make_game):
    engine, article, state = make_game(TITLE, BODY, protocol_ruleset)
    state = play(engine, article, state, "19")
    data = Engine.serialize(state)
    assert data["format_version"] == 1
    assert data["size"] == article.n_total
    assert isinstance(data["revealed"], str)
    assert data["history"] == [
        {"key": "n:19", "form": "19", "kind": "hit", "counted": True, "count": 3}
    ]
    assert json.dumps(data)


@pytest.mark.req("REQ-ENG-011")
@pytest.mark.parametrize(
    "damage",
    [
        lambda d: d.pop("revealed"),
        lambda d: d.update(format_version=2),
        lambda d: d.update(status="lost"),
        lambda d: d.update(size=d["size"] + 64),
        lambda d: d.update(history=[{"key": "x"}]),
    ],
)
def test_malformed_state_is_rejected(protocol_ruleset, make_game, damage):
    engine, article, state = make_game(TITLE, BODY, protocol_ruleset)
    data = Engine.serialize(state)
    damage(data)
    with pytest.raises(ValueError):
        Engine.deserialize(data)
