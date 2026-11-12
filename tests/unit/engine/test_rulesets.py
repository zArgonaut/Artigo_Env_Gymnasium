from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from artigo_env.core.errors import IncompatibleRulesetError, UnknownNameError
from artigo_env.engine.rulesets import PRESETS, HintSpec, Ruleset, get_ruleset
from artigo_env.engine.rulesets.components import HINT_RULES, LENGTH_RULES, NUMBER_RULES
from artigo_env.text.stopwords import StopwordSpec

pytestmark = pytest.mark.fast


@pytest.mark.req("REQ-RUL-001")
def test_presets_differ_from_canonical_only_in_the_declared_levers():
    canonical = get_ruleset("canonical")
    assert canonical == Ruleset(name="canonical")
    assert (canonical.lemmatizer, canonical.numbers, canonical.lengths_visible) == (
        "open-tables",
        "substring",
        "all",
    )
    assert (canonical.noop_cost, canonical.title_parenthetical, canonical.hint.kind) == (
        0,
        "required",
        "none",
    )
    assert canonical.stopwords == StopwordSpec(kind="game227")
    assert set(PRESETS.names()) == {
        "canonical",
        "hard",
        "no-grouping",
        "no-lengths",
        "strict-noops",
    }
    assert get_ruleset("no-grouping") == replace(
        canonical, name="no-grouping", lemmatizer="identity"
    )
    assert get_ruleset("no-lengths") == replace(
        canonical, name="no-lengths", lengths_visible="none"
    )
    assert get_ruleset("strict-noops") == replace(canonical, name="strict-noops", noop_cost=1)
    assert get_ruleset("hard") == replace(
        canonical, name="hard", lemmatizer="identity", lengths_visible="none", noop_cost=1
    )
    with pytest.raises(UnknownNameError):
        get_ruleset("nope")


@pytest.mark.req("REQ-RUL-003")
@pytest.mark.parametrize(
    "change",
    [
        {"numbers": "nope"},
        {"lengths_visible": "nope"},
        {"title_parenthetical": "nope"},
        {"lemmatizer": "nope"},
        {"tokenizer_hyphen": "nope"},
        {"guess_separator": "nope"},
        {"hint": HintSpec(kind="nope")},
        {"stopwords": StopwordSpec(kind="nope")},
    ],
)
def test_unknown_lever_names_are_rejected_at_construction(change):
    with pytest.raises(UnknownNameError, match="nope"):
        Ruleset(name="bad", **change)


@pytest.mark.req("REQ-RUL-003")
@pytest.mark.parametrize("cost", [2, -1, True, 0.5])
def test_noop_cost_must_be_zero_or_one(cost):
    with pytest.raises(ValueError):
        Ruleset(name="bad", noop_cost=cost)


@pytest.mark.req("REQ-RUL-003")
def test_hint_after_must_be_a_non_negative_integer():
    with pytest.raises(ValueError):
        HintSpec(kind="first-letter", after=-1)


@pytest.mark.req("REQ-RUL-001")
def test_content_hash_covers_config_stopwords_and_lemmatizer(protocol_ruleset, tmp_path):
    base = protocol_ruleset.content_hash()
    assert len(base) == 64
    assert base == replace(protocol_ruleset).content_hash()
    assert base != replace(protocol_ruleset, noop_cost=1).content_hash()
    assert base != replace(protocol_ruleset, lemmatizer="identity").content_hash()
    assert base != replace(protocol_ruleset, name="other").content_hash()

    # two custom stopword files under the same path name: only the content differs
    path = tmp_path / "stop.txt"
    custom = replace(protocol_ruleset, stopwords=StopwordSpec(kind="custom", path=str(path)))
    path.write_text("de\no\na\n", encoding="utf-8")
    first = custom.content_hash()
    path.write_text("de\no\na\num\n", encoding="utf-8")
    second = custom.content_hash()
    assert first != second
    assert first != base


@pytest.mark.req("REQ-RUL-001")
def test_round_trip_through_a_plain_dict(protocol_ruleset):
    ruleset = replace(
        protocol_ruleset,
        noop_cost=1,
        hint=HintSpec(kind="first-letter", after=3),
        numbers="disabled",
    )
    data = ruleset.to_dict()
    assert data["hint"] == {"kind": "first-letter", "after": 3}
    assert data["stopwords"] == {"kind": "game227", "path": None}
    assert Ruleset.from_dict(data) == ruleset
    assert Ruleset.from_dict(data).content_hash() == ruleset.content_hash()


@pytest.mark.req("REQ-RUL-004")
def test_check_build_rejects_another_hyphen_mode(protocol_ruleset):
    protocol_ruleset.check_build("split")
    with pytest.raises(IncompatibleRulesetError, match="join"):
        protocol_ruleset.check_build("join")
    replace(protocol_ruleset, tokenizer_hyphen="join").check_build("join")


class _OddDigitsOnly:
    """Fictitious number rule: reveals only odd digits of a matching token."""

    enabled = True

    def mask(self, surface: str, digits: str) -> int:
        if digits not in surface:
            return 0
        return sum(1 << i for i, char in enumerate(surface) if char in "13579")


class _TitleLengthsHidden:
    def visible(self, in_title: bool) -> bool:
        return not in_title


class _LastLetterHint:
    def reveals(self, article, n_counted, spec):
        if n_counted < spec.after:
            return {}
        return {
            pos: 1 << (int(article.lengths[pos]) - 1)
            for pos in range(article.n_title)
            if not article.stop[pos]
        }


@pytest.mark.req("REQ-RUL-003")
def test_a_new_lever_variant_needs_no_engine_change(protocol_ruleset, make_game):
    for registry, name, component in (
        (NUMBER_RULES, "test-odd-digits", _OddDigitsOnly()),
        (LENGTH_RULES, "test-body-only", _TitleLengthsHidden()),
        (HINT_RULES, "test-last-letter", _LastLetterHint()),
    ):
        if name not in registry:
            registry.register(name, component)
    ruleset = replace(
        protocol_ruleset,
        numbers="test-odd-digits",
        lengths_visible="test-body-only",
        hint=HintSpec(kind="test-last-letter", after=1),
    )
    engine, article, state = make_game("Gato", "O gato de 1234 e o rato.", ruleset)
    view = engine.screen(state, article)
    assert view.title[0].length is None
    assert view.body[1].length == 4

    state, outcome = engine.apply(state, article, engine.parse_guess("12"))
    year = article.surfaces.index("1234")
    assert outcome.count == 1
    assert outcome.newly_partial[year] == 0b0101
    assert outcome.newly_partial[0] == 0b1000  # hint: last letter of the title token
    view = engine.screen(state, article)
    assert view.body[year - article.n_title].partial == "1█3█"
    assert view.title[0].partial == "█o"
    assert not np.any(state.revealed[: article.n_title])
