from __future__ import annotations

import pytest

from artigo_env.core.errors import UnknownNameError
from artigo_env.scenarios import SCENARIOS, FilterSpec, MetricThreshold, Scenario, get_scenario

pytestmark = pytest.mark.fast


@pytest.mark.req("REQ-SCN-005")
def test_registered_presets():
    assert set(SCENARIOS.names()) == {
        "train-default",
        "train-curriculum",
        "eval-w-val",
        "eval-w-test",
        "eval-ext-val",
        "demo",
    }
    with pytest.raises(UnknownNameError):
        get_scenario("nope")
    for name in SCENARIOS.names():
        scenario = get_scenario(name)
        assert scenario.name == name
        assert Scenario.from_dict(scenario.to_dict()) == scenario


@pytest.mark.req("REQ-SCN-005")
def test_train_and_eval_presets():
    default = get_scenario("train-default")
    assert (default.split, dict(default.source_mix), default.title_distribution) == (
        "train",
        {},
        "match-w",
    )
    assert default.source_cap == 0.5 and default.curriculum == ()
    for name, split in (
        ("eval-w-val", "w-val"),
        ("eval-w-test", "w-test"),
        ("eval-ext-val", "ext-val"),
    ):
        assert get_scenario(name).split == split
        assert get_scenario(name).is_evaluation
    demo = get_scenario("demo")
    assert (demo.split, demo.title_distribution) == ("demo", "natural")


@pytest.mark.req("REQ-SCN-005")
def test_train_curriculum_stages():
    stages = get_scenario("train-curriculum").curriculum
    assert len(stages) == 3
    advance = MetricThreshold("win_rate", 0.8, 200)
    assert stages[0].filters == (
        FilterSpec("body_tokens", {"max": 800}),
        FilterSpec("title_content_words", {"max": 1}),
    )
    assert stages[1].filters == (
        FilterSpec("body_tokens", {"max": 2000}),
        FilterSpec("title_content_words", {"max": 2}),
    )
    assert stages[0].until_metric == advance and stages[1].until_metric == advance
    assert stages[2].filters == () and not stages[2].has_exit
