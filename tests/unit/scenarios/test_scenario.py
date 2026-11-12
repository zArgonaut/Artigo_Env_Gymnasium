from __future__ import annotations

import json
import pickle
import subprocess
import sys

import pytest

from artigo_env.core.errors import UnknownNameError
from artigo_env.scenarios import CurriculumStage, FilterSpec, MetricThreshold, Scenario

pytestmark = pytest.mark.fast


def _scenario() -> Scenario:
    return Scenario(
        name="nested",
        split="train",
        source_mix={"wiki": 2.0, "news": 1},
        filters=(FilterSpec("domain", {"in": ["sci", "art"]}),),
        title_distribution="match-w",
        curriculum=(
            CurriculumStage(
                filters=(FilterSpec("body_tokens", {"max": 800}),),
                until_steps=1000,
                until_metric=MetricThreshold("win_rate", 0.8, 200),
            ),
            CurriculumStage(filters=(FilterSpec("title_in_body", {"value": False}),)),
        ),
        source_cap=0.4,
    )


@pytest.mark.req("REQ-REPO-002")
def test_round_trip_with_nested_filters_and_stages():
    scenario = _scenario()
    data = scenario.to_dict()
    json.dumps(data)
    again = Scenario.from_dict(json.loads(json.dumps(data)))
    assert again == scenario
    assert again.content_hash() == scenario.content_hash()
    assert again.curriculum[0].until_metric == MetricThreshold("win_rate", 0.8, 200)
    assert again.filters[0].params["in"] == ("sci", "art")
    assert hash(again) == hash(scenario)
    assert pickle.loads(pickle.dumps(scenario)) == scenario


@pytest.mark.req("REQ-REPO-002")
def test_content_hash_is_stable_across_processes_and_sensitive_to_content():
    scenario = _scenario()
    code = (
        "import json, sys; from artigo_env.scenarios import Scenario;"
        "print(Scenario.from_dict(json.loads(sys.argv[1])).content_hash())"
    )
    out = subprocess.run(
        [sys.executable, "-c", code, json.dumps(scenario.to_dict())],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert out == scenario.content_hash()
    other = Scenario.from_dict({**scenario.to_dict(), "source_cap": 0.5})
    assert other.content_hash() != scenario.content_hash()


@pytest.mark.req("REQ-SCN-005")
def test_frozen_mappings_reject_mutation():
    scenario = _scenario()
    with pytest.raises(TypeError):
        scenario.source_mix["wiki"] = 3.0  # type: ignore[index]
    assert dict(scenario.source_mix) == {"wiki": 2.0, "news": 1.0}


@pytest.mark.req("REQ-SCN-005")
@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"split": "valid"}, ValueError),
        ({"name": ""}, ValueError),
        ({"source_mix": {"wiki": -1.0}}, ValueError),
        ({"source_mix": {"wiki": 0.0, "news": 0}}, ValueError),
        ({"source_mix": {"wiki": float("nan")}}, ValueError),
        ({"title_distribution": "uniform"}, ValueError),
        ({"source_cap": 0.0}, ValueError),
        ({"source_cap": 1.5}, ValueError),
        ({"filters": ("domain",)}, TypeError),
        ({"curriculum": (CurriculumStage(until_steps=10),)}, ValueError),  # last stage exits
    ],
)
def test_scenario_validation(kwargs, error):
    base = {"name": "s", "split": "train"}
    with pytest.raises(error):
        Scenario(**{**base, **kwargs})


@pytest.mark.req("REQ-SCN-005")
def test_stage_and_threshold_validation():
    with pytest.raises(ValueError):
        CurriculumStage(until_steps=0)
    with pytest.raises(ValueError):
        MetricThreshold("win_rate", 0.8, 0)
    with pytest.raises(ValueError):
        MetricThreshold("", 0.8, 10)
    with pytest.raises(UnknownNameError):
        Scenario.from_dict({"name": "s", "split": "train", "filters": [{"kind": "nope"}]})
    with pytest.raises(ValueError, match="unknown keys"):
        Scenario.from_dict({"name": "s", "split": "train", "colour": "red"})


@pytest.mark.req("REQ-SCN-005")
def test_known_splits_and_evaluation_flag():
    for split in ("train", "ext-val", "w-pool", "w-val", "w-test", "demo"):
        scenario = Scenario(name="s", split=split)
        assert scenario.is_evaluation == (split in ("ext-val", "w-val", "w-test"))
    assert Scenario(name="s", split="train").stages == (CurriculumStage(),)
