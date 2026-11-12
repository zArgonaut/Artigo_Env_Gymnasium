"""Scenario presets (SDD 08 §B.4). Python is the source; YAML mirrors come with the runner."""

from __future__ import annotations

from artigo_env.core.registry import Registry
from artigo_env.scenarios.filters import FilterSpec
from artigo_env.scenarios.scenario import CurriculumStage, MetricThreshold, Scenario

SCENARIOS: Registry[Scenario] = Registry("scenario")

_ADVANCE = MetricThreshold(metric="win_rate", min_value=0.8, window=200)


def _stage(max_body: int, max_title: int) -> CurriculumStage:
    return CurriculumStage(
        filters=(
            FilterSpec("body_tokens", {"max": max_body}),
            FilterSpec("title_content_words", {"max": max_title}),
        ),
        until_metric=_ADVANCE,
    )


for _scenario in (
    Scenario(name="train-default", split="train", title_distribution="match-w"),
    Scenario(
        name="train-curriculum",
        split="train",
        title_distribution="match-w",
        curriculum=(_stage(800, 1), _stage(2000, 2), CurriculumStage()),
    ),
    Scenario(name="eval-w-val", split="w-val"),
    Scenario(name="eval-w-test", split="w-test"),
    Scenario(name="eval-ext-val", split="ext-val"),
    Scenario(name="demo", split="demo"),
):
    SCENARIOS.register(_scenario.name, _scenario)


def get_scenario(name: str) -> Scenario:
    """Return the preset registered as ``name``."""
    return SCENARIOS.get(name)
