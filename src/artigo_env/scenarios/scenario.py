"""Scenario configuration: which articles an episode may draw (SDD 08 §B.1, DEC-061)."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from artigo_env.core.config import ConfigBase
from artigo_env.scenarios.filters import SCENARIO_FILTERS, FilterSpec, FrozenMapping

SPLITS = ("train", "ext-val", "w-pool", "w-val", "w-test", "demo")
EVAL_SPLITS = ("ext-val", "w-val", "w-test")
TITLE_DISTRIBUTIONS = ("natural", "match-w")


def _real(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _filter_tuple(owner: str, filters: Any) -> tuple[FilterSpec, ...]:
    if not isinstance(filters, tuple | list):
        raise TypeError(f"{owner}.filters must be a tuple of FilterSpec")
    for spec in filters:
        if not isinstance(spec, FilterSpec):
            raise TypeError(f"{owner}.filters must contain FilterSpec, got {type(spec).__name__}")
        if spec.kind not in SCENARIO_FILTERS:
            SCENARIO_FILTERS.get(spec.kind)  # raises UnknownNameError listing the kinds
    return tuple(filters)


@dataclass(frozen=True)
class MetricThreshold(ConfigBase):
    """Advance when the moving ``metric`` over ``window`` episodes is ``>= min_value``.

    Evaluated by the runner, which then calls ``set_curriculum_stage`` on every worker.
    """

    metric: str
    min_value: float
    window: int

    def __post_init__(self) -> None:
        if not isinstance(self.metric, str) or not self.metric:
            raise ValueError(f"metric must be a non-empty string, got {self.metric!r}")
        if not _real(self.min_value) or not math.isfinite(self.min_value):
            raise ValueError(f"min_value must be a finite number, got {self.min_value!r}")
        object.__setattr__(self, "min_value", float(self.min_value))
        if isinstance(self.window, bool) or not isinstance(self.window, int) or self.window < 1:
            raise ValueError(f"window must be a positive integer, got {self.window!r}")


@dataclass(frozen=True)
class CurriculumStage(ConfigBase):
    """Extra filters for one stage and when to leave it.

    ``until_steps`` counts environment steps spent in this stage (reported to the sampler with
    ``notify_steps``); ``until_metric`` is watched by the runner.
    """

    filters: tuple[FilterSpec, ...] = ()
    until_steps: int | None = None
    until_metric: MetricThreshold | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "filters", _filter_tuple("CurriculumStage", self.filters))
        steps = self.until_steps
        if steps is not None and (
            isinstance(steps, bool) or not isinstance(steps, int) or steps < 1
        ):
            raise ValueError(f"until_steps must be a positive integer or None, got {steps!r}")
        if self.until_metric is not None and not isinstance(self.until_metric, MetricThreshold):
            raise TypeError("until_metric must be a MetricThreshold or None")

    @property
    def has_exit(self) -> bool:
        return self.until_steps is not None or self.until_metric is not None


@dataclass(frozen=True)
class Scenario(ConfigBase):
    """Which documents an episode may draw and how.

    An empty ``source_mix`` means: weight each source by its number of eligible documents, capped
    at ``source_cap`` per source with the excess redistributed to the others ("water filling");
    with a single source, or when the cap cannot be met, the cap is raised to ``1 / n_sources``.
    Sources absent from a non-empty ``source_mix`` have weight zero.
    """

    name: str
    split: str
    source_mix: Mapping[str, float] = field(default_factory=FrozenMapping)
    filters: tuple[FilterSpec, ...] = ()
    title_distribution: str = "natural"
    curriculum: tuple[CurriculumStage, ...] = ()
    source_cap: float = 0.5

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError(f"scenario name must be a non-empty string, got {self.name!r}")
        if self.split not in SPLITS:
            raise ValueError(f"unknown split {self.split!r}; known: {list(SPLITS)}")
        if not isinstance(self.source_mix, Mapping):
            raise TypeError("source_mix must be a mapping of source id to weight")
        mix: dict[str, float] = {}
        for source, weight in self.source_mix.items():
            if not isinstance(source, str) or not source:
                raise ValueError(f"source_mix keys must be non-empty strings, got {source!r}")
            if not _real(weight) or not math.isfinite(weight) or weight < 0:
                raise ValueError(
                    f"source_mix[{source!r}] must be a finite non-negative number, got {weight!r}"
                )
            mix[source] = float(weight)
        if mix and sum(mix.values()) <= 0:
            raise ValueError("source_mix weights must not all be zero")
        object.__setattr__(self, "source_mix", FrozenMapping(mix))
        object.__setattr__(self, "filters", _filter_tuple("Scenario", self.filters))
        if self.title_distribution not in TITLE_DISTRIBUTIONS:
            raise ValueError(
                f"title_distribution must be one of {list(TITLE_DISTRIBUTIONS)}, "
                f"got {self.title_distribution!r}"
            )
        if not isinstance(self.curriculum, tuple | list):
            raise TypeError("curriculum must be a tuple of CurriculumStage")
        for stage in self.curriculum:
            if not isinstance(stage, CurriculumStage):
                raise TypeError(f"curriculum must contain CurriculumStage, got {stage!r}")
        object.__setattr__(self, "curriculum", tuple(self.curriculum))
        if self.curriculum and self.curriculum[-1].has_exit:
            raise ValueError("the last curriculum stage cannot have until_steps or until_metric")
        cap = self.source_cap
        if not _real(cap) or not 0 < cap <= 1:
            raise ValueError(f"source_cap must be in (0, 1], got {cap!r}")
        object.__setattr__(self, "source_cap", float(cap))

    @property
    def is_evaluation(self) -> bool:
        return self.split in EVAL_SPLITS

    @property
    def stages(self) -> tuple[CurriculumStage, ...]:
        """The curriculum, or a single unrestricted stage when there is none."""
        return self.curriculum or (CurriculumStage(),)
