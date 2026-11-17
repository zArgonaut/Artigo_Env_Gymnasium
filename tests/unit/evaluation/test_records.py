"""Trajectory schema of SDD 14 §6 (REQ-EVL-005)."""

from __future__ import annotations

import pytest

from artigo_env.evaluation.records import (
    EPISODES_SCHEMA,
    STEPS_SCHEMA,
    StepRow,
    read_episodes,
    read_steps,
    read_summary,
    write_outputs,
)
from tests.unit.evaluation.test_metrics import episode

pytestmark = pytest.mark.fast

SDD_STEP_COLUMNS = [
    "episode_id",
    "article_ref",
    "seed",
    "step",
    "action_lemma",
    "generator",
    "outcome",
    "count",
    "reward_total",
    "reward_components",
    "revealed_frac",
    "title_progress",
]
SDD_EPISODE_COLUMNS = [
    "episode_id",
    "article_ref",
    "source",
    "domain",
    "n_title_content",
    "n_body_tokens",
    "N",
    "won",
    "truncated",
    "title_oov",
    "efficiency",
    "reveal_auc",
    "first_title_hit",
    "reward_component_sums",
]


@pytest.mark.req("REQ-EVL-005")
def test_schemas_start_with_the_sdd_columns_in_order():
    assert STEPS_SCHEMA.names[: len(SDD_STEP_COLUMNS)] == SDD_STEP_COLUMNS
    assert EPISODES_SCHEMA.names[: len(SDD_EPISODE_COLUMNS)] == SDD_EPISODE_COLUMNS


@pytest.mark.req("REQ-EVL-005")
def test_write_and_read_back(tmp_path):
    step = StepRow(
        episode_id="s0-a00000",
        article_ref="ref0",
        seed=0,
        step=1,
        action_lemma="gato",
        outcome="hit",
        count=2,
        reward_total=-1.0,
        revealed_frac=0.25,
        title_progress=0.5,
        action=7,
    )
    summary = {"metrics": {"n_mean": {"value": float("nan")}}, "run_id": "x"}
    write_outputs(tmp_path / "out", [step], [episode(0, 1)], summary)
    steps = read_steps(tmp_path / "out")
    episodes = read_episodes(tmp_path / "out")
    assert steps.schema.equals(STEPS_SCHEMA)
    assert episodes.schema.equals(EPISODES_SCHEMA)
    row = steps.to_pylist()[0]
    assert row["generator"] is None and row["reward_components"] is None
    assert row["action_lemma"] == "gato" and row["action"] == 7
    erow = episodes.to_pylist()[0]
    assert erow["reward_component_sums"] is None and erow["N"] == 1
    assert read_summary(tmp_path / "out")["metrics"]["n_mean"]["value"] is None  # NaN -> null
