"""Trajectory files of an evaluation (SDD 14 §6, DEC-094): schemas, writers and readers.

``steps.parquet`` has one row per step and ``episodes.parquet`` one row per episode, with the
columns of SDD 14 §6 in that order, followed by a few extra columns (marked below). Columns that
nothing produces yet (``generator``, reward components) are present and null so that later
phases fill them without changing the schema. Reward components are a ``map<string, double>``
(a Parquet struct needs its fields up front, and the reward families are not defined yet).
``summary.json`` holds the metrics, the agent card and the five identifiers of SDD 02 §B.5.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from artigo_env.evaluation.metrics import EpisodeMetrics

EPISODES_FILE = "episodes.parquet"
STEPS_FILE = "steps.parquet"
SUMMARY_FILE = "summary.json"
SUMMARY_FORMAT_VERSION = 1

REWARD_COMPONENTS = pa.map_(pa.string(), pa.float64())

STEPS_SCHEMA = pa.schema(
    [
        pa.field("episode_id", pa.string(), nullable=False),
        pa.field("article_ref", pa.string(), nullable=False),
        pa.field("seed", pa.int64(), nullable=False),
        pa.field("step", pa.int32(), nullable=False),  # 1-based
        pa.field("action_lemma", pa.string()),
        pa.field("generator", pa.string()),  # null until candidate generators exist (F5)
        pa.field("outcome", pa.string(), nullable=False),
        pa.field("count", pa.int32(), nullable=False),
        pa.field("reward_total", pa.float64(), nullable=False),
        pa.field("reward_components", REWARD_COMPONENTS),  # null until reward wrappers exist
        pa.field("revealed_frac", pa.float64(), nullable=False),
        pa.field("title_progress", pa.float64(), nullable=False),
        # extra: the action id in V
        pa.field("action", pa.int64(), nullable=False),
    ]
)

EPISODES_SCHEMA = pa.schema(
    [
        pa.field("episode_id", pa.string(), nullable=False),
        pa.field("article_ref", pa.string(), nullable=False),
        pa.field("source", pa.string(), nullable=False),
        pa.field("domain", pa.string()),
        pa.field("n_title_content", pa.int32(), nullable=False),
        pa.field("n_body_tokens", pa.int32(), nullable=False),
        pa.field("N", pa.int32(), nullable=False),
        pa.field("won", pa.bool_(), nullable=False),
        pa.field("truncated", pa.bool_(), nullable=False),
        pa.field("title_oov", pa.bool_(), nullable=False),
        pa.field("efficiency", pa.float64(), nullable=False),
        pa.field("reveal_auc", pa.float64(), nullable=False),
        pa.field("first_title_hit", pa.int32()),
        pa.field("reward_component_sums", REWARD_COMPONENTS),  # null until reward wrappers exist
        # extras: pairing keys and behaviour
        pa.field("seed", pa.int64(), nullable=False),
        pa.field("article_index", pa.int32(), nullable=False),
        pa.field("n_steps", pa.int32(), nullable=False),
        pa.field("hit_rate", pa.float64(), nullable=False),
        pa.field("reward_total", pa.float64(), nullable=False),
    ]
)


@dataclass(frozen=True)
class StepRow:
    """One row of ``steps.parquet``."""

    episode_id: str
    article_ref: str
    seed: int
    step: int
    action_lemma: str | None
    outcome: str
    count: int
    reward_total: float
    revealed_frac: float
    title_progress: float
    action: int
    generator: str | None = None
    reward_components: Mapping[str, float] | None = None


def steps_table(rows: Sequence[StepRow]) -> pa.Table:
    data = [asdict(r) for r in rows]
    for row in data:
        if row["reward_components"] is not None:
            row["reward_components"] = list(row["reward_components"].items())
    return pa.Table.from_pylist(data, schema=STEPS_SCHEMA)


def episodes_table(episodes: Sequence[EpisodeMetrics]) -> pa.Table:
    data = []
    for e in episodes:
        row = e.to_dict()
        row["reward_component_sums"] = None
        data.append(row)
    return pa.Table.from_pylist(data, schema=EPISODES_SCHEMA)


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(v) for v in value]
    return value


def write_outputs(
    out_dir: Path,
    steps: Sequence[StepRow],
    episodes: Sequence[EpisodeMetrics],
    summary: Mapping[str, Any],
) -> None:
    """Write the two Parquet files and ``summary.json`` into ``out_dir`` (created if needed)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(steps_table(steps), out_dir / STEPS_FILE)
    pq.write_table(episodes_table(episodes), out_dir / EPISODES_FILE)
    text = json.dumps(_json_safe(summary), ensure_ascii=False, indent=2, sort_keys=True)
    (out_dir / SUMMARY_FILE).write_text(text + "\n", encoding="utf-8")


def read_summary(out_dir: Path) -> dict[str, Any]:
    data: dict[str, Any] = json.loads((out_dir / SUMMARY_FILE).read_text(encoding="utf-8"))
    return data


def read_episodes(out_dir: Path) -> pa.Table:
    return pq.read_table(out_dir / EPISODES_FILE)


def read_steps(out_dir: Path) -> pa.Table:
    return pq.read_table(out_dir / STEPS_FILE)
