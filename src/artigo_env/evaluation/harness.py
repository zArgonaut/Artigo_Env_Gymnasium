"""Minimal evaluation harness (SDD 14 §3, REQ-EVL-001/002/003/005/006, REQ-AGT-001/002).

``evaluate`` plays one episode per article of a fixed evaluation list, in list order, for each
evaluation seed (``reset(seed=s, options={"article_index": i})``). The agent receives only what
a player may see: a copy of the observation, a copy of ``info`` filtered to the environment's
allow-list and the action mask; never the environment. After each episode the harness reads
``episode_record()`` (hidden data, harness only) and computes the measures of ``metrics``. The
trajectories go to Parquet (``records``) with ``summary.json``.

Agents are any object with ``reset(obs, info)`` and ``act(obs, info, action_mask) -> int``
(``EvalAgent``; the agents layer is not imported here), optionally ``observe(transition)`` and a
``card`` (written to the summary, REQ-AGT-001). ``agent`` may also be a callable
``seed -> agent``, called once per evaluation seed, so seeded agents restart with each seed.
"""

from __future__ import annotations

import copy
import dataclasses
import json
import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

import artigo_env
from artigo_env.core.config import canonical_json, sha256_hex
from artigo_env.core.errors import ArtigoEnvError
from artigo_env.core.paths import data_path
from artigo_env.envs.core_env import INFO_KEYS, ArtigoCoreEnv
from artigo_env.envs.observation import OBS_PARENTHESES, OBS_STOPWORD, OBS_VISIBLE
from artigo_env.envs.record import EpisodeNotFinishedError, EpisodeRecord
from artigo_env.envs.registration import ENV_ID
from artigo_env.evaluation.metrics import (
    BOOTSTRAP_RESAMPLES,
    EpisodeMetrics,
    counted_guesses,
    efficiency,
    first_title_hit,
    hit_rate,
    reveal_auc,
    summarize,
)
from artigo_env.evaluation.records import StepRow, write_outputs

CALIBRATION_VERSION = "uncalibrated"  # no calibrated ruleset exists yet (F6)
FINAL_ONLY_SPLITS = frozenset({"w-test"})

Obs = Mapping[str, npt.NDArray[Any]]


@runtime_checkable
class EvalAgent(Protocol):
    """What the harness needs from an agent (structural; SDD 12 §1 is the full protocol)."""

    def reset(self, obs: Obs, info: Mapping[str, Any]) -> None: ...

    def act(
        self, obs: Obs, info: Mapping[str, Any], action_mask: npt.NDArray[np.bool_] | None
    ) -> int: ...


AgentOrFactory = EvalAgent | Callable[[int], EvalAgent]


class FinalRunRequiredError(ArtigoEnvError):
    """A held-out evaluation set (W-test) was requested outside a final run (REQ-EVL-003)."""


@dataclass(frozen=True)
class EvalResult:
    run_id: str
    eval_set: str
    out_dir: Path
    episodes: tuple[EpisodeMetrics, ...]
    steps: tuple[StepRow, ...]
    summary: Mapping[str, Any]


# -- identifiers ----------------------------------------------------------------------------------


def code_version() -> dict[str, Any]:
    """Package version, plus git commit and dirty flag when running from a source checkout."""
    info: dict[str, Any] = {
        "package_version": artigo_env.__version__,
        "git_commit": None,
        "git_dirty": None,
    }
    root = Path(artigo_env.__file__).resolve().parents[2]
    if not (root / ".git").exists():
        return info

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(root), *args],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout.strip()

    try:
        info["git_commit"] = git("rev-parse", "HEAD")
        info["git_dirty"] = bool(git("status", "--porcelain", "--untracked-files=no"))
    except (OSError, subprocess.SubprocessError):
        pass
    return info


def card_to_dict(card: Any) -> dict[str, Any] | None:
    """Plain JSON-ready form of an agent card (dataclass, mapping or object)."""
    if card is None:
        return None
    if dataclasses.is_dataclass(card) and not isinstance(card, type):
        raw: Any = {f.name: getattr(card, f.name) for f in dataclasses.fields(card)}
    elif isinstance(card, Mapping):
        raw = dict(card)
    else:
        raw = {k: v for k, v in vars(card).items() if not k.startswith("_")}
    try:
        data: dict[str, Any] = json.loads(canonical_json(raw))
    except (TypeError, ValueError):
        data = json.loads(json.dumps(raw, default=repr))
    hasher = getattr(card, "content_hash", None)
    if callable(hasher):
        data["content_hash"] = str(hasher())
    return data


def make_run_id(inputs: Mapping[str, Any]) -> str:
    """Deterministic run id: ``eval-`` plus 12 hex characters of the inputs' canonical JSON."""
    return "eval-" + sha256_hex(json.dumps(inputs, sort_keys=True, default=repr))[:12]


# -- helpers --------------------------------------------------------------------------------------


def _core(env: Any) -> ArtigoCoreEnv:
    core = getattr(env, "unwrapped", env)
    if not isinstance(core, ArtigoCoreEnv):
        raise TypeError(f"env_factory must build an ArtigoCoreEnv, got {type(core).__name__}")
    return core


def _n_articles(core: ArtigoCoreEnv) -> int:
    """Length of the scenario's article list at the current curriculum stage."""
    masks = core._stage_masks  # same lists as the fixed-list sampler builds
    return int(np.count_nonzero(masks[core.curriculum_stage]))


def filter_info(info: Mapping[str, Any]) -> dict[str, Any]:
    """Deep copy of the allow-listed ``info`` keys (SDD 09 §6)."""
    return {k: copy.deepcopy(v) for k, v in info.items() if k in INFO_KEYS}


def _copy_obs(obs: Obs) -> dict[str, npt.NDArray[Any]]:
    return {k: np.array(v, copy=True) for k, v in obs.items()}


def _hidden_body(obs: Obs) -> npt.NDArray[np.bool_]:
    n_body = int(obs["n_tokens"][0])
    flags = obs["body_flags"][:n_body]
    hidden: npt.NDArray[np.bool_] = (flags & OBS_VISIBLE) == 0
    return hidden


def _revealed_fraction(obs: Obs, hidden_at_start: npt.NDArray[np.bool_]) -> float:
    total = int(hidden_at_start.sum())
    if total == 0:
        return 1.0
    n_body = len(hidden_at_start)
    visible = (obs["body_flags"][:n_body] & OBS_VISIBLE) != 0
    return float(np.count_nonzero(visible & hidden_at_start)) / total


def _title_lemmas(obs: Obs, n_actions: int, ignore_parenthetical: bool) -> int:
    """Distinct lemmas of V among the visible required title tokens (all of them when won)."""
    n_title = int(obs["n_tokens"][1])
    flags = obs["title_flags"][:n_title]
    ids = obs["title_lemma"][:n_title]
    required = (flags & OBS_STOPWORD) == 0
    if ignore_parenthetical:
        required &= (flags & OBS_PARENTHESES) == 0
    usable = required & ((flags & OBS_VISIBLE) != 0) & (ids >= 0) & (ids < n_actions)
    return len(set(ids[usable].tolist()))


def _action_lemmas(record: EpisodeRecord) -> list[str | None]:
    """Text of each step's guess from the engine history (``None`` for invalid steps)."""
    history = list(record.final_state.get("history", ()))
    played = [o for o in record.outcomes if o != "invalid"]
    if len(played) != len(history):
        return [None] * len(record.outcomes)
    out: list[str | None] = []
    entries = iter(history)
    for outcome in record.outcomes:
        if outcome == "invalid":
            out.append(None)
            continue
        entry = next(entries)
        key = str(entry.get("key", ""))
        out.append(key if key.startswith("n:") else str(entry.get("form", "")))
    return out


def _agent_for(agent: AgentOrFactory, seed: int) -> EvalAgent:
    if hasattr(agent, "act") and hasattr(agent, "reset"):
        return agent
    if callable(agent):
        built = agent(seed)
        if not (hasattr(built, "act") and hasattr(built, "reset")):
            raise TypeError("the agent factory must return an object with reset() and act()")
        return built
    raise TypeError("agent must have reset() and act(), or be a callable seed -> agent")


def _action(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int | np.integer):
        raise TypeError(f"act() must return an integer action of V, got {value!r}")
    return int(value)


# -- evaluation -----------------------------------------------------------------------------------


def evaluate(
    agent: AgentOrFactory,
    env_factory: Callable[[], Any],
    eval_set: str,
    seeds: Sequence[int],
    deterministic: bool = True,
    *,
    limit: int | None = None,
    out_dir: str | Path | None = None,
    final: bool = False,
    resamples: int = BOOTSTRAP_RESAMPLES,
) -> EvalResult:
    """Evaluate ``agent`` on the fixed list of scenario ``eval_set`` (see the module docstring).

    ``env_factory()`` must build an ``ArtigoCoreEnv`` (possibly wrapped) whose scenario is named
    ``eval_set``; wrappers must not truncate before its ``max_steps`` (``T_max``). ``limit``
    keeps the first N articles. ``out_dir`` defaults to
    ``ARTIGO_DATA_ROOT/runs/<run_id>/eval/<eval_set>/``. ``deterministic`` is recorded (agents
    choose their own mode). Held-out sets (W-test) require ``final=True`` (REQ-EVL-003).
    """
    seeds = [int(s) for s in seeds]
    if not seeds:
        raise ValueError("at least one evaluation seed is required")
    if limit is not None and (isinstance(limit, bool) or int(limit) < 1):
        raise ValueError(f"limit must be a positive integer, got {limit!r}")

    env = env_factory()
    try:
        core = _core(env)
        scenario = core.scenario
        if scenario.name != eval_set:
            raise ValueError(
                f"eval_set {eval_set!r} does not match the environment's scenario {scenario.name!r}"
            )
        if scenario.split in FINAL_ONLY_SPLITS and not final:
            raise FinalRunRequiredError(
                f"scenario {eval_set!r} uses the held-out split {scenario.split!r}; it may only "
                "be evaluated in a run marked final (REQ-EVL-003)"
            )
        t_max = int(core.max_steps)
        n_actions = len(core.action_masks())  # |V|
        ignore_parenthetical = core.ruleset.title_parenthetical == "ignored"

        first_agent = _agent_for(agent, seeds[0])
        card = card_to_dict(getattr(first_agent, "card", None))
        identity = {k: v for k, v in (card or {}).items() if k not in ("seed", "content_hash")}
        ruleset_hash = core.ruleset.content_hash()
        identifiers: dict[str, Any] = {
            "env_id": ENV_ID,
            "ruleset": {"name": core.ruleset.name, "hash": ruleset_hash},
            "corpus_build": core.corpus_build,
            "calibration_version": CALIBRATION_VERSION,
            "code_version": code_version(),
        }
        run_inputs = {
            "agent": identity or type(first_agent).__name__,
            "env_id": ENV_ID,
            "ruleset_hash": ruleset_hash,
            "corpus_build": core.corpus_build,
            "vocab_id": core.vocab_id,
            "eval_set": eval_set,
            "seeds": seeds,
            "limit": limit,
            "deterministic": bool(deterministic),
            "t_max": t_max,
        }
        run_id = make_run_id(run_inputs)
        target = (
            Path(out_dir) if out_dir is not None else data_path("runs", run_id, "eval", eval_set)
        )

        n_articles = _n_articles(core)
        if limit is not None:
            n_articles = min(n_articles, int(limit))
        if n_articles == 0:
            raise ArtigoEnvError(f"scenario {eval_set!r} has no article in this corpus build")

        episodes: list[EpisodeMetrics] = []
        steps: list[StepRow] = []
        for seed in seeds:
            player = first_agent if seed == seeds[0] else _agent_for(agent, seed)
            for index in range(n_articles):
                metrics, rows = _play(
                    env, core, player, seed, index, t_max, n_actions, ignore_parenthetical
                )
                episodes.append(metrics)
                steps.extend(rows)
    finally:
        env.close()

    summary: dict[str, Any] = {
        "format_version": 1,
        "run_id": run_id,
        "eval_set": eval_set,
        "scenario_split": scenario.split,
        "seeds": seeds,
        "limit": limit,
        "deterministic": bool(deterministic),
        "final": bool(final),
        "t_max": t_max,
        "vocab_id": core.vocab_id,
        "identifiers": identifiers,
        "agent_card": card,
    }
    summary.update(summarize(episodes, resamples=resamples))
    write_outputs(target, steps, episodes, summary)
    return EvalResult(run_id, eval_set, target, tuple(episodes), tuple(steps), summary)


def _play(
    env: Any,
    core: ArtigoCoreEnv,
    agent: EvalAgent,
    seed: int,
    index: int,
    t_max: int,
    n_actions: int,
    ignore_parenthetical: bool,
) -> tuple[EpisodeMetrics, list[StepRow]]:
    obs, info = env.reset(seed=seed, options={"article_index": index})
    hidden = _hidden_body(obs)
    agent.reset(_copy_obs(obs), filter_info(info))
    observe = getattr(agent, "observe", None)
    rewards: list[float] = []
    revealed: list[float] = []
    title: list[float] = []
    actions: list[int] = []
    terminated = truncated = False
    while not (terminated or truncated):
        mask = core.action_masks()
        action = _action(agent.act(_copy_obs(obs), filter_info(info), mask.copy()))
        next_obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(float(reward))
        actions.append(action)
        revealed.append(_revealed_fraction(next_obs, hidden))
        title.append(float(info["title_progress"]["tokens"]))
        if callable(observe):
            observe(
                {
                    "obs": _copy_obs(obs),
                    "action": action,
                    "reward": float(reward),
                    "next_obs": _copy_obs(next_obs),
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                    "info": filter_info(info),
                }
            )
        obs = next_obs
    try:
        record = core.episode_record()
    except EpisodeNotFinishedError as exc:
        raise ArtigoEnvError(
            "the environment was truncated by a wrapper before ArtigoCoreEnv.max_steps; build "
            "the evaluation environment without a shorter TimeLimit"
        ) from exc

    episode_id = f"s{seed}-a{index:05d}"
    n = counted_guesses(record.n_counted, record.truncated, t_max)
    lemmas = _action_lemmas(record)
    rows = [
        StepRow(
            episode_id=episode_id,
            article_ref=record.article_ref,
            seed=seed,
            step=i + 1,
            action_lemma=lemmas[i],
            outcome=record.outcomes[i],
            count=record.counts[i],
            reward_total=rewards[i],
            revealed_frac=revealed[i],
            title_progress=title[i],
            action=actions[i],
        )
        for i in range(len(record.actions))
    ]
    metrics = EpisodeMetrics(
        episode_id=episode_id,
        seed=seed,
        article_index=index,
        article_ref=record.article_ref,
        source=record.source_id,
        domain=record.domain,
        n_title_content=record.n_title_content,
        n_body_tokens=record.n_body_tokens,
        n_steps=len(record.actions),
        N=n,
        won=record.won,
        truncated=record.truncated,
        title_oov=record.title_oov,
        efficiency=efficiency(_title_lemmas(obs, n_actions, ignore_parenthetical), n, record.won),
        hit_rate=hit_rate(record.counts),
        reveal_auc=reveal_auc(revealed),
        first_title_hit=first_title_hit(title),
        reward_total=float(sum(rewards)),
    )
    return metrics, rows


# -- display --------------------------------------------------------------------------------------

_LABELS = (
    ("n_mean", "N mean (main)"),
    ("win_rate", "win rate"),
    ("n_median_won", "N median (won)"),
    ("n_p90_won", "N P90 (won)"),
    ("n_median_all", "N median (all)"),
    ("n_p90_all", "N P90 (all)"),
    ("efficiency_mean", "efficiency"),
    ("hit_rate_mean", "hit rate"),
    ("reveal_auc_mean", "reveal AUC"),
    ("title_coverage", "title coverage"),
)


def _fmt(x: Any) -> str:
    return "-" if x is None else f"{x:.3f}"


def summary_table(summary: Mapping[str, Any]) -> str:
    """Human-readable table of a ``summary.json`` mapping."""
    ids = summary["identifiers"]
    lines = [
        f"run {summary['run_id']}  eval set {summary['eval_set']}  seeds {summary['seeds']}  "
        f"T_max {summary['t_max']}",
        f"{ids['env_id']}  ruleset {ids['ruleset']['name']}  corpus {ids['corpus_build']}  "
        f"episodes {summary['n_episodes']} ({summary['n_playable_episodes']} playable)",
        "",
        f"{'metric':<18} {'value':>10} {'95% CI':>22}",
    ]
    for key, label in _LABELS:
        m = summary["metrics"][key]
        ci = f"[{_fmt(m['ci_low'])}, {_fmt(m['ci_high'])}]"
        lines.append(f"{label:<18} {_fmt(m['value']):>10} {ci:>22}")
    oov = summary["title_oov"]
    if oov["n_episodes"]:
        lines.append(
            f"title_oov episodes: {oov['n_episodes']} (N mean {_fmt(oov['n_mean'])}, "
            f"win rate {_fmt(oov['win_rate'])}), excluded from the metrics above"
        )
    return "\n".join(lines)
