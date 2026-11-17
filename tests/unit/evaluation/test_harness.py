"""The evaluation harness on the packaged demo corpus (SDD 14 §3)."""

from __future__ import annotations

import dataclasses
from typing import Any

import gymnasium as gym
import numpy as np
import pytest

from artigo_env.core.errors import ArtigoEnvError
from artigo_env.envs import ArtigoCoreEnv
from artigo_env.envs.core_env import INFO_KEYS
from artigo_env.evaluation import FinalRunRequiredError, evaluate
from artigo_env.evaluation.records import read_episodes, read_steps, read_summary

pytestmark = pytest.mark.fast


def demo_env(max_steps: int = 40, **kwargs: Any):
    def factory() -> ArtigoCoreEnv:
        return ArtigoCoreEnv(corpus="demo", scenario="demo", max_steps=max_steps, **kwargs)

    return factory


class FirstAllowed:
    def reset(self, obs, info):
        pass

    def act(self, obs, info, action_mask):
        return int(np.argmax(action_mask))


@dataclasses.dataclass(frozen=True)
class Card:
    name: str = "first"
    seed: int = 0

    def content_hash(self) -> str:
        return "abc123"


class Spy(FirstAllowed):
    """Records every argument the harness hands over, and vandalizes its copies."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[Any, ...]]] = []
        self.card = Card()

    def reset(self, obs, info):
        self.calls.append(("reset", (obs, info)))

    def act(self, obs, info, action_mask):
        self.calls.append(("act", (obs, info, action_mask)))
        action = int(np.argmax(action_mask))
        for value in obs.values():
            value[...] = 0
        info.clear()
        action_mask[...] = False
        return action

    def observe(self, transition):
        self.calls.append(("observe", (transition,)))


def walk(value: Any):
    yield value
    if isinstance(value, dict):
        for v in value.values():
            yield from walk(v)
    elif isinstance(value, list | tuple):
        for v in value:
            yield from walk(v)


@pytest.mark.req("REQ-EVL-006", "REQ-AGT-002")
def test_spy_agent_receives_only_observation_filtered_info_and_mask(tmp_path):
    spy = Spy()
    baseline = evaluate(FirstAllowed(), demo_env(), "demo", [0], limit=2, out_dir=tmp_path / "a")
    result = evaluate(spy, demo_env(mask_in_info=True), "demo", [0], limit=2, out_dir=tmp_path)
    obs_keys = set(ArtigoCoreEnv().observation_space.spaces)
    kinds = [kind for kind, _ in spy.calls]
    assert kinds.count("reset") == 2
    assert kinds.count("act") == kinds.count("observe") == len(result.steps)
    for kind, args in spy.calls:
        for value in walk(args):
            assert not isinstance(value, gym.Env)
            assert isinstance(value, dict | list | tuple | np.ndarray | str | int | float | bool)
        if kind == "act":
            obs, info, mask = args
            assert set(obs) == obs_keys
            assert set(info) <= INFO_KEYS
            assert mask.dtype == np.bool_
        if kind == "observe":
            assert set(args[0]) == {
                "obs",
                "action",
                "reward",
                "next_obs",
                "terminated",
                "truncated",
                "info",
            }
            assert set(args[0]["info"]) <= INFO_KEYS
    # vandalizing the copies changed nothing in the episodes
    assert [e.N for e in result.episodes] == [e.N for e in baseline.episodes]
    assert [s.action for s in result.steps] == [s.action for s in baseline.steps]


@pytest.mark.req("REQ-EVL-001")
def test_fixed_list_in_fixed_order_for_every_seed(tmp_path):
    result = evaluate(FirstAllowed(), demo_env(), "demo", [0, 5], limit=3, out_dir=tmp_path)
    assert [(e.seed, e.article_index) for e in result.episodes] == [
        (s, i) for s in (0, 5) for i in range(3)
    ]
    refs = [e.article_ref for e in result.episodes]
    assert refs[:3] == refs[3:] and len(set(refs[:3])) == 3
    again = evaluate(FirstAllowed(), demo_env(), "demo", [0, 5], limit=3, out_dir=tmp_path / "b")
    assert [e.to_dict() for e in again.episodes] == [e.to_dict() for e in result.episodes]
    assert again.run_id == result.run_id


@pytest.mark.req("REQ-EVL-002")
def test_truncated_episodes_count_t_max(tmp_path):
    result = evaluate(FirstAllowed(), demo_env(max_steps=5), "demo", [0], limit=2, out_dir=tmp_path)
    for e in result.episodes:
        assert e.truncated and not e.won and e.N == 5 and e.efficiency == 0.0
    assert result.summary["metrics"]["n_mean"]["value"] == 5.0
    assert result.summary["t_max"] == 5


@pytest.mark.req("REQ-EVL-005", "REQ-AGT-001", "REQ-REPO-004")
def test_files_card_and_identifiers(tmp_path):
    result = evaluate(Spy(), demo_env(), "demo", [0], limit=2, out_dir=tmp_path)
    episodes, steps = read_episodes(tmp_path), read_steps(tmp_path)
    assert episodes.num_rows == 2 and steps.num_rows == len(result.steps)
    summary = read_summary(tmp_path)
    assert summary["agent_card"] == {"name": "first", "seed": 0, "content_hash": "abc123"}
    ids = summary["identifiers"]
    assert ids["env_id"] == "ArtigoEnv-v0"
    assert ids["ruleset"]["name"] == "canonical" and len(ids["ruleset"]["hash"]) == 64
    assert ids["corpus_build"].startswith("demo-")
    assert ids["calibration_version"] == "uncalibrated"
    assert {"package_version", "git_commit", "git_dirty"} <= set(ids["code_version"])
    first = steps.to_pylist()[0]
    assert first["step"] == 1 and first["action_lemma"]
    fractions = [r["revealed_frac"] for r in steps.to_pylist() if r["episode_id"] == "s0-a00000"]
    assert fractions == sorted(fractions) and all(0.0 <= f <= 1.0 for f in fractions)


@pytest.mark.req("REQ-EVL-003")
def test_w_test_needs_a_final_run(tmp_path):
    factory = lambda: ArtigoCoreEnv(corpus="demo", scenario="eval-w-test")  # noqa: E731
    with pytest.raises(FinalRunRequiredError):
        evaluate(FirstAllowed(), factory, "eval-w-test", [0], out_dir=tmp_path)
    with pytest.raises(ArtigoEnvError, match="no article"):  # final, but the demo has no W-test
        evaluate(FirstAllowed(), factory, "eval-w-test", [0], out_dir=tmp_path, final=True)


@pytest.mark.req("REQ-EVL-001")
def test_eval_set_must_match_the_scenario(tmp_path):
    with pytest.raises(ValueError, match="does not match"):
        evaluate(FirstAllowed(), demo_env(), "eval-w-val", [0], out_dir=tmp_path)


@pytest.mark.req("REQ-AGT-003")
def test_agent_factory_is_called_once_per_seed(tmp_path):
    seen: list[int] = []

    def factory(seed: int) -> FirstAllowed:
        seen.append(seed)
        return FirstAllowed()

    evaluate(factory, demo_env(), "demo", [3, 4], limit=1, out_dir=tmp_path)
    assert seen == [3, 4]


@pytest.mark.req("REQ-EVL-001")
def test_default_output_directory_under_the_data_root(tmp_path, monkeypatch):
    monkeypatch.setenv("ARTIGO_DATA_ROOT", str(tmp_path))
    result = evaluate(FirstAllowed(), demo_env(), "demo", [0], limit=1)
    assert result.out_dir == tmp_path / "runs" / result.run_id / "eval" / "demo"
    assert (result.out_dir / "summary.json").is_file()


@pytest.mark.req("REQ-EVL-006")
def test_a_shorter_time_limit_wrapper_is_refused(tmp_path):
    factory = lambda: gym.make("ArtigoEnv-v0", max_episode_steps=3)  # noqa: E731
    with pytest.raises(ArtigoEnvError, match="truncated by a wrapper"):
        evaluate(FirstAllowed(), factory, "demo", [0], limit=1, out_dir=tmp_path)


@pytest.mark.req("REQ-EVL-006")
def test_non_integer_actions_are_refused(tmp_path):
    class Text(FirstAllowed):
        def act(self, obs, info, action_mask):
            return "gato"

    with pytest.raises(TypeError, match="integer action"):
        evaluate(Text(), demo_env(), "demo", [0], limit=1, out_dir=tmp_path)
