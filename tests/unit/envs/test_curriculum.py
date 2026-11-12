"""Curriculum stage of ArtigoCoreEnv (SDD 08 §B.3, REQ-SCN-007)."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest

from artigo_env.envs import ArtigoCoreEnv
from artigo_env.scenarios import CurriculumStage, FilterSpec, Scenario

SHORT = 225  # body tokens (demo bodies have 202..260)


def _scenario(until_steps=None):
    return Scenario(
        name="demo-curriculum",
        split="demo",
        curriculum=(
            CurriculumStage(
                filters=(FilterSpec("body_tokens", {"max": SHORT}),), until_steps=until_steps
            ),
            CurriculumStage(),
        ),
    )


def _bodies(env, n=8):
    sizes = []
    for _ in range(n):
        env.reset()
        sizes.append(len(env._article.body))
    return sizes


@pytest.mark.fast
@pytest.mark.req("REQ-SCN-007")
def test_set_curriculum_stage_changes_the_drawn_documents():
    env = ArtigoCoreEnv(scenario=_scenario())
    _, info = env.reset(seed=0)
    assert info["curriculum_stage"] == 0
    assert all(n <= SHORT for n in _bodies(env))
    env.set_curriculum_stage(1)
    sizes = _bodies(env, 30)
    assert any(n > SHORT for n in sizes)
    _, info = env.reset(seed=1)  # reseeding keeps the stage
    assert info["curriculum_stage"] == 1
    _, info = env.reset(options={"curriculum_stage": 0})
    assert info["curriculum_stage"] == 0 and len(env._article.body) <= SHORT
    with pytest.raises(ValueError):
        env.set_curriculum_stage(2)
    env.close()


@pytest.mark.fast
@pytest.mark.req("REQ-SCN-007")
def test_until_steps_advances_inside_the_environment():
    env = ArtigoCoreEnv(scenario=_scenario(until_steps=3))
    env.reset(seed=0)
    for a in np.flatnonzero(env._is_numeric)[:3].tolist():
        _, _, _, _, info = env.step(a)
    assert info["curriculum_stage"] == 1
    env.close()


@pytest.mark.fast
@pytest.mark.req("REQ-SCN-007")
def test_article_index_option():
    env = ArtigoCoreEnv()
    env.reset(seed=0, options={"article_index": 4})
    assert env._doc_idx == 4
    with pytest.raises(IndexError):
        env.reset(options={"article_index": 30})
    with pytest.raises(ValueError, match="unknown reset options"):
        env.reset(options={"doc": 1})
    env.close()


@pytest.mark.slow
@pytest.mark.req("REQ-SCN-007")
def test_stage_through_a_subprocess_vector_env():
    def factory():
        return gym.make("artigo_env:ArtigoEnv-v0", scenario=_scenario())

    vec = gym.vector.AsyncVectorEnv([factory, factory], context="spawn")
    try:
        vec.reset(seed=0)
        assert vec.get_attr("curriculum_stage") == (0, 0)
        vec.call("set_curriculum_stage", 1)
        assert vec.get_attr("curriculum_stage") == (1, 1)
        _, infos = vec.reset()
        assert list(infos["curriculum_stage"]) == [1, 1]
    finally:
        vec.close()
