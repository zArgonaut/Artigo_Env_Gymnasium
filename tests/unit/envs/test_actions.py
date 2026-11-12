"""Actions, mask and invalid actions of ArtigoCoreEnv (SDD 09 §4, DEC-067, DEC-068)."""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from artigo_env.core.errors import InvalidActionError
from artigo_env.envs import ArtigoCoreEnv
from artigo_env.envs.observation import OUTCOME_CODES


@pytest.fixture
def env():
    environment = ArtigoCoreEnv()
    environment.reset(seed=0)
    yield environment
    environment.close()


@pytest.mark.fast
@pytest.mark.req("REQ-ENV-009")
def test_mask_forbids_guessed_keys(env):
    mask = env.action_masks()
    assert mask.shape == (env.action_space.n,) and mask.dtype == np.bool_ and mask.all()
    env.step(4)
    mask = env.action_masks()
    assert not mask[4] and mask.sum() == env.action_space.n - 1
    env.reset()
    assert env.action_masks().all()


@pytest.mark.fast
@pytest.mark.req("REQ-ENV-009")
def test_mask_noops_false_allows_repeats():
    env = ArtigoCoreEnv(mask_noops=False)
    env.reset(seed=0)
    env.step(4)
    assert env.action_masks().all()
    _, reward, _, _, info = env.step(4)
    assert info["outcome"] == "noop_repeat" and reward == 0.0 and not info["invalid_action"]
    env.close()


@pytest.mark.fast
@pytest.mark.req("REQ-ENV-010")
def test_strict_mode_raises_on_masked_action(env):
    env.step(4)
    with pytest.raises(InvalidActionError, match="masked"):
        env.step(4)
    with pytest.raises(InvalidActionError):
        env.step(env.action_space.n)
    with pytest.raises(InvalidActionError):
        env.step(1.5)


@pytest.mark.fast
@pytest.mark.req("REQ-ENV-010", "REQ-ENV-013")
def test_tolerant_mode_counts_a_miss():
    env = ArtigoCoreEnv(strict_actions=False)
    obs, _ = env.reset(seed=0)
    obs, _, _, _, _ = env.step(4)
    before = {k: v.copy() for k, v in obs.items()}
    obs, reward, terminated, truncated, info = env.step(4)
    assert reward == -1.0 and not terminated and not truncated
    assert info["invalid_action"] and info["counted"] and info["count"] == 0
    assert info["outcome"] == "invalid" and info["newly_revealed"] == []
    assert obs["history"][1].tolist() == [4, 0, OUTCOME_CODES["invalid"]]
    np.testing.assert_array_equal(obs["body_lemma"], before["body_lemma"])
    env.close()


@pytest.mark.fast
@pytest.mark.req("REQ-ENV-013")
def test_numeric_and_absent_lemma_actions(tmp_path):
    env = ArtigoCoreEnv()
    env.reset(seed=0)
    vocab = env._require_vocab()
    numeric = int(np.flatnonzero(env._is_numeric)[0])
    assert vocab.key(numeric).startswith("n:")
    _, reward, _, _, info = env.step(numeric)
    assert reward == -1.0 and info["counted"]
    # A key whose lemma the build does not have is a counted miss.
    env._action_lemma = env._action_lemma.copy()
    lexical = int(np.flatnonzero(~env._is_numeric)[0])
    env._action_lemma[lexical] = -1
    _, reward, _, _, info = env.step(lexical)
    assert info["outcome"] == "miss" and reward == -1.0 and info["count"] == 0
    env.close()


@pytest.mark.fast
@pytest.mark.req("REQ-ENV-009")
def test_history_records_every_step(env):
    obs, _, _, _, info = env.step(0)
    assert obs["history"][0].tolist() == [0, info["count"], OUTCOME_CODES[info["outcome"]]]
    assert (obs["history"][1:] == -1).all() and obs["step"].tolist() == [1]


_HAS_SMOKE = all(importlib.util.find_spec(m) is not None for m in ("sb3_contrib", "torch"))


@pytest.mark.slow
@pytest.mark.req("REQ-ENV-009")
@pytest.mark.skipif(not _HAS_SMOKE, reason="needs the 'smoke' group (sb3-contrib, torch)")
def test_maskable_ppo_with_subprocess_vec_env():
    from sb3_contrib import MaskablePPO
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import SubprocVecEnv

    vec = make_vec_env(
        "artigo_env:ArtigoEnv-v0",
        n_envs=2,
        seed=0,
        vec_env_cls=SubprocVecEnv,
        vec_env_kwargs={"start_method": "spawn"},
        env_kwargs={"max_steps": 64, "render_mode": None},  # SB3 defaults to rgb_array
        wrapper_kwargs=None,
    )
    try:
        masks = vec.env_method("action_masks")
        assert len(masks) == 2 and all(m.all() for m in masks)
        model = MaskablePPO(
            "MultiInputPolicy", vec, n_steps=64, batch_size=64, n_epochs=1, seed=0, device="cpu"
        )
        model.learn(total_timesteps=256)
    finally:
        vec.close()
