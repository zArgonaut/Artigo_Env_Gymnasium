"""``info`` allow-list of ArtigoCoreEnv (SDD 09 §6, DEC-070)."""

from __future__ import annotations

import numpy as np
import pytest

from artigo_env.envs import ArtigoCoreEnv
from artigo_env.envs.core_env import INFO_KEYS

STEP_KEYS = {"outcome", "count", "counted", "newly_revealed", "invalid_action"}
FORBIDDEN_WORDS = ("source", "domain", "split", "doc", "text", "url", "article")


def _episode(env, seed, n=60):
    obs, info = env.reset(seed=seed)
    yield obs, info
    rng = np.random.default_rng(seed)
    for _ in range(n):
        action = int(rng.choice(np.flatnonzero(env.action_masks())))
        obs, _, terminated, truncated, info = env.step(action)
        yield obs, info
        if terminated or truncated:
            return


@pytest.mark.fast
@pytest.mark.req("REQ-ENV-011", "REQ-SCN-006")
@pytest.mark.parametrize("preset", ["canonical", "no-lengths"])
def test_info_keys_are_allow_listed_at_every_step(preset):
    env = ArtigoCoreEnv(ruleset=preset)
    for i, (obs, info) in enumerate(_episode(env, seed=2)):
        assert set(info) <= INFO_KEYS
        assert (set(info) >= STEP_KEYS) == (i > 0)
        assert info["ruleset"] == preset and info["corpus_build"] == env.corpus_build
        assert info["env_version"] == "ArtigoEnv-v0" and info["curriculum_stage"] == 0
        assert set(info["screen"]) == {"forms", "partial"}
        assert set(info["title_progress"]) == {"tokens", "letters"}
        if preset == "no-lengths":
            assert info["title_progress"]["letters"] is None
        for key in list(info) + list(obs):
            assert not any(word in key for word in FORBIDDEN_WORDS), key
    env.close()


@pytest.mark.fast
@pytest.mark.req("REQ-ENV-011")
def test_mask_only_in_info_when_requested():
    env = ArtigoCoreEnv(mask_in_info=True)
    _, info = env.reset(seed=0)
    np.testing.assert_array_equal(info["action_mask"], env.action_masks())
    _, _, _, _, info = env.step(3)
    assert not info["action_mask"][3]
    env.close()
    env = ArtigoCoreEnv()
    _, info = env.reset(seed=0)
    assert "action_mask" not in info
    env.close()


@pytest.mark.fast
@pytest.mark.req("REQ-SCN-006", "REQ-ENV-007")
def test_info_and_observation_do_not_name_the_article():
    env = ArtigoCoreEnv()
    _, info = env.reset(seed=4)
    text = repr(info)
    article = env._article
    title_words = [s for s, stop in zip(article.title.surface, article.title.stop, strict=True)]
    for word, stop in zip(title_words, article.title.stop, strict=True):
        if not stop:
            assert word not in text
    assert article.article_ref not in text
    env.close()
