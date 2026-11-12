"""``episode_record()`` of ArtigoCoreEnv (SDD 09 §6, REQ-ENV-012)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from artigo_env.envs import ArtigoCoreEnv, EpisodeNotFinishedError


@pytest.mark.fast
@pytest.mark.req("REQ-ENV-012")
def test_record_fails_before_the_end_and_works_after():
    env = ArtigoCoreEnv(max_steps=6, strict_actions=False)
    with pytest.raises(EpisodeNotFinishedError):
        env.episode_record()
    env.reset(seed=9)
    with pytest.raises(EpisodeNotFinishedError):
        env.episode_record()
    numeric = np.flatnonzero(env._is_numeric)[:5].tolist()
    for action in [*numeric, numeric[0]]:
        _, _, terminated, truncated, _ = env.step(action)
    assert truncated and not terminated
    record = env.episode_record()
    assert record.actions == (*numeric, numeric[0])
    assert record.outcomes[-1] == "invalid" and record.counted == (True,) * 6
    assert record.n_counted == 6 and not record.won and record.truncated
    assert record.split == "demo" and record.source_id == "demo" and record.domain == "demo"
    assert record.title and record.body and record.n_body_tokens == len(env._article.body)
    assert record.n_title_content >= 1 and not record.title_oov
    assert record.final_state["n_counted"] == 5
    json.dumps(record.to_dict())
    env.reset()
    with pytest.raises(EpisodeNotFinishedError):
        env.episode_record()
    env.close()


@pytest.mark.fast
@pytest.mark.req("REQ-ENV-012")
def test_record_after_victory():
    env = ArtigoCoreEnv()
    env.reset(seed=3)
    terminated = False
    while not terminated:
        _, _, terminated, truncated, _ = env.step(int(np.flatnonzero(env.action_masks())[0]))
        assert not truncated
    record = env.episode_record()
    assert record.won and not record.truncated and record.final_state["status"] == "won"
    assert len(record.actions) == len(record.outcomes) == len(record.counts)
    env.close()
