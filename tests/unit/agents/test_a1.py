"""A1: greedy by training frequency (SDD 12 §2)."""

from __future__ import annotations

import numpy as np
import pytest

from artigo_env.agents import NoActionAvailableError, make_agent

pytestmark = pytest.mark.fast


@pytest.mark.req("REQ-AGT-003")
def test_picks_most_frequent_allowed_action(artifacts, make_obs):
    vocab, knowledge = artifacts([4, 9, 9, 1, 7])
    agent = make_agent("a1", vocab=vocab, knowledge=knowledge, ruleset="canonical")
    obs = make_obs((5,), (5,))
    agent.reset(obs, {})
    assert agent.act(obs, {}, None) == 1  # tie 1/2 broken by the smallest id
    mask = np.array([True, False, True, True, True])
    assert agent.act(obs, {}, mask) == 2
    mask = np.array([True, False, False, True, True])
    assert agent.act(obs, {}, mask) == 4
    mask = np.array([False, False, False, True, False])
    assert agent.act(obs, {}, mask) == 3
    with pytest.raises(NoActionAvailableError):
        agent.act(obs, {}, np.zeros(5, dtype=bool))


@pytest.mark.req("REQ-AGT-003")
def test_never_picks_a_masked_action(artifacts, make_obs):
    rng = np.random.default_rng(0)
    freq = rng.integers(0, 50, size=200).tolist()
    vocab, knowledge = artifacts(freq)
    agent = make_agent("a1", vocab=vocab, knowledge=knowledge, ruleset="canonical")
    obs = make_obs((5,), (5,))
    for _ in range(50):
        mask = rng.random(200) < 0.3
        mask[rng.integers(0, 200)] = True
        action = agent.act(obs, {}, mask)
        assert mask[action]
        assert freq[action] == max(f for f, m in zip(freq, mask, strict=True) if m)
