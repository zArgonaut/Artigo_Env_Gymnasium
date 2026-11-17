"""A0: random guesses weighted by ``freq + 1`` (SDD 12 §2)."""

from __future__ import annotations

import numpy as np
import pytest

from artigo_env.agents import make_agent

pytestmark = pytest.mark.fast

FREQ = [0, 1, 3, 9, 19, 7]


def play(agent, obs, mask, n):
    return [agent.act(obs, {}, mask) for _ in range(n)]


@pytest.mark.req("REQ-AGT-003")
def test_distribution_matches_freq_plus_one(artifacts, make_obs):
    vocab, knowledge = artifacts(FREQ)
    agent = make_agent("a0", vocab=vocab, knowledge=knowledge, ruleset="canonical", seed=11)
    obs = make_obs((5,), (5,))
    agent.reset(obs, {})
    mask = np.array([True, True, True, True, True, False])
    n = 40_000
    draws = np.bincount(play(agent, obs, mask, n), minlength=len(FREQ))
    assert draws[5] == 0
    weights = np.where(mask, np.asarray(FREQ) + 1.0, 0.0)
    expected = weights / weights.sum()
    np.testing.assert_allclose(draws / n, expected, atol=0.01)


@pytest.mark.req("REQ-AGT-003")
def test_deterministic_for_a_seed_also_after_reset(artifacts, make_obs):
    vocab, knowledge = artifacts(FREQ)
    obs = make_obs((5,), (5,))
    mask = np.ones(len(FREQ), dtype=bool)

    def fresh(seed):
        agent = make_agent("a0", vocab=vocab, knowledge=knowledge, ruleset="canonical", seed=seed)
        agent.reset(obs, {})
        return agent

    a, b = fresh(5), fresh(5)
    first = play(a, obs, mask, 200)
    assert first == play(b, obs, mask, 200)
    a.reset(obs, {})
    assert play(a, obs, mask, 200) == first
    assert play(fresh(6), obs, mask, 200) != first


@pytest.mark.req("REQ-AGT-003")
def test_never_draws_a_masked_action(artifacts, make_obs):
    vocab, knowledge = artifacts([1000, 0, 0, 1000])
    agent = make_agent("a0", vocab=vocab, knowledge=knowledge, ruleset="canonical", seed=1)
    obs = make_obs((5,), (5,))
    mask = np.array([False, True, False, False])
    assert set(play(agent, obs, mask, 500)) == {1}
