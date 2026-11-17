"""A2: deductive expected-value player, minimal F4 version (SDD 12 §2, F4 plan WP4.4)."""

from __future__ import annotations

import numpy as np
import pytest

from artigo_env.agents import make_agent
from artigo_env.agents.a2_deductive import gap_mask

pytestmark = pytest.mark.fast


def z(values, subset):
    sample = values[subset]
    return (values - sample.mean()) / sample.std()


@pytest.mark.req("REQ-AGT-003")
def test_gap_mask_reads_hidden_positions_with_a_length(make_obs, lmask):
    obs = make_obs((4, 70, 0), (2,), visible_title=(1,))
    assert gap_mask(obs["title_lemma"], obs["title_length"]) == lmask(4)
    assert gap_mask(obs["body_lemma"], obs["body_length"]) == lmask(2)
    obs = make_obs((4, 70))
    assert gap_mask(obs["title_lemma"], obs["title_length"]) == lmask(4, 64)
    assert gap_mask(*(np.full(5, -3), np.zeros(5))) == 0


@pytest.mark.req("REQ-AGT-003")
def test_prefers_a_title_fit_over_a_more_frequent_misfit(artifacts, make_obs, lmask):
    vocab, knowledge = artifacts(
        [90, 10, 5], [lmask(7), lmask(4), lmask(6)], title_doc_freq=[50, 1, 1]
    )
    agent = make_agent("a2", vocab=vocab, knowledge=knowledge, ruleset="canonical")
    obs = make_obs((4,), (4, 5))
    agent.reset(obs, {})
    scores = agent.scores(obs, None)
    assert np.isneginf(scores[0]) and np.isneginf(scores[2])
    assert agent.act(obs, {}, None) == 1


@pytest.mark.req("REQ-AGT-003")
def test_ignores_lengths_when_the_ruleset_hides_them(artifacts, make_obs, lmask):
    masks = [lmask(7), lmask(4), lmask(6)]
    vocab, knowledge = artifacts([90, 10, 5], masks, [50, 1, 1], ruleset="no-lengths")
    agent = make_agent("a2", vocab=vocab, knowledge=knowledge, ruleset="no-lengths")
    obs = make_obs((0,), (0, 0))  # the environment exposes length 0 for hidden tokens
    assert np.isfinite(agent.scores(obs, None)).all()
    assert agent.act(obs, {}, None) == 0
    # Even if lengths leaked into the observation, the ruleset says they are hidden.
    assert agent.act(make_obs((4,), (4,)), {}, None) == 0


@pytest.mark.req("REQ-AGT-003")
def test_first_term_only_without_a_compatible_title_gap(artifacts, make_obs, lmask):
    freq = [50, 49, 48, 47]
    masks = [lmask(5), lmask(5), lmask(5), lmask(5, 9)]
    vocab, knowledge = artifacts(freq, masks, [0, 0, 0, 90])
    agent = make_agent("a2", vocab=vocab, knowledge=knowledge, ruleset="canonical")
    expected = np.asarray(freq, dtype=np.float64) / 100  # mean_count 1 x doc_freq / n_docs
    every = np.ones(4, dtype=bool)

    obs = make_obs((11,), (5,))  # no action fits the title gap
    np.testing.assert_allclose(agent.scores(obs, None), z(expected, every))
    assert agent.act(obs, {}, None) == 0

    obs = make_obs((9,), (5,))  # only action 3 fits the title gap
    title = np.array([0, 0, 0, 90.5 / (100 + 0.5 * 4)])
    np.testing.assert_allclose(agent.scores(obs, None), z(expected, every) + z(title, every))


@pytest.mark.req("REQ-AGT-003")
def test_ties_go_to_the_smallest_id(artifacts, make_obs):
    vocab, knowledge = artifacts([3, 10, 10, 10], title_doc_freq=[0, 4, 4, 4])
    agent = make_agent("a2", vocab=vocab, knowledge=knowledge, ruleset="canonical", seed=1)
    obs = make_obs((5,), (5,))
    assert agent.act(obs, {}, None) == 1
    assert agent.act(obs, {}, np.array([True, False, True, True])) == 2
    agent.reset(obs, {})
    assert agent.act(obs, {}, None) == 1


@pytest.mark.req("REQ-AGT-003")
def test_numeric_actions_compete_for_the_title_by_default(artifacts, make_obs):
    vocab, knowledge = artifacts([10, 10], title_doc_freq=[0, 99], numeric=[1])
    agent = make_agent("a2", vocab=vocab, knowledge=knowledge, ruleset="canonical")
    obs = make_obs((5,), (5,))
    assert agent.act(obs, {}, None) == 1
    variant = make_agent(
        "a2", vocab=vocab, knowledge=knowledge, ruleset="canonical", numeric_title=False
    )
    assert variant.act(obs, {}, None) == 0  # first term only


@pytest.mark.req("REQ-AGT-003")
def test_ignores_lengths_when_nothing_fits(artifacts, make_obs, lmask):
    vocab, knowledge = artifacts([5, 20, 7], [lmask(3)] * 3)
    agent = make_agent("a2", vocab=vocab, knowledge=knowledge, ruleset="canonical")
    assert agent.act(make_obs((8,), (8,)), {}, None) == 1
    assert agent.act(make_obs((8,), (8,)), {}, np.array([True, False, True])) == 2
