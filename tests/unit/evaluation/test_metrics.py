"""Per-episode measures, summary and bootstrap intervals (SDD 14 §4–§5)."""

from __future__ import annotations

import numpy as np
import pytest

from artigo_env.evaluation.metrics import (
    BOOTSTRAP_SEED,
    MEAN,
    EpisodeMetrics,
    bootstrap_ci,
    counted_guesses,
    efficiency,
    first_title_hit,
    hit_rate,
    reveal_auc,
    summarize,
)

pytestmark = pytest.mark.fast


def episode(index, n, *, seed=0, won=True, truncated=False, oov=False, eff=0.5, hits=0.5):
    return EpisodeMetrics(
        episode_id=f"s{seed}-a{index:05d}",
        seed=seed,
        article_index=index,
        article_ref=f"ref{index}",
        source="s",
        domain=None,
        n_title_content=1,
        n_body_tokens=10,
        n_steps=n,
        N=n,
        won=won,
        truncated=truncated,
        title_oov=oov,
        efficiency=eff,
        hit_rate=hits,
        reveal_auc=0.3,
        first_title_hit=1,
        reward_total=-float(n),
    )


@pytest.mark.req("REQ-EVL-002")
def test_truncated_episodes_count_t_max():
    assert counted_guesses(37, truncated=True, t_max=400) == 400
    assert counted_guesses(12, truncated=False, t_max=400) == 12
    eps = [episode(0, 10), episode(1, 400, won=False, truncated=True)]
    summary = summarize(eps, resamples=200)
    assert summary["metrics"]["n_mean"]["value"] == pytest.approx(205.0)
    assert summary["metrics"]["win_rate"]["value"] == pytest.approx(0.5)
    assert summary["metrics"]["n_median_won"]["value"] == pytest.approx(10.0)
    assert summary["metrics"]["n_median_all"]["value"] == pytest.approx(205.0)


@pytest.mark.req("REQ-EVL-002")
def test_episode_measures():
    assert efficiency(3, 6, won=True) == pytest.approx(0.5)
    assert efficiency(3, 6, won=False) == 0.0
    assert hit_rate([0, 2, 0, 1]) == pytest.approx(0.5)
    assert hit_rate([]) == 0.0
    assert first_title_hit([0.0, 0.0, 0.5, 1.0]) == 3
    assert first_title_hit([0.0, 0.0]) is None
    assert reveal_auc([0.0, 0.5, 1.0]) == pytest.approx(0.5)
    assert reveal_auc([]) == 0.0


@pytest.mark.req("REQ-EVL-002")
def test_title_oov_episodes_are_reported_apart():
    eps = [episode(0, 10), episode(1, 20), episode(2, 400, won=False, truncated=True, oov=True)]
    summary = summarize(eps, resamples=200)
    assert summary["n_playable_episodes"] == 2
    assert summary["metrics"]["n_mean"]["value"] == pytest.approx(15.0)
    assert summary["metrics"]["title_coverage"]["value"] == pytest.approx(2 / 3)
    assert summary["title_oov"] == {
        "n_episodes": 1,
        "n_articles": 1,
        "n_mean": 400.0,
        "win_rate": 0.0,
    }


@pytest.mark.req("REQ-EVL-002")
def test_bootstrap_is_reproducible_and_brackets_the_mean():
    rng = np.random.default_rng(0)
    eps = [episode(i, int(n)) for i, n in enumerate(rng.integers(5, 100, size=40))]
    first = summarize(eps, resamples=2000)["metrics"]["n_mean"]
    second = summarize(eps, resamples=2000)["metrics"]["n_mean"]
    assert first == second
    assert first["ci_low"] < first["value"] < first["ci_high"]
    assert summarize(eps)["bootstrap"]["resamples"] == 10_000


@pytest.mark.req("REQ-EVL-002")
def test_bootstrap_matches_a_naive_loop():
    values = np.arange(12, dtype=np.float64).reshape(6, 2)  # 6 articles x 2 seeds
    low, high = bootstrap_ci(values, MEAN, resamples=300)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    stats = []
    for start in range(0, 300, 500):
        idx = rng.integers(0, 6, size=(min(500, 300 - start), 6))
        stats.extend(values[row].mean() for row in idx)
    assert (low, high) == tuple(np.percentile(stats, [2.5, 97.5]))


@pytest.mark.req("REQ-EVL-002")
def test_seeds_of_an_article_are_resampled_together():
    eps = [episode(i, 10 * (i + 1), seed=s) for i in range(3) for s in (0, 1)]
    metric = summarize(eps, resamples=500)["metrics"]["n_mean"]
    assert metric["n"] == 6
    assert metric["value"] == pytest.approx(20.0)
    assert 10.0 <= metric["ci_low"] <= metric["ci_high"] <= 30.0


@pytest.mark.req("REQ-EVL-002")
def test_empty_subsets_give_null_values():
    summary = summarize([episode(0, 400, won=False, truncated=True)], resamples=100)
    assert summary["metrics"]["n_median_won"]["value"] is None
    assert summary["metrics"]["n_median_won"]["n"] == 0
