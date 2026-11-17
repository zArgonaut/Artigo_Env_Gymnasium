"""Evaluation metrics (SDD 14 §4) and bootstrap confidence intervals (SDD 14 §5).

Per episode (``EpisodeMetrics``): ``N`` counts the guesses that count, with a truncated episode
counted as ``T_max`` (REQ-EVL-002); efficiency is the number of distinct lemmas required by the
title divided by ``N`` (the theoretical minimum over the guesses used, 0 when not won); hit rate
is the fraction of steps with ``count > 0``; the first title hit is the 1-based step at which a
required title token was first fully revealed; the reveal AUC is the mean, over steps, of the
fraction of initially hidden body tokens revealed after the step.

Summary (``summarize``): the main metric ``N̄`` and every other headline metric are computed over
the episodes whose title can be won through the discrete interface (``title_oov`` false);
``title_oov`` episodes are summarized apart. Confidence intervals resample articles (all seeds
of an article move together), 10,000 times with a fixed seed, vectorized in chunks.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20_261_112
CONFIDENCE = 0.95
_CHUNK = 500  # resamples per vectorized batch (bounds memory on large evaluation sets)

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class EpisodeMetrics:
    """Category-3 measures of one finished episode (one row of ``episodes.parquet``)."""

    episode_id: str
    seed: int
    article_index: int
    article_ref: str
    source: str
    domain: str | None
    n_title_content: int
    n_body_tokens: int
    n_steps: int
    N: int  # noqa: N815 - the SDD's name for the main measure
    won: bool
    truncated: bool
    title_oov: bool
    efficiency: float
    hit_rate: float
    reveal_auc: float
    first_title_hit: int | None
    reward_total: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# -- per-episode measures -------------------------------------------------------------------------


def counted_guesses(n_counted: int, truncated: bool, t_max: int) -> int:
    """``N`` of one episode: truncated episodes count as ``T_max`` (REQ-EVL-002)."""
    return int(t_max) if truncated else min(int(n_counted), int(t_max))


def efficiency(n_title_lemmas: int, n: int, won: bool) -> float:
    """Distinct required title lemmas ÷ counted guesses; 0 when the episode was not won."""
    if not won or n <= 0:
        return 0.0
    return min(1.0, float(n_title_lemmas) / float(n))


def hit_rate(counts: Sequence[int]) -> float:
    """Fraction of steps whose guess revealed at least one token."""
    if not counts:
        return 0.0
    return float(np.mean(np.asarray(counts) > 0))


def first_title_hit(title_fractions: Sequence[float]) -> int | None:
    """1-based step of the first increase of the revealed title fraction, ``None`` if none."""
    previous = 0.0
    for step, value in enumerate(title_fractions, start=1):
        if value > previous:
            return step
        previous = value
    return None


def reveal_auc(revealed_fractions: Sequence[float]) -> float:
    """Mean fraction of initially hidden body tokens revealed, over the steps of the episode."""
    if not revealed_fractions:
        return 0.0
    return float(np.mean(revealed_fractions))


# -- bootstrap ------------------------------------------------------------------------------------


def _article_matrix(
    episodes: Sequence[EpisodeMetrics], value: Callable[[EpisodeMetrics], float]
) -> FloatArray:
    """``(n_articles, n_seeds)`` matrix of ``value``; missing cells are NaN."""
    articles = sorted({e.article_index for e in episodes})
    seeds = sorted({e.seed for e in episodes})
    row = {a: i for i, a in enumerate(articles)}
    col = {s: j for j, s in enumerate(seeds)}
    matrix = np.full((len(articles), len(seeds)), np.nan, dtype=np.float64)
    for e in episodes:
        matrix[row[e.article_index], col[e.seed]] = value(e)
    return matrix


def bootstrap_ci(
    matrix: FloatArray,
    statistic: Callable[[FloatArray], FloatArray],
    *,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
    confidence: float = CONFIDENCE,
) -> tuple[float, float]:
    """Percentile interval of ``statistic`` over articles resampled with replacement.

    ``matrix`` has one row per article (cells NaN where excluded); ``statistic`` maps a batch
    ``(B, n_values)`` to ``(B,)`` and must ignore NaN. Returns ``(nan, nan)`` without data.
    """
    n_articles = matrix.shape[0]
    if n_articles == 0 or np.all(np.isnan(matrix)):
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    out = np.empty(resamples, dtype=np.float64)
    for start in range(0, resamples, _CHUNK):
        size = min(_CHUNK, resamples - start)
        idx = rng.integers(0, n_articles, size=(size, n_articles))
        batch = matrix[idx].reshape(size, -1)
        out[start : start + size] = statistic(batch)
    out = out[~np.isnan(out)]
    if out.size == 0:
        return (float("nan"), float("nan"))
    tail = (1.0 - confidence) / 2.0 * 100.0
    low, high = np.percentile(out, [tail, 100.0 - tail])
    return (float(low), float(high))


def _quiet(func: Callable[[FloatArray], Any]) -> Callable[[FloatArray], FloatArray]:
    """Wrap a NaN-aware statistic so all-NaN rows give NaN without warnings."""

    def wrapped(batch: FloatArray) -> FloatArray:
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            return np.asarray(func(batch), dtype=np.float64)

    return wrapped


MEAN = _quiet(lambda b: np.nanmean(b, axis=1))
MEDIAN = _quiet(lambda b: np.nanmedian(b, axis=1))
P90 = _quiet(lambda b: np.nanpercentile(b, 90, axis=1))


def _metric(
    matrix: FloatArray, statistic: Callable[[FloatArray], FloatArray], *, resamples: int
) -> dict[str, float | int | None]:
    flat = matrix.reshape(1, -1)
    value = float(statistic(flat)[0]) if matrix.size else float("nan")
    low, high = bootstrap_ci(matrix, statistic, resamples=resamples)
    n = int(np.count_nonzero(~np.isnan(matrix)))
    return {"value": _finite(value), "ci_low": _finite(low), "ci_high": _finite(high), "n": n}


def _finite(x: float) -> float | None:
    return None if not np.isfinite(x) else float(x)


def summarize(
    episodes: Sequence[EpisodeMetrics], *, resamples: int = BOOTSTRAP_RESAMPLES
) -> dict[str, Any]:
    """Headline metrics with 95% bootstrap intervals over articles (SDD 14 §4–§5)."""
    playable = [e for e in episodes if not e.title_oov]
    oov = [e for e in episodes if e.title_oov]

    def matrix(value: Callable[[EpisodeMetrics], float], subset: Sequence[EpisodeMetrics]) -> Any:
        return _article_matrix(subset, value) if subset else np.empty((0, 0))

    n_all = matrix(lambda e: float(e.N), playable)
    n_won = matrix(lambda e: float(e.N) if e.won else np.nan, playable)
    metrics = {
        "n_mean": _metric(n_all, MEAN, resamples=resamples),
        "win_rate": _metric(matrix(lambda e: float(e.won), playable), MEAN, resamples=resamples),
        "n_median_won": _metric(n_won, MEDIAN, resamples=resamples),
        "n_p90_won": _metric(n_won, P90, resamples=resamples),
        "n_median_all": _metric(n_all, MEDIAN, resamples=resamples),
        "n_p90_all": _metric(n_all, P90, resamples=resamples),
        "efficiency_mean": _metric(
            matrix(lambda e: e.efficiency, playable), MEAN, resamples=resamples
        ),
        "hit_rate_mean": _metric(matrix(lambda e: e.hit_rate, playable), MEAN, resamples=resamples),
        "reveal_auc_mean": _metric(
            matrix(lambda e: e.reveal_auc, playable), MEAN, resamples=resamples
        ),
        "title_coverage": _metric(
            matrix(lambda e: float(not e.title_oov), list(episodes)), MEAN, resamples=resamples
        ),
    }
    return {
        "n_episodes": len(episodes),
        "n_articles": len({e.article_index for e in episodes}),
        "n_playable_episodes": len(playable),
        "metrics": metrics,
        "title_oov": {
            "n_episodes": len(oov),
            "n_articles": len({e.article_index for e in oov}),
            "n_mean": _finite(float(np.mean([e.N for e in oov]))) if oov else None,
            "win_rate": _finite(float(np.mean([e.won for e in oov]))) if oov else None,
        },
        "bootstrap": {
            "resamples": resamples,
            "seed": BOOTSTRAP_SEED,
            "confidence": CONFIDENCE,
            "unit": "article",
            "method": "percentile",
        },
    }
