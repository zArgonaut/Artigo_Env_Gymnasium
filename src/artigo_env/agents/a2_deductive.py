"""A2: deductive expected-value player, minimal F4 version (SDD 12 §2; F4 plan decision 5).

For every allowed action ``w``::

    E(w) = mean_count_present[w] * (doc_freq[w] / n_docs) * compat_body(w)
    T(w) = (title_doc_freq[w] + alpha) / (n_docs + alpha * |V|) * compat_title(w)
    score(w) = z(E)(w) + lam * z(T)(w)

``compat_x(w)`` is 1 when the ruleset hides the lengths of section ``x``, else 1 when
``length_mask[w]`` shares a bit with the lengths of the gaps still hidden in ``x`` (positions
whose lemma code is ``-2`` and whose length is ``> 0``), else 0. Numeric actions take part in
the first term only (``T = 0``; hyperparameter ``numeric_title=True`` lets them use ``T`` like
lemmas, which avoids getting stuck on a hidden number in a title once it is out of reach of the
first term). ``z`` standardizes over the allowed actions compatible in at least one term
(``(x - mean) / std``, 0 when ``std`` is 0); when no allowed action is compatible with a title
gap, the score is ``z(E)`` alone; when no allowed action is compatible at all,
lengths are ignored for that decision. Ties go to the smallest id. ``context_c1`` arrives with the
knowledge store of phase F5.

Everything is vectorized over V (a few milliseconds per decision for ``|V| = 300,000``).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np
import numpy.typing as npt

from artigo_env.agents.base import HIDDEN, Obs, ReferenceAgent, allowed_actions
from artigo_env.agents.card import AgentCard
from artigo_env.core.vocab import Vocabulary
from artigo_env.engine.rulesets.components import LENGTH_RULES
from artigo_env.engine.rulesets.ruleset import Ruleset
from artigo_env.services.knowledge import KnowledgeStore


def gap_mask(lemma: npt.NDArray[Any], length: npt.NDArray[Any]) -> int:
    """Length bits (``min(len, 64) - 1``) of the hidden positions with a known length."""
    lemma = np.asarray(lemma)
    length = np.asarray(length, dtype=np.int64)
    sizes = length[(lemma == HIDDEN) & (length > 0)]
    if not len(sizes):
        return 0
    mask = 0
    for bit in np.unique(np.minimum(sizes, 64) - 1).tolist():  # at most 64 distinct lengths
        mask |= 1 << int(bit)
    return mask


_OFF = 1e300  # subtracted from the raw score of actions outside the subset (argmax only)


def _scale(
    values: npt.NDArray[np.float64],
    squares: npt.NDArray[np.float64],
    weights: npt.NDArray[np.float64],
    n: int,
) -> tuple[float, float]:
    """``(mean, 1 / std)`` over a subset of ``n`` actions; ``values * weights`` is the term.

    ``weights`` is 1 where the term is defined and 0 elsewhere (the term is 0 there). Moments are
    BLAS dot products with the 0/1 weights. The factor is 0 when every value of the subset is
    equal up to rounding.
    """
    mean = float(np.dot(values, weights)) / n
    second = float(np.dot(squares, weights)) / n
    variance = second - mean * mean
    if not variance > 1e-12 * second:
        return mean, 0.0
    return mean, 1.0 / math.sqrt(variance)


class A2Deductive(ReferenceAgent):
    """Deterministic: the random generator of the base class is never used."""

    def __init__(
        self, card: AgentCard, knowledge: KnowledgeStore, vocab: Vocabulary, ruleset: Ruleset
    ) -> None:
        super().__init__(card)
        knowledge.check(vocab=vocab)
        self.lam = float(card.hyperparams.get("lam", 1.0))
        self.alpha = float(card.hyperparams.get("alpha", 0.5))
        if not self.alpha > 0:
            raise ValueError(f"alpha must be positive, got {self.alpha!r}")
        stats = knowledge.lemma_stats
        n_docs = max(int(knowledge.n_docs), 1)
        size = vocab.size
        self.size = size
        numeric = np.asarray(vocab.is_numeric, dtype=np.bool_)
        self.length_mask = np.ascontiguousarray(vocab.length_mask, dtype=np.uint64)
        self.expected = (
            stats["mean_count_present"].astype(np.float64)
            * stats["doc_freq"].astype(np.float64)
            / n_docs
        )
        title = knowledge.title_stats["title_doc_freq"].astype(np.float64)
        self.title = (title + self.alpha) / (n_docs + self.alpha * size)
        self.numeric_title = bool(card.hyperparams.get("numeric_title", True))
        if self.numeric_title:  # variant: numbers compete for title gaps like lemmas
            numeric = np.zeros(size, dtype=np.bool_)
        self.title[numeric] = 0.0
        self.title_eligible = ~numeric
        self.expected_sq = self.expected * self.expected
        self.title_sq = self.title * self.title
        self._bits = np.empty(size, dtype=np.uint64)
        self._body = np.empty(size, dtype=np.bool_)
        self._title = np.empty(size, dtype=np.bool_)
        self._title_ok = np.empty(size, dtype=np.bool_)
        self._subset = np.empty(size, dtype=np.bool_)
        self._f_body = np.empty(size, dtype=np.float64)
        self._f_title = np.empty(size, dtype=np.float64)
        self._f_subset = np.empty(size, dtype=np.float64)
        self._raw = np.empty(size, dtype=np.float64)
        self._term = np.empty(size, dtype=np.float64)
        lengths = LENGTH_RULES.get(ruleset.lengths_visible)
        self.show_title_lengths = bool(lengths.visible(True))
        self.show_body_lengths = bool(lengths.visible(False))

    def _compat(self, shown: bool, lemma: Any, length: Any, out: Any) -> Any:
        """Compatibility with the hidden gaps of one section; ``None`` means all compatible."""
        if not shown:
            return None
        gaps = np.uint64(gap_mask(lemma, length))
        np.bitwise_and(self.length_mask, gaps, out=self._bits)
        return np.not_equal(self._bits, 0, out=out)

    def _linear(
        self, obs: Obs, action_mask: npt.NDArray[np.bool_] | None
    ) -> tuple[npt.NDArray[np.float64], float]:
        """``(raw, offset)``: ``raw - offset`` is the score on the subset; off it ``raw`` is
        about ``-1e300``.

        ``z(E) + lam * z(T) = E / sd(E) + lam * T / sd(T) - offset``, so the argmax needs only
        ``raw``. All work happens in preallocated buffers (no array allocations of size |V|).
        """
        allowed = allowed_actions(action_mask, self.size)
        body = self._compat(
            self.show_body_lengths, obs["body_lemma"], obs["body_length"], self._body
        )
        title = self._compat(
            self.show_title_lengths, obs["title_lemma"], obs["title_length"], self._title
        )
        body_ok = np.logical_and(allowed, body, out=self._body) if body is not None else allowed
        title_ok = np.logical_and(allowed, self.title_eligible, out=self._title_ok)
        if title is not None:
            np.logical_and(title_ok, title, out=title_ok)
        subset = np.logical_or(body_ok, title_ok, out=self._subset)
        n = int(np.count_nonzero(subset))
        if n == 0:  # nothing fits any gap: ignore lengths for this decision
            body_ok = allowed
            np.logical_and(allowed, self.title_eligible, out=title_ok)
            subset = self._subset
            np.copyto(subset, allowed)
            n = int(np.count_nonzero(subset))
        f_body, f_title, f_subset = self._f_body, self._f_title, self._f_subset
        np.copyto(f_body, body_ok, casting="unsafe")
        np.copyto(f_subset, subset, casting="unsafe")
        raw, term = self._raw, self._term
        mean_e, inv_e = _scale(self.expected, self.expected_sq, f_body, n)
        np.multiply(self.expected, f_body, out=raw)
        raw *= inv_e
        offset = mean_e * inv_e
        if title_ok.any():
            np.copyto(f_title, title_ok, casting="unsafe")
            mean_t, inv_t = _scale(self.title, self.title_sq, f_title, n)
            factor = self.lam * inv_t
            np.multiply(self.title, f_title, out=term)
            term *= factor
            raw += term
            offset += mean_t * factor
        np.subtract(f_subset, 1.0, out=term)  # 0 on the subset, -1 off it
        term *= _OFF
        raw += term
        return raw, offset

    def scores(
        self, obs: Obs, action_mask: npt.NDArray[np.bool_] | None
    ) -> npt.NDArray[np.float64]:
        """``z(E) + lam * z(T)`` of every action (``-inf`` outside the standardization set)."""
        raw, offset = self._linear(obs, action_mask)
        return np.where(self._subset, raw - offset, -np.inf)

    def act(
        self, obs: Obs, info: Mapping[str, Any], action_mask: npt.NDArray[np.bool_] | None
    ) -> int:
        raw, _ = self._linear(obs, action_mask)
        return int(np.argmax(raw))  # argmax returns the first maximum: ties -> smallest id
