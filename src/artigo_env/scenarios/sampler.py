"""Article samplers (SDD 08 §B.2-B.3, DEC-062).

``EpochSampler`` draws training articles without replacement inside an epoch. Each source keeps
one permutation per epoch derived only from ``(base_seed, source id, stage, epoch)``, so every
parallel worker computes the same permutation and takes a disjoint stride of it. The source of
each draw is chosen with the caller's generator. ``FixedListSampler`` walks the eligible documents
of an evaluation split in ascending order, without randomness.
"""

from __future__ import annotations

import hashlib
import math
from bisect import bisect_right
from collections.abc import Mapping, Sequence
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt

from artigo_env.core.errors import ArtigoEnvError
from artigo_env.scenarios.columns import DocColumns
from artigo_env.scenarios.filters import apply_filters
from artigo_env.scenarios.scenario import EVAL_SPLITS, Scenario

N_TITLE_BUCKETS = 4  # title content words: 1, 2, 3, >= 4
_EPOCH_TAG = 0x53434E31  # "SCN1": separates these streams from other uses of the seed


class NoEligibleDocumentsError(ArtigoEnvError, ValueError):
    """A scenario stage leaves no document that can be drawn."""


class Sampler(Protocol):
    @property
    def stage(self) -> int: ...

    @property
    def n_eligible(self) -> int: ...

    def next(self, rng: np.random.Generator) -> int:
        """Index (into the ``DocColumns`` arrays) of the next article."""
        ...

    def set_stage(self, k: int) -> None: ...

    def notify_steps(self, n: int) -> None:
        """Report ``n`` environment steps taken since the previous call."""
        ...


def _check_int(name: str, value: Any, low: int, high: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int | np.integer):
        raise TypeError(f"{name} must be an integer, got {value!r}")
    number = int(value)
    if number < low or (high is not None and number > high):
        bound = f"[{low}, {high}]" if high is not None else f">= {low}"
        raise ValueError(f"{name} must be {bound}, got {number}")
    return number


def epoch_seed_sequence(
    base_seed: int, source: str, stage: int, epoch: int
) -> np.random.SeedSequence:
    """Seed of the permutation of ``source`` in ``epoch`` of ``stage``.

    ``SeedSequence(entropy=base_seed, spawn_key=(TAG, h0, h1, stage, epoch))`` where ``h0, h1`` are
    the first two little-endian 32-bit words of SHA-256(source id, UTF-8).
    """
    digest = hashlib.sha256(source.encode("utf-8")).digest()
    h0 = int.from_bytes(digest[0:4], "little")
    h1 = int.from_bytes(digest[4:8], "little")
    return np.random.SeedSequence(base_seed, spawn_key=(_EPOCH_TAG, h0, h1, stage, epoch))


def resolve_source_weights(
    counts: Mapping[str, int], source_mix: Mapping[str, float], source_cap: float
) -> dict[str, float]:
    """Normalized drawing weight of each source with at least one eligible document.

    Explicit ``source_mix``: the given weights (absent sources and sources without eligible
    documents get zero), renormalized. Empty ``source_mix``: proportional to ``counts`` with each
    share capped at ``source_cap`` and the excess redistributed proportionally among uncapped
    sources until no share exceeds the cap; the cap is raised to ``1 / k`` when ``k`` sources
    cannot satisfy it (so one source always gets 1.0). Raises when every weight is zero.
    """
    present = sorted(source for source, n in counts.items() if n > 0)
    raw: dict[str, float]
    if source_mix:
        raw = {source: float(source_mix.get(source, 0.0)) for source in present}
    else:
        cap = max(source_cap, 1.0 / len(present)) if present else source_cap
        raw = {}
        capped: set[str] = set()
        while True:
            free = [source for source in present if source not in capped]
            remaining = 1.0 - cap * len(capped)
            total = sum(counts[source] for source in free)
            if not free or total == 0 or remaining <= 0:
                raw.update({source: 0.0 for source in free})
                break
            shares = {source: remaining * counts[source] / total for source in free}
            over = {source for source, share in shares.items() if share > cap + 1e-12}
            if not over:
                raw.update(shares)
                break
            capped |= over
            raw.update({source: cap for source in over})
    weight_sum = math.fsum(raw.values())
    if weight_sum <= 0:
        raise NoEligibleDocumentsError(
            "no eligible document in any source with positive weight "
            f"(eligible per source: {dict(counts)}, source_mix: {dict(source_mix)})"
        )
    return {source: weight / weight_sum for source, weight in raw.items() if weight > 0}


class _StagedSampler:
    """Stage bookkeeping shared by both samplers."""

    def __init__(self, docs: DocColumns, scenario: Scenario) -> None:
        if not isinstance(docs, DocColumns):
            raise TypeError(f"docs must be DocColumns, got {type(docs).__name__}")
        if not isinstance(scenario, Scenario):
            raise TypeError(f"scenario must be a Scenario, got {type(scenario).__name__}")
        self._docs = docs
        self._scenario = scenario
        self._stages = scenario.stages
        base = (docs.split == scenario.split) & apply_filters(scenario.filters, docs)
        self._masks = [base & apply_filters(stage.filters, docs) for stage in self._stages]
        for k, mask in enumerate(self._masks):
            if not mask.any():
                raise NoEligibleDocumentsError(self._empty_message(k))
        self._stage = 0
        self._steps_in_stage = 0

    def _empty_message(self, k: int) -> str:
        scenario = self._scenario
        return (
            f"scenario {scenario.name!r}, stage {k}: no eligible document "
            f"(split={scenario.split!r}, {len(self._docs)} documents in total)"
        )

    @property
    def stage(self) -> int:
        return self._stage

    @property
    def n_stages(self) -> int:
        return len(self._stages)

    def set_stage(self, k: int) -> None:
        """Switch to curriculum stage ``k``; setting the current stage again is a no-op."""
        k = _check_int("stage", k, 0, len(self._stages) - 1)
        if k != self._stage:
            self._activate(k)

    def notify_steps(self, n: int) -> None:
        """Count ``n`` new environment steps; leave stages whose ``until_steps`` is reached."""
        self._steps_in_stage += _check_int("n", n, 0)
        while self._stage < len(self._stages) - 1:
            limit = self._stages[self._stage].until_steps
            if limit is None or self._steps_in_stage < limit:
                break
            excess = self._steps_in_stage - limit
            self._activate(self._stage + 1)
            self._steps_in_stage = excess

    def _activate(self, k: int) -> None:
        self._stage = k
        self._steps_in_stage = 0
        self._on_stage(k)

    def _on_stage(self, k: int) -> None:
        raise NotImplementedError


class EpochSampler(_StagedSampler):
    """Training sampler: per-source epochs without replacement, shared across workers.

    A worker takes positions ``worker_index, worker_index + num_workers, ...`` of each epoch
    permutation and starts the next epoch when they run out. If a source has fewer eligible
    documents than workers, a worker whose index is past the end uses position
    ``worker_index % n`` (one document per epoch; workers then overlap, unavoidably).
    """

    def __init__(
        self,
        docs: DocColumns,
        scenario: Scenario,
        *,
        base_seed: int,
        worker_index: int = 0,
        num_workers: int = 1,
        target_title_shares: Sequence[float] | None = None,
    ) -> None:
        self._base_seed = _check_int("base_seed", base_seed, 0)
        self._num_workers = _check_int("num_workers", num_workers, 1)
        self._worker_index = _check_int("worker_index", worker_index, 0, self._num_workers - 1)
        self._target: npt.NDArray[np.float64] | None = None
        if scenario.title_distribution == "match-w":
            self._target = self._check_target(target_title_shares)
        super().__init__(docs, scenario)
        names, codes = np.unique(docs.source_id, return_inverse=True)
        self._source_names: list[str] = [str(name) for name in names]
        self._codes = codes.reshape(-1)
        self._buckets = (np.clip(docs.n_title_content, 1, N_TITLE_BUCKETS) - 1).astype(np.int8)
        self._stage_weights: list[dict[str, float]] = []
        for k, mask in enumerate(self._masks):
            counts = np.bincount(self._codes[mask], minlength=len(names))
            per_source = {name: int(n) for name, n in zip(self._source_names, counts, strict=True)}
            try:
                weights = resolve_source_weights(
                    per_source, scenario.source_mix, scenario.source_cap
                )
            except NoEligibleDocumentsError as err:
                raise NoEligibleDocumentsError(f"{self._empty_message(k)}: {err}") from None
            self._stage_weights.append(weights)
        self._activate(0)

    @staticmethod
    def _check_target(shares: Sequence[float] | None) -> npt.NDArray[np.float64]:
        if shares is None:
            raise ValueError("title_distribution='match-w' requires target_title_shares")
        arr = np.asarray(shares, dtype=np.float64)
        if arr.shape != (N_TITLE_BUCKETS,) or not np.isfinite(arr).all() or (arr < 0).any():
            raise ValueError(
                f"target_title_shares must be {N_TITLE_BUCKETS} finite non-negative shares "
                f"(title content words 1, 2, 3, >=4), got {shares!r}"
            )
        if arr.sum() <= 0:
            raise ValueError("target_title_shares must not all be zero")
        normalized: npt.NDArray[np.float64] = arr / arr.sum()
        return normalized

    def _on_stage(self, k: int) -> None:
        weights = self._stage_weights[k]
        mask = self._masks[k]
        eligible = np.flatnonzero(mask)
        eligible_codes = self._codes[eligible]
        self._active: list[str] = []
        self._eligible: list[npt.NDArray[np.int64]] = []
        self._doc_weights: list[npt.NDArray[np.float64] | None] = []
        cumulative: list[float] = []
        running = 0.0
        for code, name in enumerate(self._source_names):
            weight = weights.get(name, 0.0)
            if weight <= 0:
                continue
            indices = eligible[eligible_codes == code]
            self._active.append(name)
            self._eligible.append(indices)
            self._doc_weights.append(self._match_weights(indices))
            running += weight
            cumulative.append(running)
        cumulative[-1] = 1.0
        self._cumulative = cumulative
        self._orders: list[list[int]] = [[] for _ in self._active]
        self._positions = [0] * len(self._active)
        self._epochs = [-1] * len(self._active)

    def _match_weights(self, indices: npt.NDArray[np.int64]) -> npt.NDArray[np.float64] | None:
        if self._target is None:
            return None
        buckets = self._buckets[indices]
        source_share = np.bincount(buckets, minlength=N_TITLE_BUCKETS) / len(indices)
        per_bucket = np.zeros(N_TITLE_BUCKETS)
        present = source_share > 0
        per_bucket[present] = self._target[present] / source_share[present]
        doc_weights: npt.NDArray[np.float64] = per_bucket[buckets]
        return doc_weights

    @property
    def n_eligible(self) -> int:
        """Documents that can be drawn in the current stage."""
        return sum(len(indices) for indices in self._eligible)

    @property
    def weights(self) -> dict[str, float]:
        """Normalized weight of each source that can be drawn in the current stage."""
        return dict(self._stage_weights[self._stage])

    def epoch_order(self, source: str, epoch: int) -> npt.NDArray[np.int64]:
        """Full permutation (all workers) of ``source`` in ``epoch`` of the current stage."""
        try:
            slot = self._active.index(source)
        except ValueError:
            raise KeyError(f"source {source!r} cannot be drawn in stage {self._stage}") from None
        return self._order(slot, _check_int("epoch", epoch, 0))

    def _order(self, slot: int, epoch: int) -> npt.NDArray[np.int64]:
        indices = self._eligible[slot]
        seed = epoch_seed_sequence(self._base_seed, self._active[slot], self._stage, epoch)
        gen = np.random.Generator(np.random.PCG64(seed))
        doc_weights = self._doc_weights[slot]
        if doc_weights is None:
            permuted: npt.NDArray[np.int64] = indices[gen.permutation(len(indices))]
            return permuted
        # Efraimidis-Spirakis: key u ** (1 / w), descending; log(u) / w gives the same order
        # without underflow for small weights. Zero-weight documents go last, shuffled.
        u = 1.0 - gen.random(len(indices))
        positive = doc_weights > 0
        keys = np.log(u[positive]) / doc_weights[positive]
        head = indices[positive][np.argsort(-keys, kind="stable")]
        tail = indices[~positive]
        tail = tail[gen.permutation(len(tail))]
        return np.concatenate([head, tail])

    def _start_epoch(self, slot: int) -> list[int]:
        epoch = self._epochs[slot] + 1
        self._epochs[slot] = epoch
        order = self._order(slot, epoch)
        start = self._worker_index
        if start >= len(order):
            start %= len(order)
        mine: list[int] = order[start :: self._num_workers].tolist()
        self._orders[slot] = mine
        self._positions[slot] = 0
        return mine

    def next(self, rng: np.random.Generator) -> int:
        """Choose a source by weight with ``rng``, then return its next document for this worker."""
        slot = bisect_right(self._cumulative, rng.random())
        if slot >= len(self._cumulative):
            slot = len(self._cumulative) - 1
        position = self._positions[slot]
        order = self._orders[slot]
        if position >= len(order):
            order = self._start_epoch(slot)
            position = 0
        self._positions[slot] = position + 1
        return order[position]


class FixedListSampler(_StagedSampler):
    """Evaluation sampler: eligible documents of the split in ascending index order.

    ``next`` ignores the generator, walks the list and wraps around.
    """

    def __init__(self, docs: DocColumns, scenario: Scenario) -> None:
        super().__init__(docs, scenario)
        self._activate(0)

    def _on_stage(self, k: int) -> None:
        self._list: list[int] = np.flatnonzero(self._masks[k]).tolist()
        self._position = 0

    @property
    def n_eligible(self) -> int:
        return len(self._list)

    @property
    def indices(self) -> tuple[int, ...]:
        return tuple(self._list)

    def next(self, rng: np.random.Generator | None = None) -> int:
        index = self._list[self._position]
        self._position = (self._position + 1) % len(self._list)
        return index

    def index_of(self, article_index: int) -> int:
        """Document index of entry ``article_index`` of the fixed list (no state change)."""
        if isinstance(article_index, bool) or not isinstance(article_index, int | np.integer):
            raise TypeError(f"article_index must be an integer, got {article_index!r}")
        if not 0 <= article_index < len(self._list):
            raise IndexError(
                f"article_index {article_index} out of range for scenario "
                f"{self._scenario.name!r} (valid: 0..{len(self._list) - 1})"
            )
        return self._list[int(article_index)]


def make_sampler(docs: DocColumns, scenario: Scenario, **kwargs: Any) -> Sampler:
    """Fixed list for evaluation splits (``kwargs`` ignored), epoch sampler otherwise."""
    if scenario.split in EVAL_SPLITS:
        return FixedListSampler(docs, scenario)
    return EpochSampler(docs, scenario, **kwargs)
