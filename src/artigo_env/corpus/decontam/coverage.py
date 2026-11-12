"""13-gram barrier with the coverage rule, and the in-memory decontamination (SDD 06 §7.2).

A training document is contaminated when the 13-grams it shares with W, excluding the frequent
ones (present in more than ``FREQ_LIMIT`` training documents), cover at least ``MIN_NEW``
token positions that lie inside no frequent 13-gram (REQ-COR-008). The naive rule ("any rare
13-gram in common") removes clean documents: a 13-gram that extends a frequent phrase by one
coincident word is rare and escapes the frequency filter.

At scale the frequencies are counted in three passes, so that only the 13-grams of documents
that touch W are ever counted:

* A: :func:`hashes_of_interest` over every training document;
* B: :class:`DocFrequency` built from the union of A, fed every training document;
* C: :func:`is_contaminated` with the frequent hashes from B.

:func:`decontaminate` runs the three barriers in memory (title, then 13-grams, then near
duplicates) and is the reference the three-pass protocol must agree with.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from artigo_env.corpus.decontam.grams import fold_accents as fold_token
from artigo_env.corpus.decontam.grams import gram_hashes, isin_sorted
from artigo_env.corpus.decontam.minhash import NearDuplicateIndex, shingle_hashes
from artigo_env.corpus.decontam.titles import TitleBarrier

N_GRAM = 13
FREQ_LIMIT = 10
MIN_NEW = 7

REASON_TITLE = "w_title"
REASON_13GRAM = "w_13gram"
REASON_NEAR = "w_near_duplicate"


def _sorted_unique(values: NDArray[np.uint64]) -> NDArray[np.uint64]:
    return np.unique(np.asarray(values, dtype=np.uint64).reshape(-1))


class WGramIndex:
    """Sorted, duplicate-free ``uint64`` array of the 13-gram hashes of W."""

    __slots__ = ("hashes",)

    def __init__(self, hashes: NDArray[np.uint64] | None = None) -> None:
        empty = np.empty(0, dtype=np.uint64)
        self.hashes: NDArray[np.uint64] = _sorted_unique(empty if hashes is None else hashes)

    @classmethod
    def build(cls, docs_tokens: Iterable[Sequence[str]], n: int = N_GRAM) -> WGramIndex:
        parts = [gram_hashes(tokens, n) for tokens in docs_tokens]
        return cls(np.concatenate(parts) if parts else None)

    def contains(self, hashes: NDArray[np.uint64]) -> NDArray[np.bool_]:
        return isin_sorted(self.hashes, np.asarray(hashes, dtype=np.uint64))

    def __len__(self) -> int:
        return int(self.hashes.size)

    def save(self, path: Path) -> None:
        with path.open("wb") as handle:
            np.save(handle, self.hashes, allow_pickle=False)

    @classmethod
    def load(cls, path: Path) -> WGramIndex:
        array = np.load(path, allow_pickle=False)
        if array.dtype != np.uint64 or array.ndim != 1:
            raise ValueError(f"{path.name}: expected a 1-D uint64 array")
        if array.size > 1 and not bool(np.all(array[1:] > array[:-1])):
            raise ValueError(f"{path.name}: 13-gram index is not sorted and unique")
        index = cls.__new__(cls)
        index.hashes = array
        return index


def covered_new_tokens(
    n_tokens: int,
    common: NDArray[np.bool_],
    frequent: NDArray[np.bool_],
    *,
    n: int = N_GRAM,
) -> int:
    """Token positions covered by a common, non-frequent gram and by no frequent gram.

    ``common[i]`` and ``frequent[i]`` describe the gram that starts at token ``i``.
    """
    count = max(n_tokens - n + 1, 0)
    if common.shape != (count,) or frequent.shape != (count,):
        raise ValueError(f"expected {count} gram flags for {n_tokens} tokens")
    if count == 0:
        return 0

    def spans(starts: NDArray[np.bool_]) -> NDArray[np.bool_]:
        delta = np.zeros(n_tokens + 1, dtype=np.int64)
        index = np.flatnonzero(starts)
        np.add.at(delta, index, 1)
        np.add.at(delta, index + n, -1)
        result: NDArray[np.bool_] = np.cumsum(delta[:n_tokens]) > 0
        return result

    in_common = spans(common & ~frequent)
    in_frequent = spans(frequent)
    return int(np.count_nonzero(in_common & ~in_frequent))


# --- three-pass protocol ---------------------------------------------------------------------


def hashes_of_interest(tokens: Sequence[str], w_index: WGramIndex) -> NDArray[np.uint64] | None:
    """Pass A: ``None`` if no 13-gram is shared with W, else all 13-gram hashes (sorted, unique)."""
    grams = gram_hashes(tokens, N_GRAM)
    if not bool(w_index.contains(grams).any()):
        return None
    return _sorted_unique(grams)


class DocFrequency:
    """Pass B: document frequency of the hashes of interest across the training documents."""

    __slots__ = ("counts", "interest")

    def __init__(self, interest: NDArray[np.uint64]) -> None:
        self.interest: NDArray[np.uint64] = _sorted_unique(interest)
        self.counts: NDArray[np.int64] = np.zeros(self.interest.size, dtype=np.int64)

    def add(self, tokens: Sequence[str]) -> None:
        """Count 1 for each hash of interest present in the document (once per document)."""
        if self.interest.size == 0:
            return
        grams = _sorted_unique(gram_hashes(tokens, N_GRAM))
        present = isin_sorted(self.interest, grams)
        self.counts[np.searchsorted(self.interest, grams[present])] += 1

    def merge(self, other: DocFrequency) -> None:
        """Add the counts of ``other`` (same interest set), for counts made in parallel."""
        if not np.array_equal(self.interest, other.interest):
            raise ValueError("cannot merge document frequencies over different interest sets")
        self.counts += other.counts

    def frequent(self, limit: int = FREQ_LIMIT) -> NDArray[np.uint64]:
        """Hashes present in more than ``limit`` documents (sorted)."""
        result: NDArray[np.uint64] = self.interest[self.counts > limit]
        return result


def is_contaminated(
    tokens: Sequence[str], w_index: WGramIndex, frequent: NDArray[np.uint64]
) -> tuple[bool, int]:
    """Pass C: ``(covered >= MIN_NEW, covered)``; ``frequent`` is sorted (from pass B)."""
    grams = gram_hashes(tokens, N_GRAM)
    common = w_index.contains(grams)
    if not bool(common.any()):
        return False, 0
    frequent_mask = isin_sorted(np.asarray(frequent, dtype=np.uint64), grams)
    covered = covered_new_tokens(len(tokens), common, frequent_mask)
    return covered >= MIN_NEW, covered


# --- documents and verdicts ------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TrainDoc:
    """A training document as the barriers see it (tokens from ``grams.gram_tokens``)."""

    doc_id: str
    title: str
    tokens: Sequence[str]


@dataclass(frozen=True, slots=True)
class WDoc:
    """An evaluation (W) article: title and body tokens."""

    title: str
    tokens: Sequence[str]


@dataclass(frozen=True, slots=True)
class Verdict:
    """Decision for one training document; ``reason`` is ``None`` when it is kept."""

    doc_id: str
    reason: str | None = None
    covered: int = 0  # 13-gram coverage measured (0 when the title barrier decided first)
    folded: bool = False  # removed by the accent-folded 13-gram pass
    match: int | None = None  # key of the matching W document (near-duplicate barrier)

    @property
    def contaminated(self) -> bool:
        return self.reason is not None


def judge(
    doc: TrainDoc,
    *,
    titles: TitleBarrier,
    grams: WGramIndex,
    near: NearDuplicateIndex,
    frequent: NDArray[np.uint64],
) -> Verdict:
    """The three barriers in order (title, 13-grams, near duplicate) for one document."""
    if doc.title in titles:
        return Verdict(doc.doc_id, REASON_TITLE)
    hit, covered = is_contaminated(doc.tokens, grams, frequent)
    if hit:
        return Verdict(doc.doc_id, REASON_13GRAM, covered)
    match = near.match(shingle_hashes(doc.tokens))
    if match is not None:
        return Verdict(doc.doc_id, REASON_NEAR, covered, match=match)
    return Verdict(doc.doc_id, None, covered)


def _frequent_direct(gram_lists: Sequence[NDArray[np.uint64]]) -> NDArray[np.uint64]:
    """Hashes in more than ``FREQ_LIMIT`` documents, counting every gram of every document."""
    uniques = [np.unique(grams) for grams in gram_lists]
    if not uniques:
        return np.empty(0, dtype=np.uint64)
    values, counts = np.unique(np.concatenate(uniques), return_counts=True)
    result: NDArray[np.uint64] = values[counts > FREQ_LIMIT]
    return result


def decontaminate(
    train: Sequence[TrainDoc], w: Sequence[WDoc], *, fold_accents: bool = False
) -> list[Verdict]:
    """Decide every training document against W in memory, one verdict per document in order.

    Barriers: title (``w_title``), 13-gram coverage (``w_13gram``; with ``fold_accents`` a
    second pass over accent-folded tokens follows the plain one), then near duplicate of a W
    article (``w_near_duplicate``, exact Jaccard of 5-gram shingles ``>= 0.8``). Frequencies
    count all training documents, including those a barrier removes.
    """
    titles = TitleBarrier(doc.title for doc in w)
    reasons: list[str | None] = [REASON_TITLE if doc.title in titles else None for doc in train]
    covered = [0] * len(train)
    folded = [False] * len(train)

    for fold in (False, True) if fold_accents else (False,):
        w_tokens = [_fold(doc.tokens) if fold else doc.tokens for doc in w]
        tr_tokens = [_fold(doc.tokens) if fold else doc.tokens for doc in train]
        index = WGramIndex.build(w_tokens)
        gram_lists = [gram_hashes(tokens, N_GRAM) for tokens in tr_tokens]
        frequent = _frequent_direct(gram_lists)
        for i, grams in enumerate(gram_lists):
            if reasons[i] is not None:
                continue
            common = index.contains(grams)
            if not bool(common.any()):
                continue
            value = covered_new_tokens(len(tr_tokens[i]), common, isin_sorted(frequent, grams))
            if not fold:
                covered[i] = value
            if value >= MIN_NEW:
                reasons[i] = REASON_13GRAM
                covered[i] = value
                folded[i] = fold

    near = NearDuplicateIndex()
    for key, w_doc in enumerate(w):
        near.add(key, shingle_hashes(w_doc.tokens))
    matches: list[int | None] = [None] * len(train)
    for i, doc in enumerate(train):
        if reasons[i] is None:
            matches[i] = near.match(shingle_hashes(doc.tokens))
            if matches[i] is not None:
                reasons[i] = REASON_NEAR

    return [
        Verdict(doc.doc_id, reasons[i], covered[i], folded[i], matches[i])
        for i, doc in enumerate(train)
    ]


def _fold(tokens: Sequence[str]) -> list[str]:
    return [fold_token(token) for token in tokens]
