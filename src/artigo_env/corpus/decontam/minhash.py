"""Near-duplicate barrier: MinHash/LSH proposes candidates, exact Jaccard decides (SDD 06 §7.3).

Shingles are token 5-grams hashed by :func:`~artigo_env.corpus.decontam.grams.gram_hashes`. A
signature holds, for each of 128 universal hash functions ``a·x + b`` (``uint64`` arithmetic,
``a`` odd), the minimum over the shingles. LSH splits the signature into 16 bands of 8 rows;
documents sharing one band are candidates. The decision is the exact Jaccard similarity of the
shingle sets (REQ-COR-009): near 0.8 the 128-permutation estimate has a standard deviation of
about 0.035, too coarse to decide on its own.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import xxhash
from numpy.typing import NDArray

from artigo_env.corpus.decontam.grams import gram_hashes

SHINGLE = 5
N_PERM = 128
BANDS = 16
ROWS = 8
JACCARD_MIN = 0.8
SEED = 20261109

EMPTY = np.uint64(2**64 - 1)
# Bounds the (N_PERM x chunk) work matrix for very long documents.
_CHUNK = 4096


def _coefficients(name: str, odd: bool) -> NDArray[np.uint64]:
    # Derived from xxh64 rather than a NumPy generator: the values are fixed by this file alone,
    # independent of the NumPy version's random streams.
    values = [
        xxhash.xxh64_intdigest(f"minhash/{name}/{i}".encode(), seed=SEED) for i in range(N_PERM)
    ]
    array = np.array(values, dtype=np.uint64)
    if odd:
        array |= np.uint64(1)
    return array


_A = _coefficients("a", odd=True).reshape(N_PERM, 1)
_B = _coefficients("b", odd=False).reshape(N_PERM, 1)


def shingle_hashes(tokens: Sequence[str]) -> NDArray[np.uint64]:
    """Hashes of the token 5-grams of a document, unique and sorted."""
    return np.unique(gram_hashes(tokens, SHINGLE))


def signature(shingles: NDArray[np.uint64]) -> NDArray[np.uint64]:
    """MinHash signature (``N_PERM`` values); the empty set maps to all ``2**64 - 1``."""
    result = np.full(N_PERM, EMPTY, dtype=np.uint64)
    values = np.asarray(shingles, dtype=np.uint64).reshape(-1)
    for start in range(0, values.size, _CHUNK):
        block = values[start : start + _CHUNK].reshape(1, -1)
        # uint64 array arithmetic wraps modulo 2**64, which is the intended hash family.
        hashed = _A * block + _B
        np.minimum(result, hashed.min(axis=1), out=result)
    return result


def estimate_jaccard(a: NDArray[np.uint64], b: NDArray[np.uint64]) -> float:
    """Fraction of equal signature positions (the MinHash estimate; audit only)."""
    return float(np.count_nonzero(a == b)) / N_PERM


def jaccard(a: NDArray[np.uint64], b: NDArray[np.uint64]) -> float:
    """Exact Jaccard similarity of two sorted, unique hash arrays (0.0 when both are empty)."""
    common = np.intersect1d(a, b, assume_unique=True).size
    union = a.size + b.size - common
    return common / union if union else 0.0


class LSHIndex:
    """Banded LSH over signatures: ``BANDS`` bands of ``ROWS`` rows."""

    __slots__ = ("_buckets",)

    def __init__(self) -> None:
        self._buckets: list[dict[int, list[int]]] = [{} for _ in range(BANDS)]

    @staticmethod
    def _band_keys(sig: NDArray[np.uint64]) -> list[int]:
        if sig.shape != (N_PERM,):
            raise ValueError(f"signature must have shape ({N_PERM},), got {sig.shape}")
        rows = np.ascontiguousarray(sig, dtype=np.uint64).reshape(BANDS, ROWS)
        # A 64-bit digest of each band; a collision only adds a candidate, which the exact
        # Jaccard check then rejects.
        return [xxhash.xxh64_intdigest(rows[band].tobytes()) for band in range(BANDS)]

    def add(self, key: int, sig: NDArray[np.uint64]) -> None:
        for band, bucket_key in enumerate(self._band_keys(sig)):
            self._buckets[band].setdefault(bucket_key, []).append(key)

    def candidates(self, sig: NDArray[np.uint64]) -> set[int]:
        """Keys sharing at least one band with ``sig`` (sort before iterating)."""
        found: set[int] = set()
        for band, bucket_key in enumerate(self._band_keys(sig)):
            found.update(self._buckets[band].get(bucket_key, ()))
        return found


class NearDuplicateIndex:
    """Documents by integer key; ``match`` finds one with exact Jaccard ``>= JACCARD_MIN``."""

    __slots__ = ("_lsh", "_shingles", "_signatures")

    def __init__(self) -> None:
        self._lsh = LSHIndex()
        self._shingles: dict[int, NDArray[np.uint64]] = {}
        self._signatures: dict[int, NDArray[np.uint64]] = {}

    def __len__(self) -> int:
        return len(self._shingles)

    def __contains__(self, key: object) -> bool:
        return key in self._shingles

    def add(
        self, key: int, shingles: NDArray[np.uint64], sig: NDArray[np.uint64] | None = None
    ) -> None:
        """Index a document; ``sig`` may be passed when already known (it must match)."""
        key = int(key)
        if key in self._shingles:
            raise ValueError(f"duplicate key in near-duplicate index: {key}")
        shingles = np.asarray(shingles, dtype=np.uint64)
        if sig is None:
            sig = signature(shingles)
        self._shingles[key] = shingles
        self._signatures[key] = sig
        # An empty set has Jaccard 0 with everything; leaving it out of the buckets keeps all
        # short documents from piling into one bucket.
        if shingles.size:
            self._lsh.add(key, sig)

    def match(self, shingles: NDArray[np.uint64]) -> int | None:
        """Smallest key whose exact Jaccard with ``shingles`` is at least ``JACCARD_MIN``."""
        shingles = np.asarray(shingles, dtype=np.uint64)
        if shingles.size == 0:
            return None
        for key in sorted(self._lsh.candidates(signature(shingles))):
            if jaccard(shingles, self._shingles[key]) >= JACCARD_MIN:
                return key
        return None

    # --- serialization (used by WIndex) ------------------------------------------------------

    def to_arrays(self) -> dict[str, NDArray[np.uint64] | NDArray[np.int64]]:
        """Keys (ascending), CSR offsets, concatenated shingles and signatures."""
        keys = sorted(self._shingles)
        sizes = [self._shingles[key].size for key in keys]
        offsets = np.zeros(len(keys) + 1, dtype=np.int64)
        np.cumsum(sizes, out=offsets[1:])
        empty = np.empty(0, dtype=np.uint64)
        shingles = np.concatenate([self._shingles[key] for key in keys]) if keys else empty
        sigs = (
            np.stack([self._signatures[key] for key in keys])
            if keys
            else np.empty((0, N_PERM), dtype=np.uint64)
        )
        return {
            "keys": np.array(keys, dtype=np.int64),
            "offsets": offsets,
            "shingles": shingles.astype(np.uint64, copy=False),
            "signatures": sigs.astype(np.uint64, copy=False),
        }

    @classmethod
    def from_arrays(
        cls,
        keys: NDArray[np.int64],
        offsets: NDArray[np.int64],
        shingles: NDArray[np.uint64],
        signatures: NDArray[np.uint64],
    ) -> NearDuplicateIndex:
        n = keys.shape[0]
        if offsets.shape != (n + 1,) or signatures.shape != (n, N_PERM):
            raise ValueError("inconsistent near-duplicate index arrays")
        if n and (offsets[0] != 0 or offsets[-1] != shingles.size):
            raise ValueError("near-duplicate offsets do not cover the shingle array")
        index = cls()
        for i in range(n):
            index.add(
                int(keys[i]),
                np.array(shingles[offsets[i] : offsets[i + 1]], dtype=np.uint64),
                np.array(signatures[i], dtype=np.uint64),
            )
        return index
