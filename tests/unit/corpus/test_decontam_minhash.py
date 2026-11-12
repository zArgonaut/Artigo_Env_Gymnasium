"""Near-duplicate barrier: MinHash/LSH candidates, exact Jaccard decision (SDD 06 §7.3)."""

from __future__ import annotations

import numpy as np
import pytest

from artigo_env.corpus.decontam.minhash import (
    BANDS,
    JACCARD_MIN,
    N_PERM,
    ROWS,
    SEED,
    SHINGLE,
    LSHIndex,
    NearDuplicateIndex,
    estimate_jaccard,
    jaccard,
    shingle_hashes,
    signature,
)

pytestmark = pytest.mark.fast


def random_set(rng: np.random.Generator, n: int) -> np.ndarray:
    return np.unique(rng.integers(0, 2**63, size=n, dtype=np.uint64))


@pytest.mark.req("REQ-COR-009")
def test_constants():
    assert (SHINGLE, N_PERM, BANDS, ROWS, JACCARD_MIN, SEED) == (5, 128, 16, 8, 0.8, 20261109)


@pytest.mark.req("REQ-COR-009")
def test_signature_literal_values_and_empty_set():
    shingles = shingle_hashes(["o", "rio", "corre", "para", "o", "mar", "azul"])
    assert shingles.tolist() == [
        11636134312805781992,
        11762561038406843329,
        17405842161599557554,
    ]
    sig = signature(shingles)
    assert sig.dtype == np.uint64 and sig.shape == (N_PERM,)
    assert sig[:4].tolist() == [
        1840943492864531689,
        2193650825381429852,
        4364039059894790907,
        4658831893053023577,
    ]
    assert np.all(signature(np.empty(0, dtype=np.uint64)) == np.uint64(2**64 - 1))
    assert shingle_hashes(["a", "b", "c", "d"]).size == 0


@pytest.mark.req("REQ-COR-009")
def test_signature_chunking_equals_one_block():
    shingles = random_set(np.random.default_rng(1), 10_000)
    # Reference: one (N_PERM x m) block, same hash family.
    from artigo_env.corpus.decontam import minhash

    block = (minhash._A * shingles.reshape(1, -1) + minhash._B).min(axis=1)
    assert np.array_equal(signature(shingles), block)


@pytest.mark.req("REQ-COR-009")
def test_exact_jaccard():
    a = np.array([1, 2, 3, 4], dtype=np.uint64)
    b = np.array([3, 4, 5], dtype=np.uint64)
    assert jaccard(a, b) == pytest.approx(2 / 5)
    assert jaccard(a, a) == 1.0
    empty = np.empty(0, dtype=np.uint64)
    assert jaccard(empty, empty) == 0.0


@pytest.mark.req("REQ-COR-009")
def test_lsh_candidates():
    lsh = LSHIndex()
    sig = signature(random_set(np.random.default_rng(2), 300))
    other = sig.copy()
    other[ROWS:] += np.uint64(1)  # shares only the first band
    lsh.add(7, sig)
    lsh.add(3, other)
    assert lsh.candidates(sig) == {3, 7}
    far = sig + np.uint64(2)
    assert lsh.candidates(far) == set()
    with pytest.raises(ValueError):
        lsh.add(1, sig[:10])


@pytest.mark.req("REQ-COR-009")
def test_real_jaccard_0817_is_removed_when_estimate_is_below():
    # |A ∩ B| = 899, |A ∪ B| = 1101: exact Jaccard 0.8165 (0.817). Look for a draw in which the
    # 128-permutation estimate falls below 0.8 while LSH still proposes the pair.
    for seed in range(500):
        rng = np.random.default_rng(seed)
        pool = random_set(rng, 1200)[:1101]
        if pool.size < 1101:
            continue
        rng.shuffle(pool)
        a = np.sort(pool[:1000])
        b = np.sort(np.concatenate([pool[:899], pool[1000:1101]]))
        if estimate_jaccard(signature(a), signature(b)) >= JACCARD_MIN:
            continue
        index = NearDuplicateIndex()
        index.add(0, a)
        if 0 not in index._lsh.candidates(signature(b)):
            continue
        assert round(jaccard(a, b), 3) == 0.817
        assert index.match(b) == 0
        return
    pytest.fail("no draw with estimate below 0.8 and an LSH candidate")


@pytest.mark.req("REQ-COR-009", "REQ-COR-011")
def test_near_duplicate_index_returns_smallest_matching_key():
    rng = np.random.default_rng(3)
    base = random_set(rng, 500)
    near = np.unique(np.concatenate([base[:480], random_set(rng, 20)]))  # J ≈ 0.92
    index = NearDuplicateIndex()
    index.add(9, base)
    index.add(4, near)
    index.add(6, random_set(rng, 500))
    assert len(index) == 3 and 4 in index
    assert index.match(base) == 4
    assert index.match(random_set(rng, 500)) is None
    with pytest.raises(ValueError):
        index.add(9, base)


@pytest.mark.req("REQ-COR-009")
def test_below_threshold_is_not_a_match():
    rng = np.random.default_rng(4)
    pool = random_set(rng, 1300)[:1300]
    a, b = np.sort(pool[:1000]), np.sort(pool[150:1150])  # J = 850 / 1150 ≈ 0.739
    index = NearDuplicateIndex()
    index.add(0, a)
    assert jaccard(a, b) < JACCARD_MIN
    assert index.match(b) is None


@pytest.mark.req("REQ-COR-009")
def test_empty_documents_never_match():
    empty = np.empty(0, dtype=np.uint64)
    index = NearDuplicateIndex()
    index.add(0, empty)
    index.add(1, empty)
    assert index.match(empty) is None


@pytest.mark.req("REQ-COR-009")
def test_arrays_roundtrip():
    rng = np.random.default_rng(5)
    index = NearDuplicateIndex()
    sets = {2: random_set(rng, 100), 0: random_set(rng, 50), 5: np.empty(0, dtype=np.uint64)}
    for key, shingles in sets.items():
        index.add(key, shingles)
    arrays = index.to_arrays()
    assert arrays["keys"].tolist() == [0, 2, 5]
    loaded = NearDuplicateIndex.from_arrays(**arrays)
    assert len(loaded) == 3
    assert loaded.match(sets[2]) == 2
    with pytest.raises(ValueError):
        NearDuplicateIndex.from_arrays(
            arrays["keys"], arrays["offsets"][:-1], arrays["shingles"], arrays["signatures"]
        )
