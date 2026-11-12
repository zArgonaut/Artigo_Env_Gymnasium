"""Gram hashing for the decontamination barriers (SDD 06 §7.2)."""

from __future__ import annotations

import os
import subprocess
import sys

import numpy as np
import pytest
import xxhash
from hypothesis import given
from hypothesis import strategies as st

from artigo_env.corpus.decontam.grams import fold_accents, gram_hashes, gram_tokens, isin_sorted

pytestmark = [pytest.mark.fast, pytest.mark.req("REQ-COR-008")]


def test_literal_values_are_fixed():
    # xxh64("a") with seed 0 is the published test vector 0xd24ec4f1a98c6e5b.
    assert gram_hashes(["a"], 1).tolist() == [0xD24EC4F1A98C6E5B]
    assert gram_hashes(["coração", "de", "pedra"], 2).tolist() == [
        17405568364881090674,
        14007373863201427099,
    ]


def test_hashes_are_stable_across_processes():
    code = (
        "from artigo_env.corpus.decontam.grams import gram_hashes;"
        "print(gram_hashes(['coração', 'de', 'pedra'], 2).tolist())"
    )
    for seed in ("0", "12345"):
        out = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            check=True,
            env={**os.environ, "PYTHONHASHSEED": seed},
        ).stdout
        assert out.strip() == "[17405568364881090674, 14007373863201427099]"


def test_one_hash_per_start_position_and_empty_when_short():
    tokens = [f"t{i}" for i in range(20)]
    assert gram_hashes(tokens, 13).shape == (8,)
    assert gram_hashes(tokens[:12], 13).dtype == np.uint64
    assert gram_hashes(tokens[:12], 13).size == 0
    assert gram_hashes([], 1).size == 0
    with pytest.raises(ValueError):
        gram_hashes(tokens, 0)


def test_separator_keeps_token_boundaries():
    assert gram_hashes(["ab", "c"], 2)[0] != gram_hashes(["a", "bc"], 2)[0]


@given(st.lists(st.text(min_size=1, max_size=6), max_size=30), st.integers(1, 6))
def test_equals_joining_each_gram(tokens, n):
    expected = [
        xxhash.xxh64_intdigest("\x1f".join(tokens[i : i + n]).encode("utf-8"))
        for i in range(len(tokens) - n + 1)
    ]
    assert gram_hashes(tokens, n).tolist() == expected


def test_gram_tokens_lowercase_and_keep_accents():
    assert gram_tokens("Coração de PEDRA, al-Khwarizmi") == [
        "coração",
        "de",
        "pedra",
        "al",
        "khwarizmi",
    ]


def test_fold_accents():
    assert fold_accents("coração") == "coracao"
    assert fold_accents("Árvore") == "Arvore"
    assert fold_accents("rio") == "rio"


def test_isin_sorted():
    values = np.array([3, 7, 11], dtype=np.uint64)
    queries = np.array([11, 2, 7, 12, 3], dtype=np.uint64)
    assert isin_sorted(values, queries).tolist() == [True, False, True, False, True]
    assert isin_sorted(np.empty(0, dtype=np.uint64), queries).tolist() == [False] * 5
