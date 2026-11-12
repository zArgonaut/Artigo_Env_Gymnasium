"""Token n-gram hashing shared by the 13-gram barrier and MinHash shingles (SDD 06 §7.2).

A gram is a run of ``n`` consecutive tokens: the core tokenizer's forms, lowercased, with
accents kept. Its hash is ``xxh64`` (seed 0) of the tokens joined by ``"\\x1f"`` (unit
separator) and encoded in UTF-8, so it is the same in every process and on every platform.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Sequence
from itertools import accumulate

import numpy as np
import xxhash
from numpy.typing import NDArray

from artigo_env.text.normalize import lower
from artigo_env.text.tokenizer import tokenize

SEPARATOR = "\x1f"
_SEP_BYTES = len(SEPARATOR.encode("utf-8"))


def gram_tokens(text: str, hyphen: str = "split") -> list[str]:
    """Tokens of ``text`` as the barriers see them: core tokenizer forms, lowercased."""
    return [lower(token.text) for token in tokenize(text, hyphen)]


def gram_hashes(tokens: Sequence[str], n: int) -> NDArray[np.uint64]:
    """``xxh64`` of every ``n``-gram of ``tokens``, one per start position (empty if too short).

    Entry ``i`` is the hash of ``"\\x1f".join(tokens[i:i + n])`` in UTF-8.
    """
    if n < 1:
        raise ValueError(f"gram length must be positive, got {n}")
    count = len(tokens) - n + 1
    if count <= 0:
        return np.empty(0, dtype=np.uint64)
    # Encode the whole document once and hash byte slices of it: equivalent to joining each
    # gram separately, without building ``count`` intermediate strings.
    buffer = memoryview(SEPARATOR.join(tokens).encode("utf-8"))
    lengths = [len(token.encode("utf-8")) for token in tokens]
    starts = [0, *accumulate(length + _SEP_BYTES for length in lengths)]
    digest = xxhash.xxh64_intdigest
    out = [digest(buffer[starts[i] : starts[i + n] - _SEP_BYTES]) for i in range(count)]
    return np.array(out, dtype=np.uint64)


def fold_accents(token: str) -> str:
    """``token`` without combining marks (NFD, drop category ``Mn``, NFC). Audit use only."""
    decomposed = unicodedata.normalize("NFD", token)
    stripped = "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")
    return unicodedata.normalize("NFC", stripped)


def isin_sorted(
    sorted_values: NDArray[np.uint64], queries: NDArray[np.uint64]
) -> NDArray[np.bool_]:
    """Membership of each query in ``sorted_values`` (ascending, unique) by binary search."""
    if sorted_values.size == 0 or queries.size == 0:
        return np.zeros(queries.shape, dtype=np.bool_)
    positions = np.searchsorted(sorted_values, queries)
    positions[positions == sorted_values.size] = 0
    result: NDArray[np.bool_] = sorted_values[positions] == queries
    return result
