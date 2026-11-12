"""13-gram barrier: coverage rule, W index and three-pass protocol (SDD 06 §7.2)."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from artigo_env.corpus.decontam.coverage import (
    FREQ_LIMIT,
    MIN_NEW,
    N_GRAM,
    DocFrequency,
    TrainDoc,
    WDoc,
    WGramIndex,
    covered_new_tokens,
    decontaminate,
    hashes_of_interest,
    is_contaminated,
)
from artigo_env.corpus.decontam.grams import gram_hashes

pytestmark = [pytest.mark.fast, pytest.mark.req("REQ-COR-008")]


def words(prefix: str, n: int) -> list[str]:
    return [f"{prefix}{i}" for i in range(n)]


def test_constants():
    assert (N_GRAM, FREQ_LIMIT, MIN_NEW) == (13, 10, 7)


def _naive_coverage(n_tokens, common, frequent, n=N_GRAM):
    # The reference script's set formulation.
    in_frequent, in_common = set(), set()
    for p in range(len(common)):
        span = range(p, p + n)
        if frequent[p]:
            in_frequent.update(span)
        elif common[p]:
            in_common.update(span)
    return len(in_common - in_frequent)


@given(
    st.integers(0, 60).flatmap(
        lambda n: st.tuples(
            st.just(n),
            st.lists(st.booleans(), min_size=max(n - 12, 0), max_size=max(n - 12, 0)),
            st.lists(st.booleans(), min_size=max(n - 12, 0), max_size=max(n - 12, 0)),
        )
    )
)
def test_coverage_matches_set_formulation(case):
    n_tokens, common, frequent = case
    c = np.array(common, dtype=bool)
    f = np.array(frequent, dtype=bool)
    assert covered_new_tokens(n_tokens, c, f) == _naive_coverage(n_tokens, common, frequent)


def test_coverage_hand_cases():
    # One common gram covers 13 tokens; a frequent gram overlapping 8 of them leaves 5.
    common = np.zeros(20, dtype=bool)
    frequent = np.zeros(20, dtype=bool)
    common[0] = True
    assert covered_new_tokens(32, common, frequent) == 13
    frequent[5] = True
    assert covered_new_tokens(32, common, frequent) == 5
    # A gram that is both common and frequent is ignored as common.
    common[:] = False
    common[5] = True
    assert covered_new_tokens(32, common, frequent) == 0
    with pytest.raises(ValueError):
        covered_new_tokens(10, np.zeros(3, dtype=bool), np.zeros(3, dtype=bool))
    assert covered_new_tokens(5, np.zeros(0, dtype=bool), np.zeros(0, dtype=bool)) == 0


def test_w_index_build_contains_and_roundtrip(tmp_path):
    a, b = words("a", 20), words("b", 15)
    index = WGramIndex.build([a, b, a])
    assert len(index) == 8 + 3
    assert np.all(index.hashes[1:] > index.hashes[:-1])
    probe = np.concatenate([gram_hashes(b, N_GRAM), gram_hashes(words("c", 14), N_GRAM)])
    assert index.contains(probe).tolist() == [True] * 3 + [False] * 2
    path = tmp_path / "grams.npy"
    index.save(path)
    loaded = WGramIndex.load(path)
    assert np.array_equal(loaded.hashes, index.hashes)
    assert len(WGramIndex.build([])) == 0
    assert WGramIndex().contains(probe).tolist() == [False] * 5


def test_w_index_load_rejects_unsorted(tmp_path):
    path = tmp_path / "bad.npy"
    np.save(path, np.array([5, 3], dtype=np.uint64))
    with pytest.raises(ValueError):
        WGramIndex.load(path)
    np.save(path, np.array([1.0, 2.0]))
    with pytest.raises(ValueError):
        WGramIndex.load(path)


def test_pass_a_returns_none_without_overlap():
    w = WGramIndex.build([words("w", 30)])
    assert hashes_of_interest(words("x", 30), w) is None
    doc = words("x", 10) + words("w", 13) + words("y", 10)
    interest = hashes_of_interest(doc, w)
    assert interest is not None
    assert np.array_equal(interest, np.unique(gram_hashes(doc, N_GRAM)))


def test_doc_frequency_counts_once_per_document():
    phrase = words("p", 13)
    doc = phrase + words("x", 3) + phrase  # the phrase twice in one document
    interest = np.unique(gram_hashes(doc, N_GRAM))
    freq = DocFrequency(interest)
    for _ in range(FREQ_LIMIT):
        freq.add(doc)
    target = gram_hashes(phrase, N_GRAM)[0]
    assert freq.counts[np.searchsorted(freq.interest, target)] == FREQ_LIMIT
    assert freq.frequent().size == 0
    other = DocFrequency(interest)
    other.add(phrase)
    freq.merge(other)
    assert freq.frequent().tolist() == [int(target)]
    with pytest.raises(ValueError):
        freq.merge(DocFrequency(interest[:1]))


def test_pass_c_threshold():
    w_doc = words("w", 40)
    w = WGramIndex.build([w_doc])
    none = np.empty(0, dtype=np.uint64)
    excerpt = words("x", 20) + w_doc[5:18] + words("y", 20)  # one 13-gram: 13 tokens
    assert is_contaminated(excerpt, w, none) == (True, 13)
    short = words("x", 20) + w_doc[5:17] + words("y", 20)
    assert is_contaminated(short, w, none) == (False, 0)
    # The shared gram sits inside a frequent gram: nothing new is covered.
    frequent = np.unique(gram_hashes(excerpt, N_GRAM))
    assert is_contaminated(excerpt, w, frequent) == (False, 0)


def test_extending_a_frequent_phrase_by_one_word_is_not_contamination():
    # The case that broke the naive rule: a frequent 14-token phrase in W and in many training
    # documents; one document extends it with a coincident word of W.
    phrase = words("p", 14)
    w_doc = words("a", 30) + phrase + ["z"] + words("b", 30)
    clean = [
        TrainDoc(f"c{k}", f"T{k}", words(f"x{k}_", 20) + phrase + words(f"y{k}_", 20))
        for k in range(FREQ_LIMIT + 1)
    ]
    extended = TrainDoc("e", "E", words("u", 20) + phrase + ["z"] + words("v", 20))
    verdicts = decontaminate([*clean, extended], [WDoc("W", w_doc)])
    assert [v.reason for v in verdicts] == [None] * (FREQ_LIMIT + 2)
    assert verdicts[-1].covered == 1  # only the coincident word is new


def test_decontaminate_reasons_and_order():
    w_tokens = words("w", 200)
    w = [WDoc("Lisboa", w_tokens)]
    train = [
        TrainDoc("title", "lisboa", w_tokens),  # the title barrier decides first
        TrainDoc("gram", "Outro", words("x", 50) + w_tokens[10:30] + words("y", 50)),
        TrainDoc("clean", "Limpo", words("z", 100)),
    ]
    verdicts = decontaminate(train, w)
    assert [(v.doc_id, v.reason) for v in verdicts] == [
        ("title", "w_title"),
        ("gram", "w_13gram"),
        ("clean", None),
    ]
    assert verdicts[1].covered == 20
    assert verdicts[0].contaminated and not verdicts[2].contaminated
