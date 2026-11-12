from __future__ import annotations

import subprocess
import sys
from collections import Counter

import numpy as np
import pytest

from artigo_env.scenarios import (
    CurriculumStage,
    DocColumns,
    EpochSampler,
    FilterSpec,
    NoEligibleDocumentsError,
    Scenario,
    resolve_source_weights,
)

pytestmark = pytest.mark.fast


def make_docs(
    sources: dict[str, int], *, split: str = "train", titles: list[int] | None = None
) -> DocColumns:
    source_id = [name for name, n in sources.items() for _ in range(n)]
    n = len(source_id)
    return DocColumns.from_mapping(
        {
            "source_id": source_id,
            "split": [split] * n,
            "n_body_tokens": [(i * 37) % 3000 for i in range(n)],
            "n_title_content": titles if titles is not None else [1 + i % 4 for i in range(n)],
            "domain": ["x"] * n,
            "variant": [""] * n,
            "title_in_body": [True] * n,
        }
    )


TINY = make_docs({"a": 8}, titles=[1, 2, 3, 4, 1, 2, 3, 5])
TRAIN = Scenario(name="t", split="train")


def _draw(sampler: EpochSampler, count: int, seed: int = 0) -> list[int]:
    rng = np.random.default_rng(seed)
    return [sampler.next(rng) for _ in range(count)]


def _worker_epoch_sizes(n: int, workers: int) -> list[int]:
    return [len(range(w, n, workers)) for w in range(workers)]


# --- REQ-SCN-001 -----------------------------------------------------------------------------


@pytest.mark.req("REQ-SCN-001")
def test_no_document_repeats_inside_an_epoch_and_next_epoch_reorders():
    docs = make_docs({"a": 50})
    sampler = EpochSampler(docs, TRAIN, base_seed=7)
    first = _draw(sampler, 50)
    second = _draw(sampler, 50, seed=1)
    assert sorted(first) == list(range(50))
    assert sorted(second) == list(range(50))
    assert first != second


@pytest.mark.req("REQ-SCN-001")
def test_union_of_workers_covers_each_source_exactly_once_per_epoch():
    docs = make_docs({"a": 23, "b": 10})
    scenario = Scenario(name="t", split="train", source_mix={"a": 1.0, "b": 1.0})
    workers = 3
    per_source: dict[str, Counter[int]] = {"a": Counter(), "b": Counter()}
    for w in range(workers):
        sampler = EpochSampler(docs, scenario, base_seed=11, worker_index=w, num_workers=workers)
        rng = np.random.default_rng(100 + w)
        sizes = {s: _worker_epoch_sizes(n, workers)[w] for s, n in (("a", 23), ("b", 10))}
        taken = {"a": 0, "b": 0}
        while taken["a"] < sizes["a"] or taken["b"] < sizes["b"]:
            index = sampler.next(rng)
            source = str(docs.source_id[index])
            if taken[source] < sizes[source]:
                per_source[source][index] += 1
                taken[source] += 1
    assert per_source["a"] == Counter(range(23))
    assert per_source["b"] == Counter(range(23, 33))


# --- REQ-SCN-002 -----------------------------------------------------------------------------

EXPECTED_NATURAL = [2, 5, 1, 6, 0, 3, 7, 4, 5, 2, 1, 6, 3, 4, 0, 7]
EXPECTED_MATCH_W = [1, 2, 6, 5, 7, 4, 0, 3, 6, 1, 3, 0, 7, 5, 2, 4]

_SUBPROCESS = """
import numpy as np
from artigo_env.scenarios import DocColumns, EpochSampler, Scenario
n = 8
docs = DocColumns.from_mapping(dict(
    source_id=['a'] * n, split=['train'] * n, n_body_tokens=list(range(n)),
    n_title_content=[1, 2, 3, 4, 1, 2, 3, 5], domain=['x'] * n, variant=[''] * n,
    title_in_body=[True] * n))
sampler = EpochSampler(docs, Scenario(name='t', split='train'), base_seed=123)
rng = np.random.default_rng(0)
print(','.join(str(sampler.next(rng)) for _ in range(16)))
"""


@pytest.mark.req("REQ-SCN-002")
def test_literal_sequence_for_a_tiny_example():
    assert _draw(EpochSampler(TINY, TRAIN, base_seed=123), 16) == EXPECTED_NATURAL
    match_w = Scenario(name="t", split="train", title_distribution="match-w")
    sampler = EpochSampler(TINY, match_w, base_seed=123, target_title_shares=[0.25] * 4)
    assert _draw(sampler, 16) == EXPECTED_MATCH_W


@pytest.mark.req("REQ-SCN-002")
def test_same_sequence_in_a_fresh_process():
    out = subprocess.run(
        [sys.executable, "-c", _SUBPROCESS], capture_output=True, text=True, check=True
    ).stdout.strip()
    assert [int(x) for x in out.split(",")] == EXPECTED_NATURAL


@pytest.mark.req("REQ-SCN-002")
def test_same_seeds_same_sequence_and_different_base_seed_differs():
    docs = make_docs({"a": 40, "b": 30})
    one = _draw(EpochSampler(docs, TRAIN, base_seed=5), 200)
    two = _draw(EpochSampler(docs, TRAIN, base_seed=5), 200)
    other = _draw(EpochSampler(docs, TRAIN, base_seed=6), 200)
    assert one == two
    assert one != other
    a = EpochSampler(docs, TRAIN, base_seed=5)
    b = EpochSampler(docs, TRAIN, base_seed=6)
    assert not np.array_equal(a.epoch_order("a", 0), b.epoch_order("a", 0))


# --- REQ-SCN-003 -----------------------------------------------------------------------------


@pytest.mark.req("REQ-SCN-003")
@pytest.mark.parametrize("n", [40, 37, 3])
def test_four_workers_draw_disjoint_sets_whose_union_is_the_eligible_set(n):
    docs = make_docs({"a": n})
    sets = []
    for w, size in enumerate(_worker_epoch_sizes(n, 4)):
        sampler = EpochSampler(docs, TRAIN, base_seed=3, worker_index=w, num_workers=4)
        drawn = _draw(sampler, size, seed=w) if size else []
        assert len(set(drawn)) == len(drawn)
        sets.append(set(drawn))
    for i in range(4):
        for j in range(i + 1, 4):
            assert not sets[i] & sets[j]
    assert set().union(*sets) == set(range(n))


@pytest.mark.req("REQ-SCN-003")
def test_worker_past_a_small_source_still_draws():
    docs = make_docs({"a": 3})
    sampler = EpochSampler(docs, TRAIN, base_seed=3, worker_index=3, num_workers=4)
    order = sampler.epoch_order("a", 0)
    first = _draw(sampler, 1)
    assert first == [int(order[0])]  # position 3 % 3 == 0, one document per epoch


# --- source mix ------------------------------------------------------------------------------


@pytest.mark.req("REQ-SCN-002")
def test_explicit_weights_are_respected():
    docs = make_docs({"a": 100, "b": 100, "c": 100})
    scenario = Scenario(name="t", split="train", source_mix={"a": 0.6, "b": 0.3, "c": 0.1})
    sampler = EpochSampler(docs, scenario, base_seed=1)
    counts = Counter(str(docs.source_id[i]) for i in _draw(sampler, 20_000))
    for source, share in (("a", 0.6), ("b", 0.3), ("c", 0.1)):
        tolerance = 4 * (share * (1 - share) / 20_000) ** 0.5
        assert abs(counts[source] / 20_000 - share) < tolerance


@pytest.mark.req("REQ-SCN-002")
def test_automatic_mix_caps_each_source_at_half_and_renormalizes():
    # Sizes 1000/100/10: "a" is capped at 0.5; the remaining 0.5 is split 100:10 between the
    # others, so b = 0.5 * 100/110 = 0.454545..., c = 0.5 * 10/110 = 0.0454545...
    expected = {"a": 0.5, "b": 0.5 * 100 / 110, "c": 0.5 * 10 / 110}
    docs = make_docs({"a": 1000, "b": 100, "c": 10})
    sampler = EpochSampler(docs, TRAIN, base_seed=2)
    assert sampler.weights == pytest.approx(expected)
    counts = Counter(str(docs.source_id[i]) for i in _draw(sampler, 20_000))
    for source, share in expected.items():
        tolerance = 4 * (share * (1 - share) / 20_000) ** 0.5
        assert abs(counts[source] / 20_000 - share) < tolerance


@pytest.mark.req("REQ-SCN-002")
def test_resolve_source_weights_edge_cases():
    assert resolve_source_weights({"a": 5}, {}, 0.5) == {"a": 1.0}
    assert resolve_source_weights({"a": 5, "b": 1}, {}, 0.4) == {"a": 0.5, "b": 0.5}
    assert resolve_source_weights({"a": 5, "b": 0}, {"a": 1.0, "b": 3.0}, 0.5) == {"a": 1.0}
    with pytest.raises(NoEligibleDocumentsError):
        resolve_source_weights({"a": 5}, {"b": 1.0}, 0.5)


@pytest.mark.req("REQ-SCN-002")
def test_source_without_eligible_documents_is_never_drawn():
    docs = make_docs({"a": 20, "b": 20})
    scenario = Scenario(
        name="t",
        split="train",
        source_mix={"a": 1.0, "b": 1.0, "ghost": 5.0},
        filters=(FilterSpec("source", {"in": ["a", "ghost"]}),),
    )
    sampler = EpochSampler(docs, scenario, base_seed=0)
    assert sampler.weights == {"a": 1.0}
    assert sampler.n_eligible == 20
    assert {str(docs.source_id[i]) for i in _draw(sampler, 500)} == {"a"}


# --- match-w ---------------------------------------------------------------------------------


@pytest.mark.req("REQ-SCN-001")
def test_match_w_keeps_epochs_complete_and_moves_early_draws_towards_the_target():
    rng = np.random.default_rng(42)
    natural = np.array([0.7, 0.2, 0.07, 0.03])
    target = np.array([0.25, 0.3, 0.25, 0.2])
    titles = (rng.choice(4, size=5000, p=natural) + 1).tolist()
    docs = make_docs({"a": 5000}, titles=titles)
    scenario = Scenario(name="t", split="train", title_distribution="match-w")
    sampler = EpochSampler(docs, scenario, base_seed=9, target_title_shares=target.tolist())
    drawn = _draw(sampler, 5000)
    assert sorted(drawn) == list(range(5000))
    head = np.asarray(titles)[drawn[:500]] - 1
    shares = np.bincount(head, minlength=4) / 500
    source_shares = np.bincount(np.asarray(titles) - 1, minlength=4) / 5000
    assert np.abs(shares - target).sum() < np.abs(shares - source_shares).sum()


@pytest.mark.req("REQ-SCN-001")
def test_match_w_requires_valid_target_shares():
    scenario = Scenario(name="t", split="train", title_distribution="match-w")
    with pytest.raises(ValueError, match="target_title_shares"):
        EpochSampler(TINY, scenario, base_seed=0)
    with pytest.raises(ValueError, match="target_title_shares"):
        EpochSampler(TINY, scenario, base_seed=0, target_title_shares=[0.5, 0.5])
    # a zero target bucket: those documents come last but still appear once per epoch
    sampler = EpochSampler(TINY, scenario, base_seed=0, target_title_shares=[0, 1, 1, 1])
    drawn = _draw(sampler, 8)
    assert sorted(drawn) == list(range(8))
    assert set(drawn[-2:]) == {0, 4}


# --- curriculum ------------------------------------------------------------------------------


def _curriculum() -> Scenario:
    return Scenario(
        name="c",
        split="train",
        curriculum=(
            CurriculumStage(
                filters=(FilterSpec("title_content_words", {"max": 1}),), until_steps=5
            ),
            CurriculumStage(filters=(FilterSpec("title_content_words", {"max": 2}),)),
            CurriculumStage(),
        ),
    )


@pytest.mark.req("REQ-SCN-005")
def test_stage_filters_narrow_the_eligible_set_and_steps_advance():
    docs = make_docs({"a": 40})
    sampler = EpochSampler(docs, _curriculum(), base_seed=0)
    assert (sampler.stage, sampler.n_eligible) == (0, 10)
    assert {int(docs.n_title_content[i]) for i in _draw(sampler, 30)} == {1}
    sampler.notify_steps(3)
    assert sampler.stage == 0
    sampler.notify_steps(4)  # 7 >= 5: stage 1, two steps carried over
    assert (sampler.stage, sampler.n_eligible) == (1, 20)
    assert {int(docs.n_title_content[i]) for i in _draw(sampler, 60)} == {1, 2}
    sampler.notify_steps(10_000)  # stage 1 has no until_steps
    assert sampler.stage == 1
    sampler.set_stage(2)
    assert (sampler.stage, sampler.n_eligible) == (2, 40)
    sampler.set_stage(0)
    assert sampler.stage == 0


@pytest.mark.req("REQ-SCN-005")
def test_stage_out_of_range_and_empty_stage_are_errors():
    docs = make_docs({"a": 40})
    sampler = EpochSampler(docs, _curriculum(), base_seed=0)
    for bad in (-1, 3):
        with pytest.raises(ValueError, match="stage"):
            sampler.set_stage(bad)
    with pytest.raises(ValueError):
        EpochSampler(docs, TRAIN, base_seed=0).set_stage(1)
    empty = Scenario(
        name="e",
        split="train",
        curriculum=(
            CurriculumStage(filters=(FilterSpec("body_tokens", {"min": 10**9}),)),
            CurriculumStage(),
        ),
    )
    with pytest.raises(NoEligibleDocumentsError, match="stage 0"):
        EpochSampler(docs, empty, base_seed=0)


@pytest.mark.req("REQ-SCN-001")
def test_construction_errors():
    with pytest.raises(NoEligibleDocumentsError):
        EpochSampler(make_docs({"a": 5}, split="demo"), TRAIN, base_seed=0)
    with pytest.raises(ValueError):
        EpochSampler(TINY, TRAIN, base_seed=-1)
    with pytest.raises(ValueError):
        EpochSampler(TINY, TRAIN, base_seed=0, worker_index=4, num_workers=4)
    with pytest.raises(TypeError):
        EpochSampler({"source_id": []}, TRAIN, base_seed=0)  # type: ignore[arg-type]
