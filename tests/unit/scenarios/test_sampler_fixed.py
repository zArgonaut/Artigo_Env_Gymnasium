from __future__ import annotations

import numpy as np
import pytest

from artigo_env.scenarios import (
    DocColumns,
    EpochSampler,
    FilterSpec,
    FixedListSampler,
    NoEligibleDocumentsError,
    Scenario,
    get_scenario,
    make_sampler,
)

pytestmark = pytest.mark.fast

SPLITS = ["w-val", "train", "w-val", "w-test", "w-val", "ext-val", "w-val", "demo"]
DOCS = DocColumns.from_mapping(
    {
        "source_id": ["w"] * 8,
        "split": SPLITS,
        "n_body_tokens": [10, 20, 30, 40, 50, 60, 5000, 80],
        "n_title_content": [1] * 8,
        "domain": ["x"] * 8,
        "variant": [""] * 8,
        "title_in_body": [True] * 8,
    }
)


@pytest.mark.req("REQ-SCN-004")
def test_walks_the_fixed_ascending_list_and_ignores_the_generator():
    sampler = FixedListSampler(DOCS, get_scenario("eval-w-val"))
    assert sampler.indices == (0, 2, 4, 6)
    rng = np.random.default_rng(0)
    state = rng.bit_generator.state
    drawn = [sampler.next(rng) for _ in range(6)]
    assert drawn == [0, 2, 4, 6, 0, 2]
    assert rng.bit_generator.state == state  # generator untouched
    other = FixedListSampler(DOCS, get_scenario("eval-w-val"))
    assert [other.next(np.random.default_rng(99)) for _ in range(6)] == drawn


@pytest.mark.req("REQ-SCN-004")
def test_index_of_and_filters():
    sampler = FixedListSampler(DOCS, get_scenario("eval-w-val"))
    assert [sampler.index_of(i) for i in range(4)] == [0, 2, 4, 6]
    for bad in (-1, 4):
        with pytest.raises(IndexError, match="out of range"):
            sampler.index_of(bad)
    with pytest.raises(TypeError):
        sampler.index_of(True)
    filtered = Scenario(
        name="short-w-val", split="w-val", filters=(FilterSpec("body_tokens", {"max": 1000}),)
    )
    assert FixedListSampler(DOCS, filtered).indices == (0, 2, 4)
    assert sampler.stage == 0 and sampler.n_eligible == 4


@pytest.mark.req("REQ-SCN-004")
def test_make_sampler_dispatches_on_split():
    for name, expected in (("eval-w-test", (3,)), ("eval-ext-val", (5,))):
        sampler = make_sampler(DOCS, get_scenario(name), base_seed=1, worker_index=0)
        assert isinstance(sampler, FixedListSampler)
        assert sampler.indices == expected
    demo = make_sampler(DOCS, get_scenario("demo"), base_seed=1)
    assert isinstance(demo, EpochSampler)
    assert demo.next(np.random.default_rng(0)) == 7


@pytest.mark.req("REQ-SCN-004")
def test_empty_evaluation_list_is_an_error():
    docs = DocColumns.from_mapping(
        {**{k: getattr(DOCS, k) for k in DocColumns.column_names()}, "split": ["train"] * 8}
    )
    with pytest.raises(NoEligibleDocumentsError, match="eval-w-val"):
        FixedListSampler(docs, get_scenario("eval-w-val"))
