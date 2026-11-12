from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pytest

from artigo_env.core.errors import UnknownNameError
from artigo_env.scenarios import (
    SCENARIO_FILTERS,
    DocColumns,
    FilterSpec,
    Scenario,
    apply_filters,
    build_filter,
    check_params,
    make_sampler,
)

pytestmark = pytest.mark.fast

DOCS = DocColumns.from_mapping(
    {
        "source_id": ["wiki", "wiki", "news", "news", "books", "books"],
        "split": ["train"] * 6,
        "n_body_tokens": [100, 800, 801, 2000, 5000, 0],
        "n_title_content": [1, 2, 3, 4, 7, 1],
        "domain": ["sci", "art", "sci", "pol", "art", "sci"],
        "variant": ["pt-BR", "pt-PT", "", "pt-BR", "pt-PT", ""],
        "title_in_body": [True, False, True, False, True, False],
    }
)


def _mask(kind: str, **params: Any) -> list[bool]:
    return build_filter(FilterSpec(kind, params)).mask(DOCS).tolist()


@pytest.mark.req("REQ-SCN-005")
def test_body_tokens_bounds_are_inclusive_and_optional():
    assert _mask("body_tokens", max=800) == [True, True, False, False, False, True]
    assert _mask("body_tokens", min=801) == [False, False, True, True, True, False]
    assert _mask("body_tokens", min=800, max=2000) == [False, True, True, True, False, False]
    assert _mask("body_tokens") == [True] * 6


@pytest.mark.req("REQ-SCN-005")
def test_title_content_words():
    assert _mask("title_content_words", max=1) == [True, False, False, False, False, True]
    assert _mask("title_content_words", min=2, max=3) == [False, True, True, False, False, False]


@pytest.mark.req("REQ-SCN-005")
def test_membership_filters():
    assert _mask("domain", **{"in": ["sci"]}) == [True, False, True, False, False, True]
    assert _mask("variant", **{"in": [""]}) == [False, False, True, False, False, True]
    assert _mask("source", **{"in": ["news", "books"]}) == [False, False, True, True, True, True]


@pytest.mark.req("REQ-SCN-005")
def test_title_in_body():
    assert _mask("title_in_body", value=True) == [True, False, True, False, True, False]
    assert _mask("title_in_body", value=False) == [False, True, False, True, False, True]


@pytest.mark.req("REQ-SCN-005")
def test_apply_filters_is_a_logical_and_and_empty_keeps_all():
    specs = (FilterSpec("domain", {"in": ["sci"]}), FilterSpec("body_tokens", {"max": 800}))
    assert apply_filters(specs, DOCS).tolist() == [True, False, False, False, False, True]
    assert apply_filters((), DOCS).tolist() == [True] * 6


@pytest.mark.req("REQ-SCN-005")
@pytest.mark.parametrize(
    ("kind", "params"),
    [
        ("body_tokens", {"minimum": 3}),
        ("body_tokens", {"min": -1}),
        ("body_tokens", {"min": 5, "max": 4}),
        ("body_tokens", {"max": 2.5}),
        ("body_tokens", {"max": True}),
        ("domain", {}),
        ("domain", {"in": "sci"}),
        ("domain", {"in": []}),
        ("variant", {"in": [1]}),
        ("title_in_body", {"value": 1}),
        ("title_in_body", {}),
    ],
)
def test_invalid_parameters_fail_at_construction(kind, params):
    with pytest.raises(ValueError, match=kind):
        FilterSpec(kind, params)


@pytest.mark.req("REQ-SCN-005")
def test_unknown_kind_is_rejected_with_the_available_names():
    with pytest.raises(UnknownNameError, match="body_tokens"):
        FilterSpec("no-such-filter", {})


class _EvenBodyFilter:
    def __init__(self, params: Mapping[str, Any]) -> None:
        check_params("test-even-body", params, ())

    def mask(self, docs: DocColumns) -> np.ndarray:
        return docs.n_body_tokens % 2 == 0


if "test-even-body" not in SCENARIO_FILTERS:
    SCENARIO_FILTERS.register("test-even-body", _EvenBodyFilter)


@pytest.mark.req("REQ-SCN-005")
def test_a_filter_registered_in_a_test_works_in_a_scenario():
    scenario = Scenario(name="t", split="train", filters=(FilterSpec("test-even-body"),))
    assert Scenario.from_dict(scenario.to_dict()) == scenario
    sampler = make_sampler(DOCS, scenario, base_seed=0)
    rng = np.random.default_rng(0)
    drawn = {sampler.next(rng) for _ in range(50)}
    assert drawn == {0, 1, 3, 4, 5}  # even body counts: 100, 800, 2000, 5000, 0
    with pytest.raises(ValueError, match="unknown parameter"):
        FilterSpec("test-even-body", {"x": 1})


@pytest.mark.req("REQ-SCN-005")
def test_filter_returning_a_bad_mask_is_reported():
    if "test-bad-mask" not in SCENARIO_FILTERS:

        class _Bad:
            def __init__(self, params: Mapping[str, Any]) -> None:
                pass

            def mask(self, docs: DocColumns) -> np.ndarray:
                return np.ones(len(docs) + 1, dtype=bool)

        SCENARIO_FILTERS.register("test-bad-mask", _Bad)
    with pytest.raises(ValueError, match="test-bad-mask"):
        apply_filters((FilterSpec("test-bad-mask"),), DOCS)
