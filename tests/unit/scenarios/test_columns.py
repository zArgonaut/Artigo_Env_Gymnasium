from __future__ import annotations

import numpy as np
import pytest

from artigo_env.scenarios import DocColumns

pytestmark = pytest.mark.fast


def _mapping(n: int = 3) -> dict[str, object]:
    return {
        "source_id": ["a"] * n,
        "split": ["train"] * n,
        "n_body_tokens": list(range(n)),
        "n_title_content": [1] * n,
        "domain": ["x"] * n,
        "variant": ["pt-BR"] * n,
        "title_in_body": [True] * n,
    }


@pytest.mark.req("REQ-SCN-005")
def test_from_mapping_builds_typed_read_only_arrays_and_ignores_extra_keys():
    data = _mapping()
    data["variant"] = ["pt-BR", None, ""]
    data["title"] = ["ignored"] * 3
    docs = DocColumns.from_mapping(data)
    assert len(docs) == 3
    assert docs.n_body_tokens.dtype == np.int64
    assert docs.title_in_body.dtype == np.bool_
    assert docs.source_id.dtype.kind == "U"
    assert docs.variant.tolist() == ["pt-BR", "", ""]
    with pytest.raises(ValueError):
        docs.n_body_tokens[0] = 5


@pytest.mark.req("REQ-SCN-005")
def test_empty_columns_are_allowed():
    docs = DocColumns.from_mapping({key: [] for key in _mapping()})
    assert len(docs) == 0


@pytest.mark.req("REQ-SCN-005")
@pytest.mark.parametrize(
    ("key", "value", "error"),
    [
        ("n_body_tokens", [1, 2], ValueError),  # length mismatch
        ("n_body_tokens", [1.0, 2.0, 3.0], TypeError),
        ("n_body_tokens", [1, -2, 3], ValueError),
        ("title_in_body", [1, 0, 1], TypeError),
        ("source_id", ["a", None, "b"], TypeError),
        ("domain", [1, 2, 3], TypeError),
    ],
)
def test_invalid_columns_are_rejected(key, value, error):
    data = _mapping()
    data[key] = value
    with pytest.raises(error):
        DocColumns.from_mapping(data)


@pytest.mark.req("REQ-SCN-005")
def test_missing_column_is_rejected():
    data = _mapping()
    del data["domain"]
    with pytest.raises(ValueError, match="domain"):
        DocColumns.from_mapping(data)
