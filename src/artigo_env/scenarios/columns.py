"""Per-document metadata handed to scenario filters and samplers (SDD 08 §B.1).

``scenarios`` may not import ``corpus`` (SDD 02 §B.2), so the sampler never sees a corpus object:
the environment extracts these parallel arrays from a compiled corpus and passes them in.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import Any

import numpy as np
import numpy.typing as npt

_STR_COLUMNS = ("source_id", "split", "domain", "variant")
_INT_COLUMNS = ("n_body_tokens", "n_title_content")
_BOOL_COLUMNS = ("title_in_body",)


def _read_only(arr: npt.NDArray[Any]) -> npt.NDArray[Any]:
    view = arr.view()
    view.flags.writeable = False
    return view


def _str_column(name: str, values: Any) -> npt.NDArray[np.str_]:
    arr = np.asarray(values)
    if arr.ndim != 1:
        raise ValueError(f"column {name!r} must be one-dimensional, got shape {arr.shape}")
    if arr.size == 0:
        return np.zeros(0, dtype=np.str_)
    if arr.dtype.kind == "U":
        return arr
    if arr.dtype.kind != "O":
        raise TypeError(f"column {name!r} must contain strings, got dtype {arr.dtype}")
    items = arr.tolist()
    if name == "variant":
        items = ["" if item is None else item for item in items]
    for item in items:
        if not isinstance(item, str):
            raise TypeError(f"column {name!r} must contain strings, found {item!r}")
    return np.asarray(items, dtype=np.str_)


def _int_column(name: str, values: Any) -> npt.NDArray[np.int64]:
    arr = np.asarray(values)
    if arr.ndim != 1:
        raise ValueError(f"column {name!r} must be one-dimensional, got shape {arr.shape}")
    if arr.size == 0:
        return np.zeros(0, dtype=np.int64)
    if arr.dtype.kind not in "iu":
        raise TypeError(f"column {name!r} must contain integers, got dtype {arr.dtype}")
    out = arr.astype(np.int64, copy=False)
    if (out < 0).any():
        raise ValueError(f"column {name!r} must not contain negative values")
    return out


def _bool_column(name: str, values: Any) -> npt.NDArray[np.bool_]:
    arr = np.asarray(values)
    if arr.ndim != 1:
        raise ValueError(f"column {name!r} must be one-dimensional, got shape {arr.shape}")
    if arr.size == 0:
        return np.zeros(0, dtype=np.bool_)
    if arr.dtype.kind != "b":
        raise TypeError(f"column {name!r} must contain booleans, got dtype {arr.dtype}")
    return arr


@dataclass(frozen=True, eq=False)
class DocColumns:
    """Parallel read-only arrays, one entry per document of a compiled corpus.

    The position in the arrays is the document index returned by the samplers. ``variant`` is the
    empty string when unknown (``None`` is accepted and converted).
    """

    source_id: npt.NDArray[np.str_]
    split: npt.NDArray[np.str_]
    n_body_tokens: npt.NDArray[np.int64]
    n_title_content: npt.NDArray[np.int64]
    domain: npt.NDArray[np.str_]
    variant: npt.NDArray[np.str_]
    title_in_body: npt.NDArray[np.bool_]

    def __post_init__(self) -> None:
        converted: dict[str, npt.NDArray[Any]] = {}
        for name in _STR_COLUMNS:
            converted[name] = _str_column(name, getattr(self, name))
        for name in _INT_COLUMNS:
            converted[name] = _int_column(name, getattr(self, name))
        for name in _BOOL_COLUMNS:
            converted[name] = _bool_column(name, getattr(self, name))
        lengths = {name: len(arr) for name, arr in converted.items()}
        if len(set(lengths.values())) > 1:
            raise ValueError(f"DocColumns arrays must all have the same length, got {lengths}")
        for name, arr in converted.items():
            object.__setattr__(self, name, _read_only(arr))

    def __len__(self) -> int:
        return len(self.source_id)

    @classmethod
    def column_names(cls) -> tuple[str, ...]:
        return tuple(f.name for f in fields(cls))

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> DocColumns:
        """Build from a mapping of column name to array-like; extra keys are ignored.

        This lets a whole metadata table (for example ``pyarrow.Table.to_pydict()``) be passed.
        """
        names = cls.column_names()
        missing = [name for name in names if name not in mapping]
        if missing:
            raise ValueError(f"missing DocColumns columns: {missing}")
        return cls(**{name: mapping[name] for name in names})
