"""Registered scenario filters (SDD 08 §B.1, REQ-SCN-005).

A filter kind is a factory registered in ``SCENARIO_FILTERS``: it receives the ``params`` of a
``FilterSpec``, validates them and returns an object with ``mask(docs) -> NDArray[bool]``. New kinds
can be registered without touching the sampler.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any, ClassVar, Protocol

import numpy as np
import numpy.typing as npt

from artigo_env.core.config import ConfigBase, canonical_json
from artigo_env.core.registry import Registry
from artigo_env.scenarios.columns import DocColumns


class FrozenMapping(Mapping[str, Any]):
    """Immutable, hashable and picklable string-keyed mapping used inside frozen configs."""

    __slots__ = ("_data",)

    def __init__(self, data: Mapping[str, Any] | None = None) -> None:
        self._data: dict[str, Any] = dict(data or {})

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __hash__(self) -> int:
        return hash(canonical_json(self._data))

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._data!r})"

    def __reduce__(self) -> tuple[type[FrozenMapping], tuple[dict[str, Any]]]:
        return (type(self), (self._data,))


def freeze(value: Any) -> Any:
    """Lists become tuples and mappings become ``FrozenMapping``, recursively."""
    if isinstance(value, Mapping):
        return FrozenMapping({key: freeze(item) for key, item in value.items()})
    if isinstance(value, list | tuple):
        return tuple(freeze(item) for item in value)
    return value


class ScenarioFilter(Protocol):
    def mask(self, docs: DocColumns) -> npt.NDArray[np.bool_]:
        """Boolean array of ``len(docs)``: True for documents that pass the filter."""
        ...


class FilterFactory(Protocol):
    def __call__(self, params: Mapping[str, Any]) -> ScenarioFilter:
        """Validate ``params`` (raise ``ValueError``) and return the filter."""
        ...


SCENARIO_FILTERS: Registry[FilterFactory] = Registry("scenario filter")


def check_params(
    kind: str, params: Mapping[str, Any], allowed: Iterable[str], required: Iterable[str] = ()
) -> None:
    """Raise ``ValueError`` naming unknown or missing parameters of filter ``kind``."""
    allowed_set = set(allowed)
    unknown = sorted(set(params) - allowed_set)
    if unknown:
        raise ValueError(
            f"unknown parameter(s) {unknown} for scenario filter {kind!r}; "
            f"accepted: {sorted(allowed_set)}"
        )
    missing = sorted(set(required) - set(params))
    if missing:
        raise ValueError(f"missing parameter(s) {missing} for scenario filter {kind!r}")


@dataclass(frozen=True)
class FilterSpec(ConfigBase):
    """Declarative filter: ``kind`` names a ``SCENARIO_FILTERS`` entry, ``params`` its arguments.

    Construction builds the filter once, so an unknown kind or parameter fails here.
    """

    kind: str
    params: Mapping[str, Any] = field(default_factory=FrozenMapping)

    def __post_init__(self) -> None:
        if not isinstance(self.kind, str) or not self.kind:
            raise ValueError(f"filter kind must be a non-empty string, got {self.kind!r}")
        if not isinstance(self.params, Mapping):
            raise TypeError(f"filter params must be a mapping, got {type(self.params).__name__}")
        for key in self.params:
            if not isinstance(key, str):
                raise TypeError(f"filter parameter names must be str, got {key!r}")
        object.__setattr__(self, "params", freeze(self.params))
        build_filter(self)


def build_filter(spec: FilterSpec) -> ScenarioFilter:
    """Instantiate the filter described by ``spec`` (``UnknownNameError`` for an unknown kind)."""
    factory = SCENARIO_FILTERS.get(spec.kind)
    return factory(spec.params)


def apply_filters(specs: Iterable[FilterSpec], docs: DocColumns) -> npt.NDArray[np.bool_]:
    """Logical AND of the masks of ``specs``; no filter keeps every document."""
    result = np.ones(len(docs), dtype=np.bool_)
    for spec in specs:
        mask = np.asarray(build_filter(spec).mask(docs))
        if mask.shape != (len(docs),) or mask.dtype != np.bool_:
            raise ValueError(
                f"scenario filter {spec.kind!r} returned shape {mask.shape} dtype {mask.dtype}; "
                f"expected a boolean array of length {len(docs)}"
            )
        result &= mask
    return result


# --- built-in kinds --------------------------------------------------------------------------


def _optional_count(kind: str, name: str, value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{kind}.{name} must be a non-negative integer, got {value!r}")
    return value


class _RangeFilter:
    """Inclusive ``min``/``max`` bounds on an integer column; either bound may be omitted."""

    kind: ClassVar[str]
    column: ClassVar[str]

    def __init__(self, params: Mapping[str, Any]) -> None:
        check_params(self.kind, params, ("min", "max"))
        self.low = _optional_count(self.kind, "min", params.get("min"))
        self.high = _optional_count(self.kind, "max", params.get("max"))
        if self.low is not None and self.high is not None and self.low > self.high:
            raise ValueError(f"{self.kind}: min ({self.low}) is greater than max ({self.high})")

    def mask(self, docs: DocColumns) -> npt.NDArray[np.bool_]:
        values = getattr(docs, self.column)
        result = np.ones(len(docs), dtype=np.bool_)
        if self.low is not None:
            result &= values >= self.low
        if self.high is not None:
            result &= values <= self.high
        return result


class BodyTokensFilter(_RangeFilter):
    kind = "body_tokens"
    column = "n_body_tokens"


class TitleContentWordsFilter(_RangeFilter):
    kind = "title_content_words"
    column = "n_title_content"


class _MembershipFilter:
    """Keeps documents whose string column is one of ``in`` (a non-empty list of strings)."""

    kind: ClassVar[str]
    column: ClassVar[str]

    def __init__(self, params: Mapping[str, Any]) -> None:
        check_params(self.kind, params, ("in",), required=("in",))
        values = params["in"]
        if (
            isinstance(values, str)
            or not isinstance(values, tuple | list)
            or not values
            or not all(isinstance(item, str) for item in values)
        ):
            raise ValueError(f"{self.kind}.in must be a non-empty list of strings, got {values!r}")
        self.values = tuple(values)

    def mask(self, docs: DocColumns) -> npt.NDArray[np.bool_]:
        column = getattr(docs, self.column)
        return np.isin(column, np.asarray(self.values, dtype=np.str_))


class DomainFilter(_MembershipFilter):
    kind = "domain"
    column = "domain"


class VariantFilter(_MembershipFilter):
    kind = "variant"
    column = "variant"


class SourceFilter(_MembershipFilter):
    kind = "source"
    column = "source_id"


class TitleInBodyFilter:
    kind = "title_in_body"

    def __init__(self, params: Mapping[str, Any]) -> None:
        check_params(self.kind, params, ("value",), required=("value",))
        value = params["value"]
        if not isinstance(value, bool):
            raise ValueError(f"title_in_body.value must be a boolean, got {value!r}")
        self.value = value

    def mask(self, docs: DocColumns) -> npt.NDArray[np.bool_]:
        result: npt.NDArray[np.bool_] = docs.title_in_body == self.value
        return result


for _cls in (
    BodyTokensFilter,
    TitleContentWordsFilter,
    DomainFilter,
    VariantFilter,
    SourceFilter,
    TitleInBodyFilter,
):
    SCENARIO_FILTERS.register(_cls.kind, _cls)
