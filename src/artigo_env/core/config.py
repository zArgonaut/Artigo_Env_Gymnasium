"""Frozen, hashable configuration base (REQ-REPO-002).

Package configuration is plain frozen dataclasses. ``content_hash`` is the SHA-256 of a canonical
JSON rendering (sorted keys, no whitespace), so equal content yields the same hash in any process.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import types
from collections.abc import Mapping
from enum import Enum
from typing import Any, Self, Union, get_args, get_origin, get_type_hints


def _normalize(obj: object) -> Any:
    if isinstance(obj, Enum):
        return _normalize(obj.value)
    if obj is None or isinstance(obj, bool | str | int):
        return obj
    if isinstance(obj, float):
        if not math.isfinite(obj):
            raise ValueError("NaN and infinite floats cannot be hashed canonically")
        return obj
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: _normalize(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, Mapping):
        out: dict[str, Any] = {}
        for key, value in obj.items():
            if not isinstance(key, str):
                raise TypeError(f"mapping keys must be str, got {type(key).__name__}")
            out[key] = _normalize(value)
        return out
    if isinstance(obj, list | tuple):
        return [_normalize(item) for item in obj]
    if isinstance(obj, set | frozenset):
        return sorted((_normalize(item) for item in obj), key=_dumps)
    raise TypeError(f"unsupported type in configuration: {type(obj).__name__}")


def _dumps(normalized: Any) -> str:
    return json.dumps(
        normalized, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )


def canonical_json(obj: object) -> str:
    """Render ``obj`` as canonical JSON: sorted keys, no whitespace, tuples as lists."""
    return _dumps(_normalize(obj))


def sha256_hex(text: str) -> str:
    """SHA-256 of the UTF-8 encoding of ``text``, as 64 lowercase hex characters."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _coerce(hint: Any, value: Any) -> Any:
    origin = get_origin(hint)
    if origin is Union or origin is types.UnionType:
        if value is None:
            return None
        options = [arg for arg in get_args(hint) if arg is not type(None)]
        return _coerce(options[0], value) if len(options) == 1 else value
    if isinstance(hint, type) and issubclass(hint, ConfigBase) and isinstance(value, Mapping):
        return hint.from_dict(value)
    if origin is tuple and isinstance(value, list | tuple):
        args = get_args(hint)
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(_coerce(args[0], item) for item in value)
        if len(args) == len(value):
            return tuple(_coerce(arg, item) for arg, item in zip(args, value, strict=True))
        return tuple(value)
    if origin is frozenset and isinstance(value, list | tuple | set | frozenset):
        (arg,) = get_args(hint) or (Any,)
        return frozenset(_coerce(arg, item) for item in value)
    return value


class ConfigBase:
    """Mixin for ``@dataclass(frozen=True)`` configuration classes."""

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible dict (nested configs as dicts, tuples as lists)."""
        normalized = _normalize(self)
        assert isinstance(normalized, dict)
        return normalized

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Self:
        """Rebuild an instance from ``to_dict`` output; key order is irrelevant."""
        if not dataclasses.is_dataclass(cls):
            raise TypeError(f"{cls.__name__} must be a dataclass")
        hints = get_type_hints(cls)
        fields = {f.name: f for f in dataclasses.fields(cls) if f.init}
        unknown = sorted(set(data) - set(fields))
        if unknown:
            raise ValueError(f"unknown keys for {cls.__name__}: {unknown}")
        kwargs: dict[str, Any] = {}
        for name, field in fields.items():
            if name in data:
                kwargs[name] = _coerce(hints.get(name, Any), data[name])
            elif (
                field.default is dataclasses.MISSING
                and field.default_factory is dataclasses.MISSING
            ):
                raise ValueError(f"missing key for {cls.__name__}: {name!r}")
        return cls(**kwargs)

    def content_hash(self) -> str:
        """Stable SHA-256 identity of the configuration content."""
        return sha256_hex(canonical_json(self.to_dict()))
