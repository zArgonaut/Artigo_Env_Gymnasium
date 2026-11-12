"""Named component registries, the mechanism behind every extension point (SDD 01 §3.1)."""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Generic, TypeVar, overload

from artigo_env.core.errors import DuplicateNameError, UnknownNameError

T = TypeVar("T")

_NAME = re.compile(r"^[a-z0-9][a-z0-9_.:@-]*$")


class Registry(Generic[T]):
    """Maps stable lowercase names to components of one kind."""

    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._items: dict[str, T] = {}

    @overload
    def register(self, name: str) -> Callable[[T], T]: ...
    @overload
    def register(self, name: str, obj: T) -> T: ...

    def register(self, name: str, obj: T | None = None) -> T | Callable[[T], T]:
        """Register ``obj`` under ``name``; without ``obj``, act as a decorator."""
        if not _NAME.match(name):
            raise ValueError(f"invalid {self.kind} name: {name!r}")

        def add(item: T) -> T:
            if name in self._items:
                raise DuplicateNameError(f"{self.kind} {name!r} is already registered")
            self._items[name] = item
            return item

        return add if obj is None else add(obj)

    def get(self, name: str) -> T:
        try:
            return self._items[name]
        except KeyError:
            available = ", ".join(self.names()) or "<none>"
            raise UnknownNameError(
                f"unknown {self.kind} {name!r}; available: {available}"
            ) from None

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._items))

    def __contains__(self, name: object) -> bool:
        return name in self._items

    def __len__(self) -> int:
        return len(self._items)
