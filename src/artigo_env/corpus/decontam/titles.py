"""Title barrier: a training document whose title equals a title of W is removed (SDD 06 §7.1)."""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator
from pathlib import Path

from artigo_env.text.normalize import lower, normalize

_SPACES = re.compile(r"\s+")


def normalize_title(title: str) -> str:
    """Comparison key of a title: NFC, ``_`` as space, whitespace collapsed, lowercase.

    Lowercasing the whole title also makes the first letter indifferent, as MediaWiki does
    ("al-Khwarizmi" and "Al-Khwarizmi" are the same page; REQ-COR-007).
    """
    text = normalize(title).replace("_", " ")
    # NFC again after lowercasing keeps the key idempotent: the key of a key is itself.
    return normalize(lower(_SPACES.sub(" ", text).strip()))


class TitleBarrier:
    """Set of normalized W titles."""

    __slots__ = ("_keys",)

    def __init__(self, titles: Iterable[str] = ()) -> None:
        keys = {normalize_title(title) for title in titles}
        keys.discard("")
        self._keys = frozenset(keys)

    def __contains__(self, title: object) -> bool:
        return isinstance(title, str) and normalize_title(title) in self._keys

    def __len__(self) -> int:
        return len(self._keys)

    def __iter__(self) -> Iterator[str]:
        """Normalized titles in sorted order."""
        return iter(sorted(self._keys))

    def to_text(self) -> str:
        """One normalized title per line, sorted (the saved form)."""
        return "".join(f"{key}\n" for key in self)

    def save(self, path: Path) -> None:
        path.write_bytes(self.to_text().encode("utf-8"))

    @classmethod
    def load(cls, path: Path) -> TitleBarrier:
        return cls(path.read_bytes().decode("utf-8").split("\n"))
