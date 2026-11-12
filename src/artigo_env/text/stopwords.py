"""Stopword sets (SDD 05 §4, REQ-TXT-010, REQ-TXT-011).

Comparison is done on the lowercase token, before lemmatization.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from importlib.resources.abc import Traversable
from pathlib import Path

from artigo_env.core.config import ConfigBase, sha256_hex
from artigo_env.core.errors import ArtigoEnvError
from artigo_env.core.resources import resource_path
from artigo_env.text.normalize import lower

STOPWORD_KINDS = ("game227", "closed-class-all", "closed-class-any", "none", "custom")


@dataclass(frozen=True)
class StopwordSpec(ConfigBase):
    kind: str = "game227"
    path: str | None = None


@dataclass(frozen=True)
class StopwordSet:
    words: frozenset[str]

    def __contains__(self, word: object) -> bool:
        return word in self.words

    def __len__(self) -> int:
        return len(self.words)

    def content_hash(self) -> str:
        """SHA-256 of the sorted words, one per line."""
        return sha256_hex("\n".join(sorted(self.words)))


def parse_stopwords(lines: Iterable[str]) -> frozenset[str]:
    """Words one per line; blank lines and lines starting with ``#`` are ignored."""
    words: set[str] = set()
    for line in lines:
        word = line.strip()
        if word and not word.startswith("#"):
            words.add(lower(word))
    return frozenset(words)


def _read(node: Path | Traversable, label: str) -> StopwordSet:
    if not node.is_file():
        raise ArtigoEnvError(f"stopword list not found: {label}")
    return StopwordSet(parse_stopwords(node.read_text(encoding="utf-8").splitlines()))


def load_stopwords(spec: StopwordSpec) -> StopwordSet:
    """Build the stopword set described by ``spec``."""
    kind = spec.kind
    if kind not in STOPWORD_KINDS:
        raise ValueError(f"unknown stopword kind {kind!r}; available: {', '.join(STOPWORD_KINDS)}")
    if kind == "none":
        return StopwordSet(frozenset())
    if kind == "custom":
        if not spec.path:
            raise ValueError("stopword kind 'custom' requires a path")
        return _read(Path(spec.path), spec.path)
    filename = {
        "game227": "game227.txt",
        "closed-class-all": "closed-class-all.txt",
        "closed-class-any": "closed-class-any.txt",
    }[kind]
    return _read(resource_path("stopwords", filename), f"assets/stopwords/{filename}")
