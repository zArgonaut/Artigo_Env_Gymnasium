"""Lemmatizer protocol and the rule-based implementation (SDD 05 §5, REQ-TXT-014/016)."""

from __future__ import annotations

from functools import lru_cache
from typing import Protocol

from artigo_env.core.config import sha256_hex
from artigo_env.text.lemmatizers.identity import IdentityLemmatizer
from artigo_env.text.lemmatizers.tables import LemmaTables

ALGORITHM_VERSION = 1

__all__ = ["ALGORITHM_VERSION", "IdentityLemmatizer", "Lemmatizer", "RuleLemmatizer"]


class Lemmatizer(Protocol):
    name: str

    def lemma(self, form: str) -> str: ...

    def content_hash(self) -> str: ...


class RuleLemmatizer:
    """Verb rules (irregular, then regular) win; otherwise plural and gender are chained."""

    def __init__(self, name: str, tables: LemmaTables, cache_size: int = 200_000) -> None:
        self.name = name
        self.tables = tables
        self.lemma = lru_cache(maxsize=cache_size)(self._lemma)

    def _lemma(self, form: str) -> str:
        word = form.lower()
        verb = self.tables.irregular.apply(word)
        if verb is None:
            verb = self.tables.regular.apply(word)
        if verb is not None:
            return verb
        plural = self.tables.plural.apply(word)
        singular = word if plural is None else plural
        gendered = self.tables.gender.apply(singular)
        return singular if gendered is None else gendered

    def content_hash(self) -> str:
        return sha256_hex(f"{self.tables.content_hash()}:{ALGORITHM_VERSION}")
