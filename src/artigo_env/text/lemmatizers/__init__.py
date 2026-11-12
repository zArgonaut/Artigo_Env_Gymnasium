"""Lemmatizers: a rule interpreter, table files and the registry of named lemmatizers."""

from __future__ import annotations

from artigo_env.text.lemmatizers.base import (
    ALGORITHM_VERSION,
    IdentityLemmatizer,
    Lemmatizer,
    RuleLemmatizer,
)
from artigo_env.text.lemmatizers.registry import LEMMATIZERS, get_lemmatizer

__all__ = [
    "ALGORITHM_VERSION",
    "LEMMATIZERS",
    "IdentityLemmatizer",
    "Lemmatizer",
    "RuleLemmatizer",
    "get_lemmatizer",
]
