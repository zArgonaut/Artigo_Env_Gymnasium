"""Readers of raw data, one per source kind (SDD 06 §2).

Importing this package registers the built-in kinds (``hf_parquet``, ``jsonl_gz``,
``wikipedia_extracts``, ``demo``) in ``SOURCE_KINDS``.
"""

from __future__ import annotations

from artigo_env.corpus.sources import demo, extracts, hf_parquet, jsonl_gz
from artigo_env.corpus.sources.registry import SOURCE_KINDS, SourceReadError, open_source

__all__ = [
    "SOURCE_KINDS",
    "SourceReadError",
    "demo",
    "extracts",
    "hf_parquet",
    "jsonl_gz",
    "open_source",
]
