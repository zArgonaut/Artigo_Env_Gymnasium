"""Build context for the per-document stages, and the batch helpers they share.

``BuildContext`` implements ``StageContext``: it carries the playability configuration, the
tokenizer hyphen mode, the stopword set and the playability lemmatizer. Stopwords and the
lemmatizer are loaded lazily, so a context can be built (and pickled to worker processes)
before the lemmatizer data is needed.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pyarrow as pa

from artigo_env.core.config import canonical_json, sha256_hex
from artigo_env.core.errors import ArtigoEnvError
from artigo_env.corpus.types import DOC_SCHEMA, REASONS, PlayabilityConfig, StageResult
from artigo_env.text.lemmatizers.base import Lemmatizer
from artigo_env.text.lemmatizers.registry import get_lemmatizer
from artigo_env.text.normalize import lower
from artigo_env.text.stopwords import StopwordSet, StopwordSpec, load_stopwords

__all__ = ["BuildContext", "check_batch", "finish", "rebuild"]

_HYPHEN_MODES = ("split", "join")
_LEMMA_CACHE_LIMIT = 500_000


class BuildContext:
    """Concrete ``StageContext`` used by the pipeline and by tests."""

    def __init__(
        self,
        playability: PlayabilityConfig,
        tokenizer_hyphen: str = "split",
        stopwords: StopwordSpec | None = None,
        lemmatizer: str = "open-tables",
    ) -> None:
        if tokenizer_hyphen not in _HYPHEN_MODES:
            raise ValueError(
                f"unknown hyphen mode: {tokenizer_hyphen!r} (expected 'split' or 'join')"
            )
        self._playability = playability
        self._hyphen = tokenizer_hyphen
        self.stopword_spec = stopwords if stopwords is not None else StopwordSpec()
        self.lemmatizer_name = lemmatizer
        self._stopwords: StopwordSet | None = None
        self._lemmatizer: Lemmatizer | None = None
        self._lemma_cache: dict[str, str] = {}

    # --- StageContext ----------------------------------------------------------------------

    @property
    def tokenizer_hyphen(self) -> str:
        return self._hyphen

    @property
    def playability(self) -> PlayabilityConfig:
        return self._playability

    def is_stopword(self, form: str) -> bool:
        """Whether the lowercase ``form`` belongs to the stopword set."""
        return lower(form) in self.stopwords

    def lemma(self, form: str) -> str:
        """Lemma of ``form`` under the playability lemmatizer (cached per context)."""
        cached = self._lemma_cache.get(form)
        if cached is None:
            if len(self._lemma_cache) >= _LEMMA_CACHE_LIMIT:
                self._lemma_cache.clear()
            cached = self._lemma_cache[form] = self.lemmatizer.lemma(form)
        return cached

    # --- components ------------------------------------------------------------------------

    @property
    def stopwords(self) -> StopwordSet:
        if self._stopwords is None:
            self._stopwords = load_stopwords(self.stopword_spec)
        return self._stopwords

    @property
    def lemmatizer(self) -> Lemmatizer:
        if self._lemmatizer is None:
            self._lemmatizer = get_lemmatizer(self.lemmatizer_name)
        return self._lemmatizer

    def content_hash(self) -> str:
        """SHA-256 of everything that can change a stage's output under this context."""
        return sha256_hex(
            canonical_json(
                {
                    "playability": self._playability.to_dict(),
                    "tokenizer_hyphen": self._hyphen,
                    "stopwords": {
                        "spec": self.stopword_spec.to_dict(),
                        "hash": self.stopwords.content_hash(),
                    },
                    "lemmatizer": {
                        "name": self.lemmatizer_name,
                        "hash": self.lemmatizer.content_hash(),
                    },
                }
            )
        )

    # --- pickling (workers rebuild the loaded components) -----------------------------------

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_stopwords"] = None
        state["_lemmatizer"] = None
        state["_lemma_cache"] = {}
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)


# --- batch helpers -----------------------------------------------------------------------------


def check_batch(batch: pa.RecordBatch, stage: str) -> None:
    """Fail clearly when ``batch`` does not carry ``DOC_SCHEMA``."""
    if not batch.schema.equals(DOC_SCHEMA):
        raise ArtigoEnvError(f"stage {stage!r} expects a batch with DOC_SCHEMA")


def rebuild(
    batch: pa.RecordBatch,
    keep: Sequence[bool] | None = None,
    columns: Mapping[str, Sequence[Any]] | None = None,
) -> pa.RecordBatch:
    """Replace ``columns`` (full-length value lists) and keep the rows where ``keep`` is true.

    Row order is preserved; columns not named are passed through untouched.
    """
    columns = columns or {}
    unknown = sorted(set(columns) - set(DOC_SCHEMA.names))
    if unknown:
        raise ArtigoEnvError(f"unknown columns: {unknown}")
    arrays = [
        pa.array(columns[field.name], type=field.type) if field.name in columns else batch.column(i)
        for i, field in enumerate(DOC_SCHEMA)
    ]
    out = pa.RecordBatch.from_arrays(arrays, schema=DOC_SCHEMA)
    if keep is not None and not all(keep):
        out = out.filter(pa.array(keep, type=pa.bool_()))
    return out


def finish(
    stage: str,
    batch: pa.RecordBatch,
    reasons: Sequence[str | None],
    columns: Mapping[str, Sequence[Any]] | None = None,
) -> StageResult:
    """``StageResult`` from a per-row removal reason (``None`` = kept).

    ``removed`` lists every reason of the stage in ``REASONS`` order, zeros included.
    """
    allowed = REASONS.get(stage, ())
    counts = dict.fromkeys(allowed, 0)
    for reason in reasons:
        if reason is None:
            continue
        if reason not in counts:
            raise ArtigoEnvError(f"stage {stage!r} cannot remove rows for reason {reason!r}")
        counts[reason] += 1
    keep = [reason is None for reason in reasons]
    return StageResult(rebuild(batch, keep, columns), counts)
