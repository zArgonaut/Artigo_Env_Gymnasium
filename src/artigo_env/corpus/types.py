"""Shared contracts of the corpus pipeline (SDD 06 §2).

Everything that crosses a stage boundary is a ``pyarrow.RecordBatch`` with ``DOC_SCHEMA``. Stages
fill columns from left to right; a column is null until the stage that owns it has run.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import pyarrow as pa

from artigo_env.core.config import ConfigBase

# --- documents -------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RawDocument:
    """One document as read from a source, before any cleaning."""

    source_id: str
    doc_id: str  # stable inside the source
    title: str | None
    body: str  # plain text; headers as "== X ==" lines when the source has them
    url: str | None = None
    date: str | None = None  # ISO 8601 when known
    license_id: str = "unknown"
    provenance: Mapping[str, str] = field(default_factory=dict)
    variant: str | None = None  # "pt-BR", "pt-PT" or None


@dataclass(frozen=True, slots=True)
class ShardRef:
    """One unit of resumable work: a raw file (or a slice of one) of a source."""

    source_id: str
    shard_id: str  # stable, file-system safe; used in work paths
    path: Path


DOC_SCHEMA = pa.schema(
    [
        # S1 ingest
        pa.field("source_id", pa.string(), nullable=False),
        pa.field("doc_id", pa.string(), nullable=False),
        pa.field("title", pa.string()),
        pa.field("body", pa.string(), nullable=False),
        pa.field("url", pa.string()),
        pa.field("date", pa.string()),
        pa.field("license_id", pa.string(), nullable=False),
        pa.field("provenance", pa.string(), nullable=False),  # canonical JSON object
        pa.field("variant", pa.string()),
        # S2 origin
        pa.field("url_domain", pa.string()),
        # S4 tokenize
        pa.field("n_body_tokens", pa.int32()),
        pa.field("n_title_tokens", pa.int32()),
        pa.field("n_title_content", pa.int32()),
        # S5 playable
        pa.field("title_in_body", pa.bool_()),
        # S8 annotate
        pa.field("domain", pa.string()),
        pa.field("length_band", pa.string()),
        # S9 partition
        pa.field("split", pa.string()),
    ]
)

Row = dict[str, Any]


def raw_to_row(doc: RawDocument) -> Row:
    """A ``DOC_SCHEMA`` row with only the S1 columns filled."""
    row: Row = dict.fromkeys(DOC_SCHEMA.names)
    row.update(
        source_id=doc.source_id,
        doc_id=doc.doc_id,
        title=doc.title,
        body=doc.body,
        url=doc.url,
        date=doc.date,
        license_id=doc.license_id,
        provenance=json.dumps(dict(doc.provenance), sort_keys=True, ensure_ascii=False),
        variant=doc.variant,
    )
    return row


def to_batch(rows: Iterable[Row]) -> pa.RecordBatch:
    """Rows (dicts keyed by ``DOC_SCHEMA`` names; missing keys become null) as one batch."""
    return pa.RecordBatch.from_pylist(list(rows), schema=DOC_SCHEMA)


def to_rows(batch: pa.RecordBatch) -> list[Row]:
    rows: list[Row] = batch.to_pylist()
    return rows


# --- removal reasons (closed vocabulary; the report groups by stage and reason) ----------------

REASONS: Mapping[str, tuple[str, ...]] = {
    "origin": ("wikimedia_domain", "blocked_domain", "blocked_metadata"),
    "normalize": ("empty_body", "empty_title"),
    "playable": (
        "title_not_informative",
        "title_content_out_of_range",
        "body_too_short",
        "body_too_long",
        "title_not_in_body",
    ),
    "decontam": ("w_title", "w_13gram", "w_near_duplicate"),
    "dedup": ("dup_url", "dup_body", "near_duplicate"),
}


# --- stages ----------------------------------------------------------------------------------


@dataclass(frozen=True)
class PlayabilityConfig(ConfigBase):
    """Playability criteria (SDD 06 §6, DEC-048)."""

    min_title_content: int = 1
    max_title_content: int = 6
    require_title_lemmas_in_body: str = "all"  # "all" | "none"
    min_body_tokens: int = 150
    max_body_tokens: int = 16_384  # L_max ([LOCAL-OBS-1])

    def __post_init__(self) -> None:
        if self.require_title_lemmas_in_body not in ("all", "none"):
            raise ValueError("require_title_lemmas_in_body must be 'all' or 'none'")
        if not 1 <= self.min_title_content <= self.max_title_content:
            raise ValueError("title content range must satisfy 1 <= min <= max")
        if not 1 <= self.min_body_tokens <= self.max_body_tokens:
            raise ValueError("body token range must satisfy 1 <= min <= max")


@dataclass(frozen=True)
class StageResult:
    """Output of one stage over one batch: surviving rows and removal counts by reason."""

    batch: pa.RecordBatch
    removed: Mapping[str, int] = field(default_factory=dict)


class StageContext(Protocol):
    """What a stage may read. Concrete contexts are built by the pipeline (and by tests)."""

    @property
    def tokenizer_hyphen(self) -> str: ...

    @property
    def playability(self) -> PlayabilityConfig: ...

    def is_stopword(self, form: str) -> bool:
        """Membership in the stopword set used for playability (``game227``)."""
        ...

    def lemma(self, form: str) -> str:
        """Lemma of a lowercase form under the playability lemmatizer (``open-tables``)."""
        ...


class PipelineStage(Protocol):
    name: str
    version: str

    def run(self, batch: pa.RecordBatch, ctx: StageContext) -> StageResult: ...


class DocumentSource(Protocol):
    """A registered kind of raw data (REQ-COR-001)."""

    source_id: str

    def iter_shards(self, raw_root: Path) -> Iterable[ShardRef]:
        """Shards in a deterministic order (sorted by ``shard_id``)."""
        ...

    def read_shard(self, shard: ShardRef) -> Iterator[RawDocument]:
        """Documents of one shard in file order, with ``doc_id`` stable across runs."""
        ...
