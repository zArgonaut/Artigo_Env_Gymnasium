"""Registry of source kinds and helpers shared by the tabular readers (REQ-COR-001)."""

from __future__ import annotations

import datetime as dt
import re
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from artigo_env.core.errors import ArtigoEnvError
from artigo_env.core.registry import Registry
from artigo_env.corpus.manifest import SourceManifestEntry
from artigo_env.corpus.types import DocumentSource, RawDocument, ShardRef

SourceFactory = Callable[[SourceManifestEntry], DocumentSource]

SOURCE_KINDS: Registry[SourceFactory] = Registry("source kind")


class SourceReadError(ArtigoEnvError):
    """Raw data of a source is missing or does not match the manifest's column mapping."""


def open_source(entry: SourceManifestEntry) -> DocumentSource:
    """The ``DocumentSource`` that reads ``entry`` (built-in kinds are registered on import)."""
    return SOURCE_KINDS.get(entry.kind)(entry)


# --- helpers for file-per-shard sources --------------------------------------------------------

_UNSAFE = re.compile(r"[^A-Za-z0-9._-]")


def shard_id_for(relative: str, suffix: str) -> str:
    """File-system safe shard id from a POSIX path relative to the source dir.

    ``cc_news_pt/train/0003.parquet`` -> ``cc_news_pt__train__0003``.
    """
    stem = relative[: -len(suffix)] if relative.endswith(suffix) else relative
    return "__".join(_UNSAFE.sub("_", part) for part in stem.split("/"))


def find_shards(entry: SourceManifestEntry, raw_root: Path, suffix: str) -> list[ShardRef]:
    """One shard per ``*<suffix>`` file under the source dir (or its ``subdir``), by shard id.

    Shard ids are relative to the source dir, so the ``subdir`` option filters files without
    changing ids. Hidden directories (such as the ``.cache`` left by ``huggingface_hub``) are
    skipped.
    """
    source_dir = entry.source_dir(raw_root)
    scan_dir = source_dir / entry.options["subdir"] if "subdir" in entry.options else source_dir
    if not scan_dir.is_dir():
        raise SourceReadError(
            f"no raw data for source {entry.id!r}: {scan_dir} does not exist "
            f"(run: artigo-env data fetch {entry.id})"
        )
    shards: dict[str, ShardRef] = {}
    for path in scan_dir.rglob(f"*{suffix}"):
        relative = path.relative_to(source_dir).as_posix()
        if not path.is_file() or any(part.startswith(".") for part in relative.split("/")):
            continue
        shard_id = shard_id_for(relative, suffix)
        if shard_id in shards:
            raise SourceReadError(f"two files of {entry.id!r} map to shard id {shard_id!r}")
        shards[shard_id] = ShardRef(entry.id, shard_id, path)
    if not shards:
        raise SourceReadError(f"no {suffix} files for source {entry.id!r} under {scan_dir}")
    return [shards[key] for key in sorted(shards)]


def as_text(value: Any) -> str | None:
    """A raw cell as text: ``None`` stays ``None``; dates and datetimes become ISO 8601."""
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, dt.date | dt.datetime):
        return value.isoformat()
    return str(value)


@dataclass(frozen=True)
class ColumnMap:
    """Raw columns mapped to ``RawDocument`` fields (the manifest's column options)."""

    body: str
    title: str | None = None
    url: str | None = None
    date: str | None = None
    id: str | None = None

    @classmethod
    def from_entry(cls, entry: SourceManifestEntry) -> ColumnMap:
        body = entry.column("body")
        if body is None:
            raise SourceReadError(f"source {entry.id!r} needs a 'body' column option")
        return cls(
            body=body,
            title=entry.column("title"),
            url=entry.column("url"),
            date=entry.column("date"),
            id=entry.column("id"),
        )

    def columns(self) -> list[str]:
        """Mapped raw column names, unique, in a fixed order."""
        names = (self.id, self.title, self.body, self.url, self.date)
        out: list[str] = []
        for name in names:
            if name is not None and name not in out:
                out.append(name)
        return out

    def check(self, available: Sequence[str], where: str) -> None:
        missing = [name for name in self.columns() if name not in available]
        if missing:
            raise SourceReadError(
                f"{where}: mapped columns {missing} not found; available: {list(available)}"
            )

    def documents(
        self,
        entry: SourceManifestEntry,
        shard: ShardRef,
        first_row: int,
        records: Mapping[str, Sequence[Any]],
        provenance: Mapping[str, str],
    ) -> Iterator[RawDocument]:
        """Documents for a block of rows given column-wise, numbered from ``first_row``."""
        n_rows = len(records[self.body])
        for offset in range(n_rows):
            row = first_row + offset
            yield self.document(entry, shard, row, records, offset, provenance)

    def document(
        self,
        entry: SourceManifestEntry,
        shard: ShardRef,
        row: int,
        records: Mapping[str, Sequence[Any]],
        offset: int,
        provenance: Mapping[str, str],
    ) -> RawDocument:
        def cell(name: str | None) -> str | None:
            return None if name is None else as_text(records[name][offset])

        if self.id is None:
            doc_id = f"{shard.shard_id}:{row}"
        else:
            raw_id = cell(self.id)
            if raw_id is None or not raw_id.strip():
                raise SourceReadError(f"{shard.path}: row {row} has an empty id column {self.id!r}")
            doc_id = raw_id
        return RawDocument(
            source_id=entry.id,
            doc_id=doc_id,
            title=cell(self.title),
            body=cell(self.body) or "",
            url=cell(self.url),
            date=cell(self.date),
            license_id=entry.license_id,
            provenance=provenance,
            variant=entry.variant,
        )
