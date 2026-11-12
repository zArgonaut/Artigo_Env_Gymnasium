"""Generic gzip-compressed JSON Lines source (for example the yearly CloverSearch files)."""

from __future__ import annotations

import gzip
import json
from collections.abc import Iterator
from pathlib import Path

from artigo_env.corpus.manifest import SourceManifestEntry
from artigo_env.corpus.sources.registry import (
    SOURCE_KINDS,
    ColumnMap,
    SourceReadError,
    find_shards,
)
from artigo_env.corpus.types import RawDocument, ShardRef

SUFFIX = ".jsonl.gz"


class JsonlGzSource:
    """One shard per ``.jsonl.gz`` file, streamed line by line.

    Blank lines are skipped; ``row`` counts the remaining records from 0, so ``doc_id`` is the
    ``id`` field when one is mapped, else ``"<shard_id>:<row>"``. A mapped field that is absent
    from a record reads as null.
    """

    def __init__(self, entry: SourceManifestEntry) -> None:
        self.entry = entry
        self.source_id = entry.id
        self.columns = ColumnMap.from_entry(entry)

    def iter_shards(self, raw_root: Path) -> list[ShardRef]:
        return find_shards(self.entry, raw_root, SUFFIX)

    def read_shard(self, shard: ShardRef) -> Iterator[RawDocument]:
        names = self.columns.columns()
        provenance = {"shard": shard.shard_id}
        row = 0
        with gzip.open(shard.path, "rt", encoding="utf-8") as lines:
            for line_no, line in enumerate(lines, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise SourceReadError(
                        f"{shard.path}:{line_no}: invalid JSON: {error}"
                    ) from None
                if not isinstance(record, dict):
                    raise SourceReadError(f"{shard.path}:{line_no}: expected a JSON object")
                if self.columns.body not in record:
                    raise SourceReadError(
                        f"{shard.path}:{line_no}: body field {self.columns.body!r} not found"
                    )
                columns = {name: [record.get(name)] for name in names}
                yield self.columns.document(self.entry, shard, row, columns, 0, provenance)
                row += 1


SOURCE_KINDS.register("jsonl_gz", JsonlGzSource)
