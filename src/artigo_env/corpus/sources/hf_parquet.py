"""Generic Parquet source (Hugging Face datasets such as CC-News pt and RecognaSumm)."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pyarrow.parquet as pq

from artigo_env.corpus.manifest import SourceManifestEntry
from artigo_env.corpus.sources.registry import SOURCE_KINDS, ColumnMap, find_shards
from artigo_env.corpus.types import RawDocument, ShardRef

BATCH_ROWS = 2_000
SUFFIX = ".parquet"


class HfParquetSource:
    """One shard per ``.parquet`` file; documents in row order, only mapped columns are read.

    ``doc_id`` is the value of the ``id`` column when one is mapped, else ``"<shard_id>:<row>"``.
    """

    def __init__(self, entry: SourceManifestEntry) -> None:
        self.entry = entry
        self.source_id = entry.id
        self.columns = ColumnMap.from_entry(entry)

    def iter_shards(self, raw_root: Path) -> list[ShardRef]:
        return find_shards(self.entry, raw_root, SUFFIX)

    def read_shard(self, shard: ShardRef) -> Iterator[RawDocument]:
        parquet = pq.ParquetFile(shard.path)
        self.columns.check(parquet.schema_arrow.names, str(shard.path))
        names = self.columns.columns()
        provenance = {"shard": shard.shard_id}
        row = 0
        for batch in parquet.iter_batches(batch_size=BATCH_ROWS, columns=names):
            records = {name: batch.column(name).to_pylist() for name in names}
            yield from self.columns.documents(self.entry, shard, row, records, provenance)
            row += batch.num_rows


SOURCE_KINDS.register("hf_parquet", HfParquetSource)
