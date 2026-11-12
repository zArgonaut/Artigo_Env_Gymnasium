"""S9 partition: stable train/ext-val split by hash (SDD 06 §9, DEC-051; REQ-COR-013)."""

from __future__ import annotations

import pyarrow as pa
import xxhash

from artigo_env.corpus.stages.context import check_batch, finish
from artigo_env.corpus.types import StageContext, StageResult

__all__ = ["EXT_VAL", "TRAIN", "TRAIN_PER_MILLE", "PartitionStage", "split_of"]

TRAIN = "train"
EXT_VAL = "ext-val"
TRAIN_PER_MILLE = 900


def split_of(source_id: str, doc_id: str) -> str:
    """``train`` if ``xxh64(source_id \\x1f doc_id) % 1000 < 900`` (seed 0), else ``ext-val``."""
    key = f"{source_id}\x1f{doc_id}".encode()
    return TRAIN if xxhash.xxh64_intdigest(key) % 1000 < TRAIN_PER_MILLE else EXT_VAL


class PartitionStage:
    """Fills ``split``; depends only on the document's own ids, so it is stable across builds."""

    name = "partition"
    version = "1"

    def run(self, batch: pa.RecordBatch, ctx: StageContext) -> StageResult:
        check_batch(batch, self.name)
        splits = [
            split_of(source_id, doc_id)
            for source_id, doc_id in zip(
                batch.column("source_id").to_pylist(),
                batch.column("doc_id").to_pylist(),
                strict=True,
            )
        ]
        return finish(self.name, batch, [None] * batch.num_rows, {"split": splits})
