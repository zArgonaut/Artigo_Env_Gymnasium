"""Demonstration corpus bundled with the package (``assets/demo/articles.jsonl``; SDD 02 §B.8)."""

from __future__ import annotations

import json
from collections.abc import Iterator
from importlib.resources.abc import Traversable
from pathlib import Path

from artigo_env.core.resources import resource_path
from artigo_env.corpus.manifest import SourceManifestEntry
from artigo_env.corpus.sources.registry import SOURCE_KINDS, SourceReadError
from artigo_env.corpus.types import RawDocument, ShardRef

SHARD_ID = "demo"


def default_articles() -> Traversable:
    return resource_path("demo", "articles.jsonl")


class DemoSource:
    """A single shard ``"demo"`` read from the package, independent of ``ARTIGO_DATA_ROOT``.

    Each line is an object with ``title``, ``body`` and optionally ``license_id`` (default: the
    manifest's) and ``id`` (default: the 0-based line number among non-blank lines).
    """

    def __init__(self, entry: SourceManifestEntry, articles: Traversable | Path | None = None):
        self.entry = entry
        self.source_id = entry.id
        self.articles = default_articles() if articles is None else articles

    def iter_shards(self, raw_root: Path) -> list[ShardRef]:
        """The bundled file; ``raw_root`` is ignored."""
        node = self.articles
        if not node.is_file():
            raise SourceReadError(
                f"demo articles not found at {node}: the package was built without "
                "assets/demo/articles.jsonl"
            )
        if not isinstance(node, Path):
            raise SourceReadError(f"demo articles at {node} are not on the file system")
        return [ShardRef(self.entry.id, SHARD_ID, node)]

    def read_shard(self, shard: ShardRef) -> Iterator[RawDocument]:
        if not shard.path.is_file():
            raise SourceReadError(f"demo articles not found at {shard.path}")
        row = 0
        with shard.path.open(encoding="utf-8") as lines:
            for line_no, line in enumerate(lines, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise SourceReadError(
                        f"{shard.path}:{line_no}: invalid JSON: {error}"
                    ) from None
                if not isinstance(record, dict) or not isinstance(record.get("body"), str):
                    raise SourceReadError(f"{shard.path}:{line_no}: expected an object with body")
                title = record.get("title")
                yield RawDocument(
                    source_id=self.entry.id,
                    doc_id=str(record.get("doc_id", record.get("id", row))),
                    title=None if title is None else str(title),
                    body=record["body"],
                    license_id=str(record.get("license_id") or self.entry.license_id),
                    provenance={},
                    variant=self.entry.variant,
                )
                row += 1


SOURCE_KINDS.register("demo", DemoSource)
