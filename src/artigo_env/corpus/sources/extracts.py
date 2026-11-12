"""Snapshot of the TextExtracts API answers for the evaluation list (set W; plan decision 5).

The snapshot is written by ``tools/w/fetch_extracts.py`` under ``raw/<raw_dir>/<subdir>/``:
``index.jsonl`` holds one row per requested title (``requested``, ``resolved``, ``pageid``,
``lastrevid``, ``status``, ``file``) and ``pages/<file>`` holds ``{"requested", "fetched_at",
"response"}`` with the text at ``response["query"]["pages"][0]["extract"]``.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from artigo_env.corpus.manifest import SourceManifestEntry
from artigo_env.corpus.sources.registry import SOURCE_KINDS, SourceReadError
from artigo_env.corpus.types import RawDocument, ShardRef

SHARD_ID = "w"
INDEX = "index.jsonl"
PAGES = "pages"


def read_index(index_path: Path) -> list[dict[str, Any]]:
    """Rows of ``index.jsonl``; an unterminated last line (a write in progress) is ignored."""
    text = index_path.read_text(encoding="utf-8")
    lines = text.split("\n")
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            if number == len(lines):  # last line without "\n": still being written
                break
            raise SourceReadError(f"{index_path}:{number}: invalid JSON: {error}") from None
        if not isinstance(row, dict):
            raise SourceReadError(f"{index_path}:{number}: expected a JSON object")
        rows.append(row)
    return rows


def select_pages(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One ``ok`` row per page, ordered by ``pageid``.

    Several requested titles can redirect to the same page. The row whose requested title equals
    the resolved one wins, then the smallest requested title, so the choice does not depend on
    the order in which titles were fetched.
    """
    best: dict[int, dict[str, Any]] = {}
    for row in rows:
        if row.get("status") != "ok":
            continue
        pageid = row.get("pageid")
        if not isinstance(pageid, int) or isinstance(pageid, bool):
            raise SourceReadError(f"index row for {row.get('requested')!r} has no integer pageid")
        current = best.get(pageid)
        if current is None or _rank(row) < _rank(current):
            best[pageid] = row
    return [best[pageid] for pageid in sorted(best)]


def _rank(row: dict[str, Any]) -> tuple[bool, str]:
    requested = str(row.get("requested"))
    return (requested != row.get("resolved"), requested)


class ExtractsSource:
    """A single shard ``"w"``; ``doc_id`` = page id, ``title`` = resolved title.

    ``provenance`` keeps the requested title and the ``lastrevid`` of the snapshot. ``url`` is
    left empty on purpose: a Wikipedia URL would be removed by the origin stage (S2).
    """

    def __init__(self, entry: SourceManifestEntry) -> None:
        if "subdir" not in entry.options:
            raise SourceReadError(
                f"source {entry.id!r} needs a 'subdir' option naming the snapshot directory"
            )
        self.entry = entry
        self.source_id = entry.id

    def snapshot_dir(self, raw_root: Path) -> Path:
        return self.entry.source_dir(raw_root) / self.entry.options["subdir"]

    def iter_shards(self, raw_root: Path) -> list[ShardRef]:
        directory = self.snapshot_dir(raw_root)
        if not (directory / INDEX).is_file():
            raise SourceReadError(
                f"no extracts snapshot for {self.entry.id!r}: {directory / INDEX} does not exist "
                "(it is written by tools/w/fetch_extracts.py)"
            )
        return [ShardRef(self.entry.id, SHARD_ID, directory)]

    def read_shard(self, shard: ShardRef) -> Iterator[RawDocument]:
        rows = select_pages(read_index(shard.path / INDEX))
        for row in rows:
            page_path = shard.path / PAGES / str(row["file"])
            try:
                payload = json.loads(page_path.read_text(encoding="utf-8"))
                page = payload["response"]["query"]["pages"][0]
            except (OSError, ValueError, KeyError, IndexError, TypeError) as error:
                raise SourceReadError(f"cannot read page file {page_path}: {error!r}") from None
            extract = page.get("extract")
            if not isinstance(extract, str):
                raise SourceReadError(f"page file {page_path} has no text extract")
            yield RawDocument(
                source_id=self.entry.id,
                doc_id=str(row["pageid"]),
                title=row.get("resolved") or page.get("title"),
                body=extract,
                url=None,
                date=None,
                license_id=self.entry.license_id,
                provenance={
                    "requested": str(row["requested"]),
                    "lastrevid": str(row.get("lastrevid")),
                },
                variant=self.entry.variant,
            )


SOURCE_KINDS.register("wikipedia_extracts", ExtractsSource)
