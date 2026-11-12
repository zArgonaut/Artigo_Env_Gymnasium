"""Lemmatizer table files (JSON, ``format_version`` 1)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib.resources.abc import Traversable
from pathlib import Path

from artigo_env.core.config import canonical_json, sha256_hex
from artigo_env.text.lemmatizers.rules import RuleNode, node_from_dict

FORMAT_VERSION = 1
STAGES = ("irregular", "regular", "plural", "gender")


@dataclass(frozen=True)
class LemmaTables:
    """The four rule stages of a rule-based lemmatizer."""

    name: str
    source: str
    irregular: RuleNode
    regular: RuleNode
    plural: RuleNode
    gender: RuleNode

    def to_dict(self) -> dict[str, object]:
        return {
            "format_version": FORMAT_VERSION,
            "name": self.name,
            "source": self.source,
            "stages": {stage: getattr(self, stage).to_dict() for stage in STAGES},
        }

    def content_hash(self) -> str:
        """SHA-256 of the canonical JSON of the whole table (any rule change changes it)."""
        return sha256_hex(canonical_json(self.to_dict()))


def tables_from_dict(data: object) -> LemmaTables:
    if not isinstance(data, dict):
        raise ValueError("lemma table must be a JSON object")
    version = data.get("format_version")
    if version != FORMAT_VERSION or isinstance(version, bool):
        raise ValueError(f"unsupported lemma table format_version: {version!r}")
    name, source, stages = data.get("name"), data.get("source"), data.get("stages")
    if not isinstance(name, str) or not isinstance(source, str) or not isinstance(stages, dict):
        raise ValueError("lemma table needs string 'name' and 'source' and an object 'stages'")
    nodes: dict[str, RuleNode] = {}
    for stage in STAGES:
        node = stages.get(stage)
        if not isinstance(node, dict):
            raise ValueError(f"lemma table is missing stage {stage!r}")
        nodes[stage] = node_from_dict(node)
    return LemmaTables(name, source, **nodes)


def load_tables(path: Path | Traversable) -> LemmaTables:
    """Read a table file; raises ``ValueError`` on an unsupported ``format_version``."""
    return tables_from_dict(json.loads(path.read_text(encoding="utf-8")))
