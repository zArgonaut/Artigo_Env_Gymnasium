"""The packaged demo corpus and vocabularies (decision 7 of the F4 plan)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from artigo_env.core.resources import resource_path
from artigo_env.engine.rulesets import PRESETS, get_ruleset
from tools.demo.build_demo import build_demo

DEMO = Path(str(resource_path("demo")))


def _digest(root: Path) -> dict[str, str]:
    return {
        p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


@pytest.mark.fast
@pytest.mark.req("REQ-ENV-004")
def test_rebuild_gives_identical_bytes(tmp_path):
    summary = build_demo(tmp_path)
    for part in ("corpus", "vocab", "knowledge"):
        assert _digest(tmp_path / part) == _digest(DEMO / part), part
    manifest = json.loads((DEMO / "corpus" / "manifest.json").read_text(encoding="utf-8"))
    assert summary["build_id"] == manifest["build_id"] and manifest["l_max"] == 1024
    assert manifest["n_docs"] == 30
    assert sorted(summary["knowledge"]) == sorted(p.name for p in (DEMO / "vocab").iterdir())


@pytest.mark.fast
@pytest.mark.req("REQ-ENV-004")
def test_one_vocabulary_per_preset_with_cached_map():
    manifest = json.loads((DEMO / "corpus" / "manifest.json").read_text(encoding="utf-8"))
    found = {}
    for path in sorted((DEMO / "vocab").glob("*/manifest.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["build_id"] == manifest["build_id"]
        assert (path.parent / "maps" / manifest["build_id"] / "manifest.json").is_file()
        found[data["ruleset_name"]] = data["ruleset_hash"]
    assert set(found) == set(PRESETS.names())
    for name, ruleset_hash in found.items():
        assert get_ruleset(name).content_hash() == ruleset_hash
