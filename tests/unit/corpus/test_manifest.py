from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from artigo_env.core.errors import ArtigoEnvError
from artigo_env.corpus.manifest import (
    SourceManifestEntry,
    SourceManifestError,
    get_source,
    load_sources,
    parse_sources,
    require_trainable,
)
from artigo_env.corpus.sources import SOURCE_KINDS, open_source

pytestmark = pytest.mark.fast

SHA = "0" * 64


def entry_dict(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "id": "news",
        "kind": "hf_parquet",
        "state": "pending",
        "priority": 10,
        "license_id": "unknown",
        "options": {"body": "text"},
    }
    base.update(overrides)
    return base


def write_manifest(tmp_path: Path, sources: list[dict[str, object]]) -> Path:
    path = tmp_path / "sources.yaml"
    path.write_text(yaml.safe_dump({"version": 1, "sources": sources}), encoding="utf-8")
    return path


@pytest.mark.req("DEC-024", "REQ-COR-002")
def test_real_manifest_loads_with_expected_sources():
    sources = load_sources()
    assert list(sources) == [
        "cc_news_pt",
        "recognasumm",
        "wikipedia_offline_extracts",
        "demo_synthetic",
    ]
    cc = sources["cc_news_pt"]
    assert cc.state == "pending" and not cc.trainable
    assert cc.revision == "98c2d2ab1c43c232f51da9f56ea7f03e08636286"
    assert [f.path.rsplit("/", 1)[1] for f in cc.files] == [
        "0003.parquet",
        "0022.parquet",
        "0045.parquet",
    ]
    assert all(f.size > 0 and len(f.sha256) == 64 for f in cc.files)
    assert cc.redistributable is False and cc.variant is None
    assert cc.column("body") == "text" and cc.column("id") is None
    assert sources["recognasumm"].state == "pending" and sources["recognasumm"].files == ()
    w = sources["wikipedia_offline_extracts"]
    assert (w.kind, w.state, w.license_id, w.redistributable) == (
        "wikipedia_extracts",
        "evaluation",
        "CC-BY-SA-4.0",
        True,
    )
    demo = sources["demo_synthetic"]
    assert (demo.kind, demo.state, demo.license_id, demo.redistributable) == (
        "demo",
        "accepted",
        "MIT",
        True,
    )
    for entry in sources.values():  # every kind in the manifest has a registered reader
        assert entry.kind in SOURCE_KINDS
        assert open_source(entry).source_id == entry.id


@pytest.mark.req("DEC-024")
def test_manifest_paths_are_relative():
    for entry in load_sources().values():
        for file in entry.files:
            assert not file.path.startswith("/") and ":" not in file.path
        assert entry.source_dir(Path("raw")).parts[0] == "raw"


@pytest.mark.req("REQ-COR-002")
def test_pending_source_is_refused_for_training():
    pending = SourceManifestEntry.from_dict(entry_dict(state="pending"))
    with pytest.raises(ArtigoEnvError, match="not 'accepted'"):
        require_trainable(pending)
    for state in ("rejected", "stats-only", "evaluation"):
        with pytest.raises(ArtigoEnvError):
            require_trainable(SourceManifestEntry.from_dict(entry_dict(state=state)))
    require_trainable(SourceManifestEntry.from_dict(entry_dict(state="accepted")))


@pytest.mark.req("REQ-COR-002")
def test_real_cc_news_is_not_trainable_yet():
    with pytest.raises(ArtigoEnvError):
        require_trainable(get_source("cc_news_pt"))


@pytest.mark.req("REQ-COR-002")
def test_duplicate_id_fails(tmp_path):
    path = write_manifest(tmp_path, [entry_dict(), entry_dict(priority=20)])
    with pytest.raises(SourceManifestError, match="duplicate source id 'news'"):
        load_sources(path)


@pytest.mark.req("REQ-COR-002")
def test_invalid_state_fails(tmp_path):
    path = write_manifest(tmp_path, [entry_dict(state="approved")])
    with pytest.raises(SourceManifestError, match="invalid state 'approved'"):
        load_sources(path)


@pytest.mark.req("REQ-COR-001")
@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"kind": "csv"}, "unknown kind"),
        ({"access": "paid"}, "invalid access"),
        ({"id": "Bad Id"}, "invalid source id"),
        ({"variant": "pt"}, "variant"),
        ({"options": {"colour": "x"}}, "unknown option"),
        ({"options": {"subdir": "../escape"}}, "relative POSIX path"),
        ({"redistributable": "yes"}, "redistributable"),
        ({"surprise": 1}, "unknown keys"),
    ],
)
def test_invalid_entries_fail(overrides, message):
    with pytest.raises(SourceManifestError, match=message):
        parse_sources({"version": 1, "sources": [entry_dict(**overrides)]})


@pytest.mark.req("REQ-REPO-006")
@pytest.mark.parametrize(
    ("file", "message"),
    [
        ({"path": "/abs/x.parquet", "sha256": SHA, "size": 1}, "relative POSIX path"),
        ({"path": "C:/x.parquet", "sha256": SHA, "size": 1}, "relative POSIX path"),
        ({"path": "a/../x.parquet", "sha256": SHA, "size": 1}, "relative POSIX path"),
        ({"path": "x.parquet", "sha256": "ABC", "size": 1}, "64 lowercase hex"),
        ({"path": "x.parquet", "sha256": SHA, "size": -1}, "non-negative"),
        ({"path": "x.parquet", "sha256": SHA}, "missing key"),
    ],
)
def test_invalid_pinned_files_fail(file, message):
    with pytest.raises(SourceManifestError, match=message):
        parse_sources({"version": 1, "sources": [entry_dict(files=[file])]})


@pytest.mark.req("REQ-REPO-006")
def test_unquoted_yaml_date_option_is_rejected(tmp_path):
    path = tmp_path / "sources.yaml"
    path.write_text(
        "version: 1\nsources:\n  - id: w\n    kind: wikipedia_extracts\n    state: evaluation\n"
        "    priority: 1\n    license_id: CC-BY-SA-4.0\n    options:\n      subdir: 2026-11-12\n",
        encoding="utf-8",
    )
    with pytest.raises(SourceManifestError, match="quote it in YAML"):
        load_sources(path)


@pytest.mark.req("DEC-024")
def test_unknown_source_lists_known_ids():
    with pytest.raises(SourceManifestError, match="known: cc_news_pt, demo_synthetic"):
        get_source("nope")


@pytest.mark.req("REQ-REPO-002")
def test_entry_round_trips_and_hashes_stably():
    entry = load_sources()["cc_news_pt"]
    again = SourceManifestEntry.from_dict(entry.to_dict())
    assert again == entry
    assert again.content_hash() == entry.content_hash()
