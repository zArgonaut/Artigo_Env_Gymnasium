from __future__ import annotations

import hashlib
import io
from pathlib import Path

import pytest
import yaml

from artigo_env.cli.main import build_parser, main
from artigo_env.corpus.manifest import SourceFile, SourceManifestEntry

pytestmark = pytest.mark.fast

CONTENT = b"raw shard\n" * 50


def write_manifest(tmp_path: Path, **overrides: object) -> Path:
    entry: dict[str, object] = {
        "id": "news",
        "kind": "hf_parquet",
        "state": "pending",
        "priority": 1,
        "license_id": "unknown",
        "license_note": "internal research use",
        "revision": "abc",
        "files": [
            {
                "path": "train/0001.parquet",
                "sha256": hashlib.sha256(CONTENT).hexdigest(),
                "size": len(CONTENT),
            }
        ],
        "options": {"repo": "someone/news", "body": "text"},
    }
    entry.update(overrides)
    path = tmp_path / "sources.yaml"
    path.write_text(yaml.safe_dump({"version": 1, "sources": [entry]}), encoding="utf-8")
    return path


def run(*argv: str) -> tuple[int, str]:
    out = io.StringIO()
    code = main(list(argv), io.StringIO(""), out)
    return code, out.getvalue()


@pytest.fixture
def data_root(tmp_path, monkeypatch) -> Path:
    root = tmp_path / "data"
    monkeypatch.setenv("ARTIGO_DATA_ROOT", str(root))
    return root


def install_fake_download(monkeypatch, content: bytes) -> list[str]:
    calls: list[str] = []

    def fake(entry: SourceManifestEntry, file: SourceFile, target: Path) -> None:
        calls.append(file.path)
        target.write_bytes(content)

    monkeypatch.setattr("artigo_env.corpus.fetch.hf_download", fake)
    return calls


@pytest.mark.req("REQ-REPO-006")
def test_data_fetch_downloads_and_verifies(tmp_path, data_root, monkeypatch):
    calls = install_fake_download(monkeypatch, CONTENT)
    manifest = write_manifest(tmp_path)
    code, text = run("data", "fetch", "news", "--manifest", str(manifest))
    assert code == 0, text
    assert calls == ["train/0001.parquet"]
    assert (data_root / "raw" / "news" / "train" / "0001.parquet").read_bytes() == CONTENT
    assert "news: state pending, licence unknown" in text
    assert "internal research use" in text
    assert "1 file(s) verified" in text


@pytest.mark.req("REQ-REPO-006")
def test_data_fetch_refuses_tampered_file(tmp_path, data_root, monkeypatch):
    install_fake_download(monkeypatch, CONTENT + b"tampered")
    manifest = write_manifest(tmp_path)
    code, text = run("data", "fetch", "news", "--files", "0001", "--manifest", str(manifest))
    assert code == 2
    assert "does not match the manifest" in text
    assert not (data_root / "raw" / "news" / "train" / "0001.parquet").exists()


@pytest.mark.req("REQ-REPO-006")
def test_data_fetch_needs_accept_terms(tmp_path, data_root, monkeypatch):
    calls = install_fake_download(monkeypatch, CONTENT)
    manifest = write_manifest(tmp_path, terms="Research only.")
    code, text = run("data", "fetch", "news", "--manifest", str(manifest))
    assert code == 2 and "--accept-terms" in text and calls == []
    code, text = run("data", "fetch", "news", "--accept-terms", "--manifest", str(manifest))
    assert code == 0, text


@pytest.mark.req("DEC-024")
def test_data_fetch_manual_source_prints_instructions(tmp_path, data_root, monkeypatch):
    calls = install_fake_download(monkeypatch, CONTENT)
    manifest = write_manifest(tmp_path, access="request", url="https://example.org/ask")
    code, text = run("data", "fetch", "news", "--manifest", str(manifest))
    assert code == 1
    assert "request access from the provider from https://example.org/ask" in text
    assert calls == []


@pytest.mark.req("DEC-024")
def test_data_fetch_real_manifest_extracts_and_unknown_source(data_root):
    code, text = run("data", "fetch", "wikipedia_offline_extracts")
    assert code == 1 and "tools.w.fetch_extracts" in text
    code, text = run("data", "fetch", "demo_synthetic")
    assert code == 0 and "bundled with the package" in text
    code, text = run("data", "fetch", "nope")
    assert code == 2 and "unknown source 'nope'" in text


@pytest.mark.req("DEC-024")
def test_data_group_requires_a_subcommand():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["data"])
    args = build_parser().parse_args(["data", "fetch", "cc_news_pt", "--files", "0003", "0022"])
    assert (args.command, args.data_command, args.files) == ("data", "fetch", ["0003", "0022"])
