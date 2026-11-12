from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest

from artigo_env.corpus.fetch import (
    ChecksumMismatchError,
    FetchError,
    ManualFetchRequired,
    TermsNotAcceptedError,
    fetch,
    hf_download,
    select_files,
    sha256_file,
)
from artigo_env.corpus.manifest import SourceFile, SourceManifestEntry, load_sources

pytestmark = pytest.mark.fast

GOOD = b"parquet bytes of shard one\n" * 100
OTHER = b"second shard\n" * 10


def pin(path: str, content: bytes) -> dict[str, object]:
    return {"path": path, "sha256": hashlib.sha256(content).hexdigest(), "size": len(content)}


def make_entry(**fields: object) -> SourceManifestEntry:
    base: dict[str, object] = {
        "id": "news",
        "kind": "hf_parquet",
        "state": "pending",
        "priority": 1,
        "license_id": "unknown",
        "revision": "abc123",
        "files": [pin("data/train/0001.parquet", GOOD), pin("data/train/0002.parquet", OTHER)],
        "options": {"repo": "someone/news", "body": "text"},
    }
    base.update(fields)
    return SourceManifestEntry.from_dict(base)


class FakeDownload:
    """Writes the given bytes for each requested path and records the calls."""

    def __init__(self, contents: dict[str, bytes]) -> None:
        self.contents = contents
        self.calls: list[str] = []

    def __call__(self, entry: SourceManifestEntry, file: SourceFile, target: Path) -> None:
        self.calls.append(file.path)
        target.write_bytes(self.contents[file.path])


CORRECT = {"data/train/0001.parquet": GOOD, "data/train/0002.parquet": OTHER}


@pytest.mark.req("REQ-REPO-006")
def test_sha256_file_reads_in_blocks(tmp_path):
    path = tmp_path / "big.bin"
    content = bytes(range(256)) * 9000  # > 2 MiB, several blocks
    path.write_bytes(content)
    assert sha256_file(path) == hashlib.sha256(content).hexdigest()


@pytest.mark.req("REQ-REPO-006")
def test_fetch_downloads_and_verifies(tmp_path):
    download = FakeDownload(CORRECT)
    messages: list[str] = []
    paths = fetch(make_entry(), tmp_path, download=download, log=messages.append)
    assert paths == [
        tmp_path / "news" / "data" / "train" / "0001.parquet",
        tmp_path / "news" / "data" / "train" / "0002.parquet",
    ]
    assert paths[0].read_bytes() == GOOD
    assert download.calls == ["data/train/0001.parquet", "data/train/0002.parquet"]
    assert any(m.startswith("ok data/train/0001.parquet") for m in messages)
    # a second run verifies the files in place and downloads nothing
    again = FakeDownload(CORRECT)
    assert fetch(make_entry(), tmp_path, download=again) == paths
    assert again.calls == []


@pytest.mark.req("REQ-REPO-006")
def test_tampered_download_is_refused_and_deleted(tmp_path):
    tampered = dict(CORRECT, **{"data/train/0001.parquet": GOOD[:-1] + b"X"})
    with pytest.raises(ChecksumMismatchError, match="SHA-256"):
        fetch(make_entry(), tmp_path, download=FakeDownload(tampered))
    assert not (tmp_path / "news" / "data" / "train" / "0001.parquet").exists()


@pytest.mark.req("REQ-REPO-006")
def test_truncated_download_is_refused_and_deleted(tmp_path):
    truncated = dict(CORRECT, **{"data/train/0002.parquet": OTHER[:5]})
    with pytest.raises(ChecksumMismatchError, match="size 5 bytes"):
        fetch(make_entry(), tmp_path, download=FakeDownload(truncated))
    assert not (tmp_path / "news" / "data" / "train" / "0002.parquet").exists()
    assert (tmp_path / "news" / "data" / "train" / "0001.parquet").read_bytes() == GOOD


@pytest.mark.req("REQ-REPO-006")
def test_tampered_local_file_is_deleted_and_downloaded_again(tmp_path):
    target = tmp_path / "news" / "data" / "train" / "0001.parquet"
    target.parent.mkdir(parents=True)
    target.write_bytes(GOOD[:-1] + b"X")
    download = FakeDownload(CORRECT)
    fetch(make_entry(), tmp_path, files=["0001"], download=download)
    assert download.calls == ["data/train/0001.parquet"]
    assert target.read_bytes() == GOOD


@pytest.mark.req("REQ-REPO-006")
def test_manual_source_verifies_placed_files(tmp_path):
    entry = make_entry(access="registration", url="https://example.org/apply")
    target = tmp_path / "news" / "data" / "train" / "0001.parquet"
    target.parent.mkdir(parents=True)
    target.write_bytes(GOOD)
    download = FakeDownload(CORRECT)
    with pytest.raises(ManualFetchRequired, match="register with the provider") as info:
        fetch(entry, tmp_path, download=download)
    assert "missing: data/train/0002.parquet" in str(info.value)
    assert download.calls == []  # never downloads
    assert fetch(entry, tmp_path, files=["0001.parquet"], download=download) == [target]
    target.write_bytes(GOOD + b"extra")
    with pytest.raises(ChecksumMismatchError, match="was deleted"):
        fetch(entry, tmp_path, files=["0001.parquet"], download=download)
    assert not target.exists()


@pytest.mark.req("REQ-REPO-006")
def test_terms_must_be_accepted(tmp_path):
    entry = make_entry(terms="Use only for research.")
    with pytest.raises(TermsNotAcceptedError, match="Use only for research"):
        fetch(entry, tmp_path, download=FakeDownload(CORRECT))
    assert not (tmp_path / "news").exists()
    assert len(fetch(entry, tmp_path, accept_terms=True, download=FakeDownload(CORRECT))) == 2


@pytest.mark.req("REQ-REPO-006")
def test_file_selection(tmp_path):
    entry = make_entry()
    assert [f.path for f in select_files(entry, ["0002"])] == ["data/train/0002.parquet"]
    assert len(select_files(entry, ["data/train/0001.parquet", "0001.parquet"])) == 1
    with pytest.raises(FetchError, match="not a pinned file"):
        select_files(entry, ["0003"])


@pytest.mark.req("DEC-024")
def test_sources_without_downloadable_files(tmp_path):
    sources = load_sources()
    with pytest.raises(FetchError, match="no files are pinned"):
        fetch(sources["recognasumm"], tmp_path, download=FakeDownload({}))
    with pytest.raises(ManualFetchRequired, match="tools.w.fetch_extracts"):
        fetch(sources["wikipedia_offline_extracts"], tmp_path, download=FakeDownload({}))
    assert fetch(sources["demo_synthetic"], tmp_path, download=FakeDownload({})) == []
    assert list(tmp_path.iterdir()) == []


@pytest.mark.req("DEC-024")
def test_hf_download_without_huggingface_hub_has_install_hint(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)  # import now raises ImportError
    entry = make_entry()
    with pytest.raises(FetchError, match=r"artigo-env\[data\]"):
        hf_download(entry, entry.files[0], tmp_path / "news" / "data" / "train" / "0001.parquet")


@pytest.mark.req("DEC-024")
def test_hf_download_calls_hub_with_pinned_revision(tmp_path, monkeypatch):
    calls: list[dict[str, object]] = []

    class FakeHub:
        @staticmethod
        def hf_hub_download(**kwargs: object) -> str:
            calls.append(kwargs)
            local = Path(str(kwargs["local_dir"])) / str(kwargs["filename"])
            local.parent.mkdir(parents=True, exist_ok=True)
            local.write_bytes(GOOD)
            return str(local)

    monkeypatch.setitem(sys.modules, "huggingface_hub", FakeHub)
    paths = fetch(make_entry(), tmp_path, files=["0001"])
    assert paths[0].read_bytes() == GOOD
    assert calls == [
        {
            "repo_id": "someone/news",
            "filename": "data/train/0001.parquet",
            "repo_type": "dataset",
            "revision": "abc123",
            "local_dir": tmp_path / "news",
        }
    ]
