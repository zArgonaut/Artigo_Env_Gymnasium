"""``artigo-env data build|report|w-splits|windex`` on synthetic data."""

from __future__ import annotations

import io
import json
import random
from pathlib import Path

import pytest
import yaml

from artigo_env.cli.main import main
from tests.unit.corpus.test_pipeline_config import (
    article,
    news_rows,
    write_shard,
    write_snapshot,
)

pytestmark = pytest.mark.fast


def run(*argv: str) -> tuple[int, str]:
    out = io.StringIO()
    code = main(list(argv), io.StringIO(""), out)
    return code, out.getvalue()


@pytest.fixture
def data_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "data"
    monkeypatch.setenv("ARTIGO_DATA_ROOT", str(root))
    return root


def write_manifest(tmp_path: Path, state: str = "pending") -> Path:
    sources = [
        {
            "id": "news",
            "kind": "hf_parquet",
            "state": state,
            "priority": 10,
            "license_id": "unknown",
            "options": {"subdir": "train", "title": "title", "body": "text", "url": "url"},
        },
        {
            "id": "wiki",
            "kind": "wikipedia_extracts",
            "state": "evaluation",
            "priority": 100,
            "license_id": "CC-BY-SA-4.0",
            "options": {"raw_dir": "wiki", "subdir": "snap"},
        },
    ]
    path = tmp_path / "sources.yaml"
    path.write_text(yaml.safe_dump({"version": 1, "sources": sources}), encoding="utf-8")
    return path


def write_config(tmp_path: Path, **fields: object) -> Path:
    data: dict[str, object] = {
        "name": "cli",
        "kind": "measurement",
        "sources": ["news"],
        "lemmatizer": "open-tables",
        "lemma_views": ["identity"],
        "workers": 1,
    }
    data.update(fields)
    path = tmp_path / "cli.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


@pytest.mark.req("REQ-COR-005", "REQ-COR-015")
def test_build_then_report(tmp_path: Path, data_root: Path) -> None:
    write_shard(data_root, "news", "0001", news_rows(1, 8))
    write_shard(data_root, "news", "0002", news_rows(2, 8))
    manifest = write_manifest(tmp_path)
    cfg = write_config(tmp_path)
    args = ("data", "build", str(cfg), "--manifest", str(manifest), "--lemmatizer", "identity")
    code, text = run(*args, "--shards", "train__0002")
    assert code == 0, text
    assert "8 document(s)" in text
    build_id = text.split("build_id ")[1].split()[0]
    code, again = run(*args, "--shards", "train__0002", "--recompile")
    assert code == 0 and f"build_id {build_id}" in again
    assert "P1: reusing 1 finished shard(s)" in again
    code, report = run("data", "report", build_id)
    assert code == 0 and "decontam: skipped" in report
    code, raw = run("data", "report", build_id, "--json")
    assert json.loads(raw)["n_docs"] == 8
    assert run("data", "report", "0123456789abcdef")[0] == 2


@pytest.mark.req("REQ-COR-002")
def test_build_errors_are_reported(tmp_path: Path, data_root: Path) -> None:
    write_shard(data_root, "news", "0001", news_rows(1, 3))
    manifest = write_manifest(tmp_path, state="pending")
    cfg = write_config(tmp_path, kind="train", w_index="0" * 16)
    code, text = run("data", "build", str(cfg), "--manifest", str(manifest))
    assert code == 2 and "not 'accepted'" in text
    cfg = write_config(tmp_path)  # open-tables is not loaded in tests unless installed
    code, text = run("data", "build", str(cfg), "--manifest", str(manifest), "--lemmatizer", "nope")
    assert code == 2 and "error:" in text


@pytest.mark.req("REQ-COR-013")
def test_w_splits_and_windex(tmp_path: Path, data_root: Path) -> None:
    rng = random.Random(4)
    pages = []
    for i in range(30):
        title, body = article(rng, n_words=120 + 20 * i, n_title=1 + i % 3)
        pages.append((title, title, 100 + i, body))
    snapshot = write_snapshot(data_root, pages, missing=["Perdida"])
    titles = tmp_path / "titles.tsv"
    titles.write_text(
        "# a\n# b\ntitulo\tem_completa\tem_lista\tsublistas\n"
        + "".join(f"{p[0]}\t1\t1\tX\n" for p in pages)
        + "Perdida\t1\t1\tX\n",
        encoding="utf-8",
    )
    out = tmp_path / "w_splits.tsv"
    code, text = run(
        "data",
        "w-splits",
        "--snapshot",
        str(snapshot),
        "--titles",
        str(titles),
        "--out",
        str(out),
        "--n-test",
        "10",
        "--n-val",
        "5",
    )
    assert code == 0, text
    assert "eligible: 30; w-test 10, w-val 5, w-pool 15" in text
    assert "excluded (missing): 1" in text
    assert "L_max: 1024" in text
    manifest = write_manifest(tmp_path)
    cfg = write_config(
        tmp_path, kind="evaluation", sources=["wiki"], lemmatizer="identity", w_splits=str(out)
    )
    code, text = run("data", "build", str(cfg), "--manifest", str(manifest))
    assert code == 0, text
    build_id = text.split("build_id ")[1].split()[0]
    assert "W index written" in text
    code, text = run("data", "windex", build_id, "--manifest", str(manifest))
    assert code == 0 and "content hash" in text
