"""End-to-end corpus builds on synthetic data: resume, determinism, workers, duplicates, W."""

from __future__ import annotations

from pathlib import Path

import pytest

from artigo_env.corpus import pipeline
from artigo_env.corpus.pipeline import build
from artigo_env.corpus.store import CorpusStore
from tests.leakage.test_corpus_isolation import build_w, contaminated_rows
from tests.unit.corpus.test_pipeline_config import (
    config,
    file_bytes,
    news_rows,
    synthetic_sources,
    write_shard,
)

pytestmark = pytest.mark.slow

N_SHARDS = 5


def write_raw(root: Path) -> None:
    for k in range(N_SHARDS):
        rows = news_rows(100 + k, 15)
        if k == 1:
            rows[0]["text"] = "curto"  # removed by S5
        write_shard(root, "news", f"{k:04d}", rows)
    blog = news_rows(100, 6, host="blog.net")  # same texts as shard 0000 of news
    blog[1]["url"] = "https://www.noticias.com.br/101/3?utm_source=x"  # URL of a news document
    write_shard(root, "blog", "0000", blog)


def make_config(**fields):
    return config(sources=("news", "blog"), **fields)


class Interrupt(Exception):
    pass


@pytest.mark.req("REQ-COR-005", "REQ-COR-016")
def test_resume_after_interruption_gives_identical_bytes(tmp_path, monkeypatch) -> None:
    reference_root, root = tmp_path / "a", tmp_path / "b"
    write_raw(reference_root)
    write_raw(root)
    reference = build(make_config(), data_root=reference_root, sources=synthetic_sources())

    real = pipeline.run_playable_shard
    calls = {"n": 0}

    def flaky(task):
        calls["n"] += 1
        if calls["n"] == 3:
            raise Interrupt
        return real(task)

    monkeypatch.setattr(pipeline, "run_playable_shard", flaky)
    with pytest.raises(Interrupt):
        build(make_config(), data_root=root, sources=synthetic_sources())
    assert not (root / "corpus" / reference.build_id).exists()
    monkeypatch.setattr(pipeline, "run_playable_shard", real)
    resumed = build(make_config(), data_root=root, sources=synthetic_sources())
    assert resumed.build_id == reference.build_id
    assert resumed.shards_reused == 2 and resumed.shards_processed == N_SHARDS + 1 - 2
    assert file_bytes(resumed.out_dir) == file_bytes(reference.out_dir)


@pytest.mark.req("REQ-COR-016")
def test_one_and_three_workers_give_identical_bytes(tmp_path) -> None:
    one_root, three_root = tmp_path / "one", tmp_path / "three"
    write_raw(one_root)
    write_raw(three_root)
    one = build(make_config(workers=1), data_root=one_root, sources=synthetic_sources())
    three = build(make_config(workers=3), data_root=three_root, sources=synthetic_sources())
    assert one.build_id == three.build_id
    assert file_bytes(one.out_dir) == file_bytes(three.out_dir)
    again = build(make_config(), data_root=one_root, sources=synthetic_sources(), recompile=True)
    assert again.shards_reused == N_SHARDS + 1
    assert file_bytes(again.out_dir) == file_bytes(one.out_dir)


@pytest.mark.req("REQ-COR-011")
def test_planted_duplicates_across_sources(tmp_path) -> None:
    root = tmp_path / "data"
    write_raw(root)
    result = build(make_config(), data_root=root, sources=synthetic_sources())
    dedup = result.report["stages"]["dedup"]
    # blog repeats 6 news texts: 5 exact bodies + 1 that also carries a news URL
    assert dedup["removed"]["dup_body"] + dedup["removed"]["dup_url"] == 6
    assert dedup["removed"]["dup_url"] == 1
    assert result.report["by_source"]["blog"]["dedup"]["out"] == 0
    with CorpusStore.open(result.out_dir) as store:
        sources = {store.doc(i)["source_id"] for i in range(store.n_docs)}
        keys = [(store.doc(i)["source_id"], store.doc(i)["doc_id"]) for i in range(store.n_docs)]
    assert sources == {"news"}  # news has the lower priority number, so it is kept
    assert keys == sorted(keys)
    assert result.n_docs == N_SHARDS * 15 - 1


@pytest.mark.req("REQ-COR-005", "REQ-COR-010", "REQ-COR-016")
def test_decontaminated_build_resumes_and_is_reproducible(tmp_path, monkeypatch) -> None:
    roots = [tmp_path / "x", tmp_path / "y"]
    ids = []
    for root in roots:
        w_id, pages = build_w(root, tmp_path / root.name)
        ids.append(w_id)
        write_shard(root, "news", "0000", news_rows(7, 10) + contaminated_rows(pages))
        write_shard(root, "news", "0001", news_rows(8, 10))
    assert ids[0] == ids[1]
    cfg = config(kind="train", w_index=ids[0], workers=2)
    reference = build(cfg, data_root=roots[0], sources=synthetic_sources())

    real = pipeline.run_judge
    monkeypatch.setattr(pipeline, "run_judge", lambda task: (_ for _ in ()).throw(Interrupt()))
    with pytest.raises(Interrupt):
        build(config(kind="train", w_index=ids[0]), data_root=roots[1], sources=synthetic_sources())
    monkeypatch.setattr(pipeline, "run_judge", real)
    resumed = build(cfg, data_root=roots[1], sources=synthetic_sources())
    assert resumed.build_id == reference.build_id
    assert resumed.shards_reused == 2
    assert sum(resumed.report["stages"]["decontam"]["removed"].values()) == 4
    assert file_bytes(resumed.out_dir) == file_bytes(reference.out_dir)
