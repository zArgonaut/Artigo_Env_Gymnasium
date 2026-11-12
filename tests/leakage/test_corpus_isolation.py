"""Isolation of a training build from W (REQ-COR-010) and of the demo corpus (REQ-REPO-007)."""

from __future__ import annotations

import json
import os
import random
from pathlib import Path

import numpy as np
import pytest

from artigo_env.corpus.decontam import (
    N_GRAM,
    TitleBarrier,
    WGramIndex,
    covered_new_tokens,
    gram_hashes,
    gram_tokens,
    jaccard,
    shingle_hashes,
)
from artigo_env.corpus.pipeline import build
from artigo_env.corpus.store import CorpusStore
from artigo_env.corpus.wset import (
    default_context,
    eligible,
    load_title_list,
    make_splits,
    measure_snapshot,
    write_w_splits,
)
from tests.unit.corpus.test_pipeline_config import (
    VOCAB,
    article,
    config,
    news_rows,
    synthetic_sources,
    write_shard,
    write_snapshot,
)


def build_w(root: Path, tmp_path: Path) -> tuple[str, list[tuple[str, str, int, str]]]:
    rng = random.Random(11)
    pages = []
    for i in range(24):
        title, body = article(rng, n_words=260, n_title=1 + i % 2)
        pages.append((title, title, 500 + i, body))
    snapshot = write_snapshot(root, pages, missing=["Titulo Sem Pagina"])
    titles = tmp_path / "titles.tsv"
    titles.write_text(
        "#\n#\ntitulo\tem_completa\tem_lista\tsublistas\n"
        + "".join(f"{p[0]}\t1\t1\tX\n" for p in pages),
        encoding="utf-8",
    )
    ctx = default_context()
    result = eligible(measure_snapshot(snapshot, ctx), load_title_list(titles), ctx)
    splits = tmp_path / "w_splits.tsv"
    write_w_splits(splits, make_splits(result.pages, n_test=8, n_val=4), max_tokens=260)
    cfg = config(name="w", kind="evaluation", sources=("wiki",), w_splits=str(splits))
    return build(cfg, data_root=root, sources=synthetic_sources()).build_id, pages


def contaminated_rows(pages: list[tuple[str, str, int, str]]) -> list[dict[str, str]]:
    """Training documents that copy W in each of the ways the barriers must catch."""
    rng = random.Random(12)
    rows: list[dict[str, str]] = []

    def add(kind: str, title: str, body: str) -> None:
        rows.append(
            {
                "title": title,
                "text": body,
                "url": f"https://planted.com.br/{kind}",
                "date": "2021-01-01",
            }
        )

    # same title as a W page (first letter in lower case), different text
    w_title = pages[0][1]
    words = w_title.split()
    _, body = article(rng)
    add("title", w_title[0].lower() + w_title[1:], " ".join(words) + " " + body)
    # verbatim copy of a W body under another title
    title, _ = article(rng, n_title=1)
    add("copy", title, title.lower() + " " + pages[1][3])
    # 20-word excerpt of W inside an otherwise clean document
    title, body = article(rng)
    excerpt = " ".join(pages[2][3].split()[40:60])
    add("excerpt", title, body + "\n\n" + excerpt)
    # near duplicate: W body with a few words changed, sentence boundaries kept
    title, _ = article(rng, n_title=1)
    near = pages[3][3].split(" ")
    for i in range(0, len(near), 60):
        near[i] = rng.choice(VOCAB)
    add("near", title, title.lower() + " " + " ".join(near))
    return rows


@pytest.mark.fast
@pytest.mark.req("REQ-COR-010")
def test_training_build_shares_nothing_with_w(tmp_path: Path) -> None:
    root = tmp_path / "data"
    w_id, pages = build_w(root, tmp_path)
    clean = news_rows(21, 20)
    write_shard(root, "news", "0001", clean[:10] + contaminated_rows(pages))
    write_shard(root, "news", "0002", clean[10:])
    result = build(config(kind="train", w_index=w_id), data_root=root, sources=synthetic_sources())
    removed = result.report["stages"]["decontam"]["removed"]
    assert result.report["decontam"]["status"] == "applied"
    assert sum(removed.values()) == 4 and removed["w_title"] == 1
    assert result.n_docs == 20

    w_tokens = [gram_tokens(p[3]) for p in pages]
    barrier = TitleBarrier([p[1] for p in pages] + ["Titulo Sem Pagina"])
    w_grams = WGramIndex.build(w_tokens)
    w_shingles = [shingle_hashes(t) for t in w_tokens]
    with CorpusStore.open(result.out_dir) as store:
        assert store.manifest["trainable"] is True
        for i in range(store.n_docs):
            assert store.doc(i)["url"].startswith("https://www.noticias.com.br/")
            assert store.title_text(i) not in barrier
            tokens = gram_tokens(store.body_text(i))
            grams = gram_hashes(tokens, N_GRAM)
            common = w_grams.contains(grams)
            no_frequent = np.zeros(common.shape, dtype=np.bool_)
            assert covered_new_tokens(len(tokens), common, no_frequent) < 7
            shingles = shingle_hashes(tokens)
            assert max(jaccard(shingles, w) for w in w_shingles) < 0.8


def _local_titles() -> Path | None:
    root = os.environ.get("ARTIGO_DATA_ROOT", "").strip()
    if not root:
        return None
    path = Path(root) / "measurements" / "wikipedia-offline-titulos.tsv"
    return path if path.is_file() else None


@pytest.mark.local
@pytest.mark.req("REQ-REPO-007")
def test_demo_titles_are_not_w_titles(repo_root: Path) -> None:
    path = _local_titles()
    if path is None:
        pytest.skip("title list not found under ARTIGO_DATA_ROOT/measurements/")
    barrier = TitleBarrier(entry.title for entry in load_title_list(path))
    articles = repo_root / "src" / "artigo_env" / "assets" / "demo" / "articles.jsonl"
    titles = [json.loads(line)["title"] for line in articles.read_text("utf-8").splitlines()]
    assert [t for t in titles if t in barrier] == []
