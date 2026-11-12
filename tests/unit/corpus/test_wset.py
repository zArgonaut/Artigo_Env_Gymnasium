"""Evaluation set W: title list, eligibility, stratified splits, L_max, W build and W index."""

from __future__ import annotations

import random
from collections import Counter
from pathlib import Path

import pytest

from artigo_env.corpus.decontam.windex import WIndex
from artigo_env.corpus.pipeline import WINDEX_DIR, build, build_w_index
from artigo_env.corpus.store import CorpusStore
from artigo_env.corpus.wset import (
    WPage,
    WSetError,
    default_context,
    eligible,
    l_max,
    load_title_list,
    load_w_splits,
    make_splits,
    measure_snapshot,
    write_w_splits,
)
from tests.unit.corpus.test_pipeline_config import (
    article,
    config,
    file_bytes,
    synthetic_sources,
    write_snapshot,
)

pytestmark = pytest.mark.fast


def snapshot_pages(n: int, seed: int = 3) -> list[tuple[str, str, int, str]]:
    rng = random.Random(seed)
    pages = []
    for i in range(n):
        title, body = article(rng, n_words=100 + 37 * i, n_title=1 + i % 4)
        pages.append((title, title, 1000 + i, body))
    return pages


def write_titles(path: Path, main: list[str], other: list[str] = ()) -> Path:  # type: ignore[assignment]
    lines = ["# list", "# colunas", "titulo\tem_completa\tem_lista\tsublistas"]
    lines += [f"{t}\t1\t1\tX" for t in main] + [f"{t}\t0\t1\tY" for t in other]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


@pytest.fixture
def w_root(tmp_path: Path) -> tuple[Path, Path, list[tuple[str, str, int, str]]]:
    root = tmp_path / "data"
    pages = snapshot_pages(40)
    rng = random.Random(9)
    _, body = article(rng)
    pages += [
        ("Mais Você", "Mais Você", 5001, body),  # no content word
        ("Anexo:Lista de coisas", "Anexo:Lista de coisas", 5002, body),
        ("Extra fora", "Extra fora", 5003, body),  # not in the main list
        ("@", "Arroba Simbolo", 5004, body),  # redirect from a title without content word
        ("vazio", "vazio", 5005, ""),  # empty text
    ]
    pages.append(("Redirecionado", pages[0][1], pages[0][2], pages[0][3]))  # same page twice
    write_snapshot(root, pages, missing=["Sumida"])
    main = [p[0] for p in pages if p[0] != "Extra fora"] + ["Sumida"]
    titles = write_titles(tmp_path / "titles.tsv", main, ["Extra fora"])
    return root, titles, pages


@pytest.mark.req("REQ-COR-013")
def test_load_title_list(tmp_path: Path) -> None:
    entries = load_title_list(write_titles(tmp_path / "t.tsv", ["A", "B"], ["C"]))
    assert [(e.title, e.in_complete, e.in_list) for e in entries] == [
        ("A", True, True),
        ("B", True, True),
        ("C", False, True),
    ]
    (tmp_path / "bad.tsv").write_text("titulo\tem_completa\tem_lista\nA\t2\t1\n", encoding="utf-8")
    with pytest.raises(WSetError):
        load_title_list(tmp_path / "bad.tsv")


@pytest.mark.req("REQ-COR-013", "REQ-COR-017")
def test_eligibility_rules(w_root) -> None:
    root, titles, pages = w_root
    ctx = default_context()
    measure = measure_snapshot(root / "raw" / "wiki" / "snap", ctx)
    assert len(measure.pages) == 44  # 45 distinct pages, one empty
    assert measure.missing == ["Sumida"]
    result = eligible(measure, load_title_list(titles), ctx, cap=1500)
    ex = result.excluded
    assert ex["no_content_word"] == ["Arroba Simbolo", "Mais Você"]
    assert ex["annex_or_project"] == ["Anexo:Lista de coisas"]
    assert ex["not_in_main_list"] == ["Extra fora"]
    assert ex["missing"] == ["Sumida"]
    assert ex["empty"] == ["vazio"]
    too_long = {p[1] for p in pages[:40] if 100 + 37 * pages.index(p) > 1500}
    assert set(ex["too_long"]) <= too_long and ex["too_long"]
    assert all(p.n_body_tokens <= 1500 for p in result.pages)
    redirected = next(p for p in result.pages if p.pageid == pages[0][2])
    assert redirected.requested == tuple(sorted({pages[0][0], "Redirecionado"}))


def fake_pages(n: int) -> list[WPage]:
    rng = random.Random(1)
    return [
        WPage(
            i,
            f"Titulo {i:04d}",
            (f"Titulo {i:04d}",),
            rng.randint(150, 9000),
            rng.choice((1, 1, 2, 2, 2, 3, 4, 5)),
        )
        for i in range(n)
    ]


@pytest.mark.req("REQ-COR-013")
def test_make_splits_sizes_strata_and_determinism() -> None:
    pages = fake_pages(1500)
    rows = make_splits(pages)
    counts = Counter(row.split for row in rows)
    assert counts == {"w-test": 1000, "w-val": 200, "w-pool": 300}
    assert rows == make_splits(list(reversed(pages)))  # input order does not matter
    assert rows != make_splits(pages, seed=1)
    assert [r.title for r in rows] == sorted(r.title for r in rows)
    strata = Counter(r.stratum for r in rows)
    test_strata = Counter(r.stratum for r in rows if r.split == "w-test")
    for label, size in strata.items():  # proportional allocation (largest remainder)
        assert abs(test_strata[label] - 1000 * size / 1500) < 1
    assert {r.stratum.split("-")[0] for r in rows} == {"c1", "c2", "c3", "c4+"}
    with pytest.raises(WSetError):
        make_splits(pages[:1000])


@pytest.mark.req("REQ-COR-017")
def test_l_max() -> None:
    assert l_max(1) == 1024
    assert l_max(1024) == 1024
    assert l_max(1025) == 2048
    assert l_max(15873) == 16384
    assert l_max(99999) == 32768


@pytest.mark.req("REQ-COR-013")
def test_split_file_round_trip(tmp_path: Path) -> None:
    rows = make_splits(fake_pages(30), n_test=10, n_val=5)
    path = tmp_path / "w_splits.tsv"
    write_w_splits(path, rows, max_tokens=5000)
    text = path.read_text(encoding="utf-8")
    assert "L_max 5120" in text
    splits = load_w_splits(path)
    assert len(splits) == 30
    assert splits["titulo 0003"] == next(r.split for r in rows if r.title == "Titulo 0003")
    path.write_text("title\tsplit\nA\tw-extra\n", encoding="utf-8")
    with pytest.raises(WSetError):
        load_w_splits(path)


def run_w_build(root: Path, titles: Path, tmp_path: Path, **fields):
    ctx = default_context()
    measure = measure_snapshot(root / "raw" / "wiki" / "snap", ctx)
    result = eligible(measure, load_title_list(titles), ctx)
    rows = make_splits(result.pages, n_test=10, n_val=5)
    splits = tmp_path / "w_splits.tsv"
    write_w_splits(splits, rows, max_tokens=1)
    cfg = config(kind="evaluation", sources=("wiki",), w_splits=str(splits), **fields)
    return build(cfg, data_root=root, sources=synthetic_sources()), rows


@pytest.mark.req("REQ-COR-013", "REQ-COR-016")
def test_w_build_keeps_every_page_and_writes_the_index(w_root, tmp_path: Path) -> None:
    root, titles, pages = w_root
    result, rows = run_w_build(root, titles, tmp_path)
    assert result.n_docs == 44  # every non-empty page, eligible or not
    expected = {r.title: r.split for r in rows}
    with CorpusStore.open(result.out_dir) as store:
        assert store.manifest["trainable"] is False
        for i in range(store.n_docs):
            row = store.doc(i)
            assert row["split"] == expected.get(store.title_text(i), "w-extra")
            assert row["title_in_body"] in (True, False)
    w = result.report["w"]
    assert w["per_split"]["w-test"] == 10 and w["per_split"]["w-val"] == 5
    assert w["split_titles_not_found"] == 0
    windex = WIndex.load(result.out_dir / WINDEX_DIR)
    for title in ("Sumida", "@", "Arroba Simbolo", "Redirecionado", "Extra fora"):
        assert title in windex.titles
    assert len(windex.near) == 44
    before = file_bytes(result.out_dir / WINDEX_DIR)
    build_w_index(result.build_id, data_root=root, sources=synthetic_sources())
    assert file_bytes(result.out_dir / WINDEX_DIR) == before
