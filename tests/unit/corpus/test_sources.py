from __future__ import annotations

import datetime as dt
import gzip
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from artigo_env.core.errors import ArtigoEnvError
from artigo_env.corpus.manifest import SourceManifestEntry, load_sources
from artigo_env.corpus.sources import SOURCE_KINDS, SourceReadError, open_source
from artigo_env.corpus.sources.demo import DemoSource
from artigo_env.corpus.sources.hf_parquet import HfParquetSource
from artigo_env.corpus.types import DocumentSource, RawDocument

pytestmark = pytest.mark.fast


def make_entry(**fields: object) -> SourceManifestEntry:
    base: dict[str, object] = {"state": "pending", "priority": 1, "license_id": "lic"}
    base.update(fields)
    return SourceManifestEntry.from_dict(base)


def read_all(source: DocumentSource, raw_root: Path) -> list[RawDocument]:
    return [doc for shard in source.iter_shards(raw_root) for doc in source.read_shard(shard)]


def assert_contract(source: DocumentSource, raw_root: Path) -> list[RawDocument]:
    """Deterministic shard order, stable ids across two reads, ids unique in the source."""
    shards = list(source.iter_shards(raw_root))
    assert [s.shard_id for s in shards] == sorted(s.shard_id for s in shards)
    assert all(s.source_id == source.source_id for s in shards)
    first = read_all(source, raw_root)
    second = read_all(source, raw_root)
    assert first == second
    assert len({d.doc_id for d in first}) == len(first)
    assert all(d.source_id == source.source_id for d in first)
    return first


# --- hf_parquet ------------------------------------------------------------------------------


def write_parquet(path: Path, n: int, start: int = 0, with_id: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns: dict[str, list[object]] = {
        "headline": [f"Title {start + i}" for i in range(n)],
        "text": [f"Body {start + i}." if i != 1 else None for i in range(n)],
        "link": [f"https://example.com.br/{start + i}" for i in range(n)],
        "published": [dt.date(2020, 1, 1 + (i % 28)) for i in range(n)],
        "ignored": ["x"] * n,
    }
    if with_id:
        columns["uid"] = [f"u{start + i}" for i in range(n)]
    pq.write_table(pa.table(columns), path, row_group_size=3)


PARQUET_OPTIONS = {
    "title": "headline",
    "body": "text",
    "url": "link",
    "date": "published",
    "subdir": "data",
}


@pytest.mark.req("REQ-COR-001")
def test_hf_parquet_contract(tmp_path, monkeypatch):
    monkeypatch.setattr("artigo_env.corpus.sources.hf_parquet.BATCH_ROWS", 4)
    raw = tmp_path / "raw"
    # written out of order on purpose; a hidden cache dir and a file outside subdir are ignored
    write_parquet(raw / "news" / "data" / "train" / "0002.parquet", 5, start=100)
    write_parquet(raw / "news" / "data" / "train" / "0001.parquet", 7)
    write_parquet(raw / "news" / "data" / ".cache" / "junk.parquet", 2)
    write_parquet(raw / "news" / "other" / "x.parquet", 2)
    entry = make_entry(id="news", kind="hf_parquet", variant="pt-BR", options=PARQUET_OPTIONS)
    source = open_source(entry)
    assert isinstance(source, HfParquetSource)
    shards = list(source.iter_shards(raw))
    assert [s.shard_id for s in shards] == ["data__train__0001", "data__train__0002"]
    docs = assert_contract(source, raw)
    assert len(docs) == 12
    assert docs[0] == RawDocument(
        source_id="news",
        doc_id="data__train__0001:0",
        title="Title 0",
        body="Body 0.",
        url="https://example.com.br/0",
        date="2020-01-01",
        license_id="lic",
        provenance={"shard": "data__train__0001"},
        variant="pt-BR",
    )
    assert docs[1].body == ""  # null body reads as empty (removed later by S3)
    assert [d.doc_id for d in docs[5:8]] == [
        "data__train__0001:5",
        "data__train__0001:6",
        "data__train__0002:0",
    ]
    assert docs[7].title == "Title 100"


@pytest.mark.req("REQ-COR-001")
def test_hf_parquet_id_column_and_missing_column(tmp_path):
    raw = tmp_path / "raw"
    write_parquet(raw / "news" / "a.parquet", 3, with_id=True)
    entry = make_entry(id="news", kind="hf_parquet", options={"body": "text", "id": "uid"})
    docs = assert_contract(open_source(entry), raw)
    assert [d.doc_id for d in docs] == ["u0", "u1", "u2"]
    assert docs[0].title is None and docs[0].url is None

    bad = make_entry(id="news", kind="hf_parquet", options={"body": "content"})
    with pytest.raises(SourceReadError, match=r"mapped columns \['content'\] not found"):
        read_all(open_source(bad), raw)


@pytest.mark.req("REQ-COR-001")
def test_hf_parquet_reads_only_mapped_columns(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    write_parquet(raw / "news" / "a.parquet", 3)
    seen: list[list[str]] = []
    original = pq.ParquetFile.iter_batches

    def spy(self, *args, **kwargs):
        seen.append(list(kwargs["columns"]))
        return original(self, *args, **kwargs)

    monkeypatch.setattr(pq.ParquetFile, "iter_batches", spy)
    entry = make_entry(id="news", kind="hf_parquet", options={"body": "text", "title": "headline"})
    read_all(open_source(entry), raw)
    assert seen == [["headline", "text"]]


@pytest.mark.req("REQ-COR-001")
def test_missing_raw_dir_says_how_to_fetch(tmp_path):
    entry = make_entry(id="news", kind="hf_parquet", options={"body": "text"})
    with pytest.raises(SourceReadError, match="artigo-env data fetch news"):
        open_source(entry).iter_shards(tmp_path)


# --- jsonl_gz --------------------------------------------------------------------------------


def write_jsonl_gz(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n\n")


@pytest.mark.req("REQ-COR-001")
def test_jsonl_gz_contract(tmp_path):
    raw = tmp_path / "raw"
    write_jsonl_gz(
        raw / "clover" / "2017" / "pt.jsonl.gz",
        [{"t": "Três", "c": "Corpo três", "u": "https://a.pt/3"}],
    )
    write_jsonl_gz(
        raw / "clover" / "2016" / "pt.jsonl.gz",
        [{"t": "Um", "c": "Corpo um", "d": "2016-05-01"}, {"t": "Dois", "c": "Corpo dois"}],
    )
    options = {"title": "t", "body": "c", "url": "u", "date": "d"}
    source = open_source(make_entry(id="clover", kind="jsonl_gz", options=options))
    docs = assert_contract(source, raw)
    assert [(d.doc_id, d.title, d.body) for d in docs] == [
        ("2016__pt:0", "Um", "Corpo um"),
        ("2016__pt:1", "Dois", "Corpo dois"),
        ("2017__pt:0", "Três", "Corpo três"),
    ]
    assert docs[0].date == "2016-05-01" and docs[0].url is None
    assert docs[2].url == "https://a.pt/3"


@pytest.mark.req("REQ-COR-001")
def test_jsonl_gz_rejects_records_without_body(tmp_path):
    raw = tmp_path / "raw"
    write_jsonl_gz(raw / "clover" / "a.jsonl.gz", [{"t": "x"}])
    source = open_source(make_entry(id="clover", kind="jsonl_gz", options={"body": "c"}))
    with pytest.raises(SourceReadError, match="body field 'c' not found"):
        read_all(source, raw)


# --- wikipedia_extracts ----------------------------------------------------------------------


def add_page(snapshot: Path, row: dict[str, object], extract: str | None) -> str:
    pages = snapshot / "pages"
    pages.mkdir(parents=True, exist_ok=True)
    name = f"{len(list(pages.iterdir())):04d}.json"
    page: dict[str, object] = {"pageid": row.get("pageid"), "title": row.get("resolved")}
    if extract is not None:
        page["extract"] = extract
    payload = {
        "requested": row["requested"],
        "fetched_at": "t",
        "response": {"query": {"pages": [page]}},
    }
    (pages / name).write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return name


def write_snapshot(snapshot: Path, unterminated_tail: bool = False) -> None:
    rows = [
        ("Zebra", "Zebra", 30, 7, "ok", "Zebras são equídeos."),
        ("Lua", "Lua", 10, 5, "ok", "A Lua é o satélite natural."),
        ("Página inexistente", "Página inexistente", None, None, "missing", None),
        ("Satélite da Terra", "Lua", 10, 5, "ok", "A Lua é o satélite natural."),
        ("Vazio", "Vazio", 20, 3, "empty", ""),
    ]
    lines = []
    for requested, resolved, pageid, lastrevid, status, extract in rows:
        row: dict[str, object] = {
            "requested": requested,
            "resolved": resolved,
            "pageid": pageid,
            "lastrevid": lastrevid,
            "status": status,
        }
        row["file"] = add_page(snapshot, row, extract)
        lines.append(json.dumps(row, ensure_ascii=False) + "\n")
    text = "".join(lines)
    if unterminated_tail:
        text += '{"requested": "Meio escr'
    (snapshot / "index.jsonl").write_text(text, encoding="utf-8")


EXTRACTS_OPTIONS = {"raw_dir": "wikipedia_extracts", "subdir": "2026-01-01"}


@pytest.mark.req("REQ-COR-001")
@pytest.mark.parametrize("unterminated_tail", [False, True])
def test_extracts_contract(tmp_path, unterminated_tail):
    raw = tmp_path / "raw"
    write_snapshot(raw / "wikipedia_extracts" / "2026-01-01", unterminated_tail)
    entry = make_entry(
        id="w_extracts", kind="wikipedia_extracts", license_id="CC-BY-SA-4.0",
        options=EXTRACTS_OPTIONS,
    )  # fmt: skip
    source = open_source(entry)
    assert [s.shard_id for s in source.iter_shards(raw)] == ["w"]
    docs = assert_contract(source, raw)
    # ordered by page id; missing/empty skipped; the redirect to "Lua" merged into one page
    assert [(d.doc_id, d.title) for d in docs] == [("10", "Lua"), ("30", "Zebra")]
    assert docs[0].provenance == {"requested": "Lua", "lastrevid": "5"}
    assert docs[0].body == "A Lua é o satélite natural."
    assert docs[0].url is None and docs[0].license_id == "CC-BY-SA-4.0"


@pytest.mark.req("REQ-COR-001")
def test_extracts_missing_snapshot(tmp_path):
    entry = make_entry(id="w_extracts", kind="wikipedia_extracts", options=EXTRACTS_OPTIONS)
    with pytest.raises(SourceReadError, match="fetch_extracts"):
        open_source(entry).iter_shards(tmp_path)
    with pytest.raises(SourceReadError, match="subdir"):
        open_source(make_entry(id="w_extracts", kind="wikipedia_extracts"))


# --- demo ------------------------------------------------------------------------------------


@pytest.mark.req("REQ-COR-001")
def test_demo_contract(tmp_path):
    path = tmp_path / "articles.jsonl"
    records = [
        {"title": "Ilha Verde", "body": "A Ilha Verde fica longe.", "license_id": "MIT"},
        {"title": "Rio Torto", "body": "O Rio Torto corre.", "id": "rio"},
    ]
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    entry = load_sources()["demo_synthetic"]
    source = DemoSource(entry, path)
    docs = assert_contract(source, tmp_path / "unused")
    assert [(d.doc_id, d.title, d.license_id) for d in docs] == [
        ("0", "Ilha Verde", "MIT"),
        ("rio", "Rio Torto", "MIT"),
    ]


@pytest.mark.req("REQ-COR-001")
def test_demo_missing_file_gives_clear_error(tmp_path):
    source = DemoSource(load_sources()["demo_synthetic"], tmp_path / "articles.jsonl")
    with pytest.raises(ArtigoEnvError, match="demo articles not found"):
        source.iter_shards(tmp_path)


@pytest.mark.req("REQ-COR-001")
def test_every_registered_kind_is_a_document_source():
    assert SOURCE_KINDS.names() == ("demo", "hf_parquet", "jsonl_gz", "wikipedia_extracts")
