"""Corpus build configs, build kinds and build identity; synthetic data helpers for build tests.

The helpers at the top (``news_rows``, ``write_shard``, ``write_snapshot``, ``synthetic_sources``,
``config``) are imported by the other corpus build tests.
"""

from __future__ import annotations

import json
import random
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from artigo_env.corpus.manifest import SourceNotAcceptedError, parse_sources
from artigo_env.corpus.pipeline import (
    BuildError,
    CorpusConfig,
    build,
    load_config,
)
from artigo_env.corpus.store import CorpusStore
from artigo_env.corpus.types import PlayabilityConfig
from artigo_env.text.stopwords import StopwordSpec, load_stopwords

REPO = Path(__file__).resolve().parents[3]

# --- synthetic data ----------------------------------------------------------------------------

_STOP = load_stopwords(StopwordSpec())
_SYL = [c + v for c in "bdfgklmnprstvz" for v in "aeiou"]


def vocabulary(seed: int = 7, size: int = 3000) -> list[str]:
    rng = random.Random(seed)
    words: set[str] = set()
    while len(words) < size:
        word = "".join(rng.choice(_SYL) for _ in range(rng.choice((2, 3, 3, 4))))
        if word not in _STOP:
            words.add(word)
    return sorted(words)


VOCAB = vocabulary()


def article(rng: random.Random, n_words: int = 220, n_title: int = 2) -> tuple[str, str]:
    """A title of ``n_title`` pseudo words and a body that contains them."""
    title_words = rng.sample(VOCAB, n_title)
    words = [rng.choice(VOCAB) for _ in range(n_words)]
    for i, word in enumerate(title_words):
        words[5 + 7 * i] = word
    sentences = [" ".join(words[i : i + 12]).capitalize() + "." for i in range(0, n_words, 12)]
    paragraphs = [" ".join(sentences[i : i + 5]) for i in range(0, len(sentences), 5)]
    return " ".join(w.capitalize() for w in title_words), "\n\n".join(paragraphs)


def news_rows(seed: int, n: int, host: str = "noticias.com.br") -> list[dict[str, Any]]:
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        title, body = article(rng)
        rows.append(
            {
                "title": title,
                "text": body,
                "url": f"https://www.{host}/{seed}/{i}",
                "date": f"2020-01-{1 + i % 28:02d}",
            }
        )
    return rows


def write_shard(root: Path, raw_dir: str, name: str, rows: Sequence[Mapping[str, Any]]) -> Path:
    """``root/raw/<raw_dir>/train/<name>.parquet`` (shard id ``train__<name>``)."""
    path = root / "raw" / raw_dir / "train" / f"{name}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = {key: [row[key] for row in rows] for key in ("title", "text", "url", "date")}
    pq.write_table(pa.table(columns), path)
    return path


def write_snapshot(
    root: Path,
    pages: Sequence[tuple[str, str, int, str]],
    missing: Sequence[str] = (),
    raw_dir: str = "wiki",
    subdir: str = "snap",
) -> Path:
    """A TextExtracts snapshot: ``pages`` are ``(requested, resolved, pageid, text)``."""
    directory = root / "raw" / raw_dir / subdir
    (directory / "pages").mkdir(parents=True, exist_ok=True)
    lines = []
    for n, (requested, resolved, pageid, text) in enumerate(pages):
        name = f"p{n:05d}.json"
        response = {"query": {"pages": [{"pageid": pageid, "title": resolved, "extract": text}]}}
        payload = {"requested": requested, "response": response}
        (directory / "pages" / name).write_text(json.dumps(payload), encoding="utf-8")
        row = {"requested": requested, "resolved": resolved, "pageid": pageid}
        row.update(lastrevid=1000 + n, status="ok", file=name)
        lines.append(json.dumps(row))
    for n, requested in enumerate(missing):
        row = {"requested": requested, "resolved": requested, "pageid": None, "lastrevid": None}
        row.update(status="missing", file=f"m{n}.json")
        lines.append(json.dumps(row))
    (directory / "index.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return directory


def synthetic_sources(news_state: str = "accepted", blog_state: str = "accepted"):
    def tabular(source_id: str, state: str, priority: int) -> dict[str, Any]:
        return {
            "id": source_id,
            "kind": "hf_parquet",
            "state": state,
            "priority": priority,
            "license_id": "unknown",
            "options": {
                "subdir": "train",
                "title": "title",
                "body": "text",
                "url": "url",
                "date": "date",
            },
        }

    wiki = {
        "id": "wiki",
        "kind": "wikipedia_extracts",
        "state": "evaluation",
        "priority": 100,
        "license_id": "CC-BY-SA-4.0",
        "options": {"raw_dir": "wiki", "subdir": "snap"},
    }
    data = {
        "version": 1,
        "sources": [tabular("news", news_state, 10), tabular("blog", blog_state, 20), wiki],
    }
    return parse_sources(data)


def config(**fields: Any) -> CorpusConfig:
    base: dict[str, Any] = {
        "name": "synthetic",
        "kind": "measurement",
        "sources": ("news",),
        "lemmatizer": "identity",
        "lemma_views": ("identity",),
        "workers": 1,
    }
    base.update(fields)
    return CorpusConfig(**base)


def file_bytes(directory: Path) -> dict[str, bytes]:
    """Every file of a compiled corpus except ``report.md``, by relative path."""
    return {
        p.relative_to(directory).as_posix(): p.read_bytes()
        for p in sorted(directory.rglob("*"))
        if p.is_file() and p.name != "report.md"
    }


# --- tests ---------------------------------------------------------------------------------------


@pytest.fixture
def news_root(tmp_path: Path) -> Path:
    root = tmp_path / "data"
    write_shard(root, "news", "0001", news_rows(1, 12))
    write_shard(root, "news", "0002", news_rows(2, 12))
    return root


@pytest.mark.fast
@pytest.mark.req("REQ-COR-006")
@pytest.mark.parametrize("name", ["ccnews-v1", "ccnews-measure", "w-v1", "demo"])
def test_shipped_configs_load(name: str) -> None:
    cfg = load_config(REPO / "configs" / "corpus" / f"{name}.yaml")
    assert cfg.name == name
    assert (
        cfg.kind
        == {
            "ccnews-v1": "train",
            "ccnews-measure": "measurement",
            "w-v1": "evaluation",
            "demo": "demo",
        }[name]
    )


@pytest.mark.fast
@pytest.mark.req("REQ-COR-006")
@pytest.mark.parametrize(
    "fields",
    [
        {"kind": "other"},
        {"sources": ()},
        {"sources": ("news", "news")},
        {"shards": {"blog": ("x",)}},
        {"workers": 0},
        {"kind": "evaluation", "w_index": "abc"},
        {"w_splits": "x.tsv"},
        {"tokenizer_hyphen": "glue"},
        {"lemma_views": ()},
    ],
)
def test_invalid_configs_are_refused(fields: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        config(**fields)


@pytest.mark.fast
@pytest.mark.req("REQ-COR-006")
def test_config_from_yaml_coerces_shards(tmp_path: Path) -> None:
    path = tmp_path / "c.yaml"
    path.write_text(
        "name: x\nkind: measurement\nsources: [news]\nshards: {news: [train__0002, train__0001]}\n"
        "playability: {min_body_tokens: 100}\nlemmatizer: identity\nlemma_views: [identity]\n",
        encoding="utf-8",
    )
    cfg = load_config(path)
    assert cfg.shards == {"news": ("train__0002", "train__0001")}
    assert cfg.playability == PlayabilityConfig(min_body_tokens=100)
    path.write_text("name: x\nkind: measurement\nsources: [news]\nbogus: 1\n", encoding="utf-8")
    with pytest.raises(BuildError):
        load_config(path)


@pytest.mark.fast
@pytest.mark.req("REQ-COR-002")
def test_pending_source_is_refused_for_train(news_root: Path) -> None:
    sources = synthetic_sources(news_state="pending")
    with pytest.raises(SourceNotAcceptedError):
        build(config(kind="train", w_index="0" * 16), data_root=news_root, sources=sources)
    assert not (news_root / "corpus").exists()


@pytest.mark.fast
@pytest.mark.req("REQ-COR-002")
def test_pending_source_is_allowed_for_measurement(news_root: Path) -> None:
    sources = synthetic_sources(news_state="pending")
    result = build(config(), data_root=news_root, sources=sources)
    assert result.n_docs == 24
    with CorpusStore.open(result.out_dir) as store:
        assert store.manifest["trainable"] is False
        assert store.manifest["kind"] == "measurement"
    assert result.report["decontam"]["status"] == "skipped"


@pytest.mark.fast
@pytest.mark.req("REQ-COR-002")
def test_train_build_needs_a_w_index(news_root: Path) -> None:
    with pytest.raises(BuildError, match="W index"):
        build(config(kind="train"), data_root=news_root, sources=synthetic_sources())
    with pytest.raises(BuildError, match="no W index"):
        build(
            config(kind="train", w_index="f" * 16), data_root=news_root, sources=synthetic_sources()
        )


@pytest.mark.fast
@pytest.mark.req("REQ-COR-002")
def test_evaluation_build_refuses_a_training_source(news_root: Path) -> None:
    with pytest.raises(BuildError, match="evaluation"):
        build(config(kind="evaluation"), data_root=news_root, sources=synthetic_sources())


@pytest.mark.fast
@pytest.mark.req("REQ-COR-016", "REQ-COR-006")
def test_build_id_ignores_workers_and_tracks_settings(news_root: Path) -> None:
    sources = synthetic_sources()
    first = build(config(), data_root=news_root, sources=sources)
    again = build(config(workers=2), data_root=news_root, sources=sources)
    assert again.build_id == first.build_id and again.reused
    other = build(
        config(playability=PlayabilityConfig(min_body_tokens=100)),
        data_root=news_root,
        sources=sources,
    )
    assert other.build_id != first.build_id
    with CorpusStore.open(other.out_dir) as store:
        assert store.manifest["playability"]["min_body_tokens"] == 100
        assert store.manifest["config"]["playability"]["min_body_tokens"] == 100
        assert "workers" not in store.manifest["config"]
    subset = build(config(shards={"news": ("train__0001",)}), data_root=news_root, sources=sources)
    assert subset.build_id != first.build_id
    assert subset.n_docs == 12


@pytest.mark.fast
@pytest.mark.req("REQ-COR-016")
def test_unknown_shard_is_an_error(news_root: Path) -> None:
    with pytest.raises(BuildError, match="no shard"):
        build(
            config(shards={"news": ("train__9999",)}),
            data_root=news_root,
            sources=synthetic_sources(),
        )
