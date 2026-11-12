"""Compiled corpus reader: memory mapping, document access, lemma views."""

from __future__ import annotations

import shutil
import time
from pathlib import Path

import numpy as np
import pytest

from artigo_env.core.errors import ArtigoEnvError
from artigo_env.corpus.store import CorpusStore, CorpusWriter, MissingLemmaViewError
from artigo_env.corpus.types import DOC_SCHEMA, Row
from artigo_env.engine.rulesets import Ruleset
from artigo_env.text.lemmatizers import get_lemmatizer

DOCS: list[tuple[str | None, str, str]] = [
    ("Gato", "O gato (animal) dorme.\n\n== Hábitos ==\nEm 1990 o gato caçava.", "train"),
    (None, "Texto sem título e sem cabeçalho.", "ext-val"),
    ("Cão", "O cão late; o Cão-guia ajuda (muito) desde 1950.", "train"),
]


def make_row(i: int, title: str | None, body: str, split: str) -> Row:
    row: Row = dict.fromkeys(DOC_SCHEMA.names)
    row.update(
        source_id="synthetic",
        doc_id=f"doc-{i}",
        title=title,
        body=body,
        url=f"https://example.com/{i}",
        license_id="MIT",
        provenance="{}",
        variant="pt-BR",
        domain="unknown",
        split=split,
        title_in_body=True,
    )
    return row


def build(out: Path, docs=DOCS, views=("identity",)) -> Path:
    writer = CorpusWriter(out)
    for i, (title, body, split) in enumerate(docs):
        writer.add(make_row(i, title, body, split))
    for name in views:
        writer.add_lemma_view(get_lemmatizer(name))
    writer.finish({"build_id": "reader-build"})
    return out


@pytest.fixture
def store(tmp_path):
    opened = CorpusStore.open(build(tmp_path / "corpus"))
    yield opened
    opened.close()


@pytest.mark.fast
@pytest.mark.req("REQ-COR-014")
def test_arrays_are_memory_mapped_and_store_can_be_deleted(tmp_path):
    path = build(tmp_path / "corpus")
    store = CorpusStore.open(path)
    # No loop variable may keep a map alive past close().
    assert all(isinstance(a, np.memmap) and a.mode == "r" for a in store._arrays.values())
    assert isinstance(store.lemma_view("identity").form2lemma, np.memmap)
    assert store.body_text(0).startswith("O gato")
    store.article(0, Ruleset(name="r", lemmatizer="identity"))
    store.close()
    with pytest.raises(ArtigoEnvError, match="closed"):
        store.body_forms(0)
    shutil.rmtree(path)  # nothing is left mapped (Windows locks mapped files)
    assert not path.exists()


@pytest.mark.fast
@pytest.mark.req("REQ-COR-014")
def test_document_access(store):
    assert store.n_docs == 3
    assert store.manifest["build_id"] == "reader-build"
    assert store.docs.num_rows == 3
    row = store.doc(0)
    assert list(row) == DOC_SCHEMA.names
    assert row["title"] == "Gato" and row["body"] == DOCS[0][1]
    assert row["url"] == "https://example.com/0" and row["variant"] == "pt-BR"
    assert row["n_body_tokens"] == store.n_body_tokens(0) == 10
    assert store.doc(1)["title"] is None
    assert store.title_text(1) == ""
    assert store.n_title_tokens(1) == 0
    forms = store.body_forms(0)
    assert forms.dtype == np.uint32
    assert [store.form(int(f)) for f in forms[:3]] == ["o", "gato", "animal"]
    assert store.form(int(store.title_forms(2)[0])) == "cão"
    assert store.form_id("gato") == int(store.title_forms(0)[0])
    assert store.form_id("inexistente") == -1
    assert store.split_indices("train").tolist() == [0, 2]
    assert store.split_indices("ext-val").tolist() == [1]
    assert store.split_indices("w-test").tolist() == []
    with pytest.raises(IndexError):
        store.doc(3)


@pytest.mark.fast
@pytest.mark.req("REQ-COR-014")
def test_body_case(store):
    words = [
        store.body_text(2)[s:e]
        for s, e in store._arrays["body_spans"][
            store._arrays["body_offsets"][2] : store._arrays["body_offsets"][3]
        ].tolist()
    ]
    case = dict(zip(words, store.body_case(2).tolist(), strict=True))
    assert case["O"] == 1 and case["cão"] == 0 and case["Cão"] == 1 and case["1950"] == 0


@pytest.mark.fast
@pytest.mark.req("REQ-COR-014")
def test_missing_lemma_view_names_available_views(store, protocol_ruleset):
    with pytest.raises(MissingLemmaViewError, match=r"test-protocol.*available: identity \("):
        store.article(0, protocol_ruleset)
    with pytest.raises(MissingLemmaViewError, match="identity"):
        store.lemma_view(get_lemmatizer("test-protocol"))
    view = store.lemma_view(get_lemmatizer("identity"))
    assert view is store.lemma_view("identity")
    assert view.name == "identity"
    gato = view.lemma_id("gato")
    assert gato >= 0 and view.lemma_text(gato) == "gato"
    assert view.lemma_id("cachorro") == -1
    with pytest.raises(KeyError):
        view.lemma_text(len(view))


@pytest.mark.fast
@pytest.mark.req("REQ-COR-014")
def test_stop_mask_is_cached_per_stopword_set(store):
    from artigo_env.text.stopwords import StopwordSpec

    mask = store.stop_mask(StopwordSpec())
    assert mask is store.stop_mask(StopwordSpec())
    assert mask[store.form_id("o")] and not mask[store.form_id("gato")]
    assert not store.stop_mask(StopwordSpec(kind="none")).any()


@pytest.mark.perf
@pytest.mark.req("REQ-COR-014")
def test_perf_article_3000_tokens(tmp_path):
    rng = np.random.default_rng(7)
    vocabulary = [f"termo{i}" for i in range(2000)] + ["o", "a", "de", "Brasil", "(nota)", "1990"]
    words = rng.choice(vocabulary, size=3000).tolist()
    lines = [" ".join(words[i : i + 25]) for i in range(0, 3000, 25)]
    body = "\n".join(lines[:4] + ["== Seção ==", *lines[4:]])
    docs = [("Brasil", body, "train")] * 3
    store = CorpusStore.open(build(tmp_path / "corpus", docs=docs))
    try:
        ruleset = Ruleset(name="perf", lemmatizer="identity")
        for _ in range(20):
            store.article(1, ruleset)
        assert store.article(1, ruleset).n_total >= 3000
        times = []
        for _ in range(200):
            start = time.perf_counter()
            store.article(1, ruleset)
            times.append(time.perf_counter() - start)
        median_ms = 1000 * float(np.median(times))
        print(f"\narticle() 3000 tokens: median {median_ms:.3f} ms, min {1000 * min(times):.3f} ms")
        assert median_ms < 50  # informative target: < 2 ms
    finally:
        store.close()
