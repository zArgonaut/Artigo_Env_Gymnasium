"""Compiled corpus writer: layout, determinism and token annotations."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pyarrow.ipc as ipc
import pyarrow.parquet as pq
import pytest

from artigo_env.corpus.store import CorpusWriter
from artigo_env.corpus.store.writer import token_case
from artigo_env.corpus.types import DOC_SCHEMA, Row
from artigo_env.engine.types import (
    FLAG_HAS_DIGIT,
    FLAG_HEADER,
    FLAG_INSIDE_PARENTHESES,
    FLAG_PARAGRAPH_BREAK,
)
from artigo_env.text.lemmatizers import get_lemmatizer

DOCS: list[tuple[str | None, str]] = [
    ("Gato", "O gato (animal doméstico) dorme.\n\n== Hábitos ==\nEm 1990 o Gato caçava ratos."),
    ("ONU", "A ONU foi fundada em 1945; McDonald e iPhone. A ONU tem 193 membros."),
    (None, "Texto sem título, com norte-americano e 1º lugar."),
]


def make_row(i: int, title: str | None, body: str) -> Row:
    row: Row = dict.fromkeys(DOC_SCHEMA.names)
    row.update(
        source_id="synthetic",
        doc_id=f"doc-{i}",
        title=title,
        body=body,
        license_id="MIT",
        provenance='{"k":"v"}',
        domain="unknown",
        split="train",
        n_title_content=1,
    )
    return row


def write(out: Path, docs=DOCS, views=("identity", "test-protocol")) -> Path:
    writer = CorpusWriter(out)
    for i, (title, body) in enumerate(docs):
        assert writer.add(make_row(i, title, body)) == i
    for name in views:
        writer.add_lemma_view(get_lemmatizer(name))
    writer.finish({"build_id": "b1", "sources": ["synthetic"]})
    return out


def digests(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


@pytest.mark.fast
@pytest.mark.req("REQ-COR-016")
def test_two_builds_are_byte_identical(tmp_path):
    first = digests(write(tmp_path / "a"))
    second = digests(write(tmp_path / "b"))
    assert first == second
    assert "lemma_views" in "".join(first)
    third = digests(write(tmp_path / "c", docs=list(reversed(DOCS))))
    assert third["body_forms.npy"] != first["body_forms.npy"]


@pytest.mark.fast
@pytest.mark.req("REQ-COR-014")
def test_layout_dtypes_and_shapes(tmp_path):
    out = write(tmp_path / "corpus")
    assert not (tmp_path / "corpus.tmp").exists()
    expected = {
        "body_forms": ("<u4", 1),
        "body_spans": ("<u4", 2),
        "body_flags": ("|u1", 1),
        "body_case": ("|u1", 1),
        "title_forms": ("<u4", 1),
        "title_spans": ("<u4", 2),
        "title_flags": ("|u1", 1),
    }
    offsets = np.load(out / "body_offsets.npy")
    title_offsets = np.load(out / "title_offsets.npy")
    assert offsets.dtype.str == title_offsets.dtype.str == "<u8"
    assert offsets.shape == title_offsets.shape == (len(DOCS) + 1,)
    assert offsets[0] == 0 and title_offsets[0] == 0
    for name, (dtype, columns) in expected.items():
        array = np.load(out / f"{name}.npy")
        assert array.dtype.str == dtype, name
        total = offsets[-1] if name.startswith("body") else title_offsets[-1]
        assert array.shape == ((total, 2) if columns == 2 else (total,)), name
    assert title_offsets.tolist() == [0, 1, 2, 2]  # the row without title has no title tokens

    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["build_id"] == "b1" and manifest["sources"] == ["synthetic"]
    assert manifest["format_version"] == 1
    assert manifest["tokenizer_hyphen"] == "split"
    assert manifest["n_docs"] == 3
    assert manifest["n_body_tokens"] == int(offsets[-1])
    names = {info["name"] for info in manifest["lemma_views"].values()}
    assert names == {"identity", "test-protocol"}
    for key, info in manifest["lemma_views"].items():
        assert info["content_hash"].startswith(key) and len(key) == 16
        form2lemma = np.load(out / "lemma_views" / key / "form2lemma.npy")
        assert form2lemma.dtype.str == "<u4" and form2lemma.shape == (manifest["n_forms"],)
        lemmas = pq.read_table(out / "lemma_views" / key / "lemmas.parquet")
        assert lemmas.column_names == ["lemma_id", "lemma"]
        assert int(form2lemma.max()) == lemmas.num_rows - 1

    texts = ipc.open_file(str(out / "texts.arrow")).read_all()
    assert texts.column_names == ["title", "body"]
    assert texts.column("title").to_pylist() == [t for t, _ in DOCS]
    docs = pq.read_table(out / "docs.parquet")
    assert docs.column_names[0] == "doc_idx"
    assert "title" not in docs.column_names and "body" not in docs.column_names
    assert docs.column("n_body_tokens").to_pylist() == np.diff(offsets).tolist()
    assert docs.column("n_title_content").to_pylist() == [1, 1, 1]


@pytest.mark.fast
@pytest.mark.req("REQ-COR-014")
def test_forms_table(tmp_path):
    out = write(tmp_path / "corpus")
    forms = pq.read_table(out / "forms.parquet")
    assert forms.column_names == ["form_id", "form", "n_chars", "all_digits", "has_digit"]
    table = forms.to_pylist()
    assert [r["form_id"] for r in table] == list(range(len(table)))
    words = [r["form"] for r in table]
    assert words[:2] == ["gato", "o"]  # first appearance: title, then body
    assert len(set(words)) == len(words)
    assert all(w == w.lower() for w in words)
    by_form = {r["form"]: r for r in table}
    assert by_form["1990"]["all_digits"] and by_form["1990"]["has_digit"]
    assert by_form["1º"]["has_digit"] and not by_form["1º"]["all_digits"]
    assert by_form["doméstico"]["n_chars"] == 9


@pytest.mark.fast
@pytest.mark.req("REQ-COR-014")
def test_flags_spans_and_case(tmp_path):
    out = write(tmp_path / "corpus")
    end = int(np.load(out / "body_offsets.npy")[1])
    spans = np.load(out / "body_spans.npy")[:end]
    flags = np.load(out / "body_flags.npy")[:end]
    case = np.load(out / "body_case.npy")
    body = DOCS[0][1]
    surfaces = [body[s:e] for s, e in spans.tolist()]
    assert surfaces[:5] == ["O", "gato", "animal", "doméstico", "dorme"]
    at = surfaces.index
    assert flags[at("animal")] & FLAG_INSIDE_PARENTHESES
    assert not flags[at("dorme")] & FLAG_INSIDE_PARENTHESES
    assert flags[at("Hábitos")] & FLAG_HEADER and flags[at("Hábitos")] & FLAG_PARAGRAPH_BREAK
    assert flags[at("Em")] & FLAG_PARAGRAPH_BREAK and not flags[at("Em")] & FLAG_HEADER
    assert flags[at("1990")] == FLAG_HAS_DIGIT
    assert case[at("O")] == 1 and case[at("gato")] == 0 and case[at("Gato")] == 1
    assert case[at("1990")] == 0
    second = np.load(out / "body_spans.npy")[end:]
    body2 = DOCS[1][1]
    case2 = case[end:]
    words2 = [body2[s:e] for s, e in second[: len(case2)].tolist()]
    got = dict(zip(words2, case2.tolist(), strict=False))
    assert got["ONU"] == 2 and got["McDonald"] == 3 and got["iPhone"] == 3 and got["A"] == 1


@pytest.mark.fast
@pytest.mark.req("REQ-COR-014")
@pytest.mark.parametrize(
    ("text", "expected"),
    [("gato", 0), ("1990", 0), ("Gato", 1), ("A", 1), ("ÉPOCA", 2), ("1A", 2), ("McDonald", 3)],
)
def test_token_case(text, expected):
    assert token_case(text) == expected


@pytest.mark.fast
@pytest.mark.req("REQ-COR-014")
def test_writer_guards(tmp_path):
    out = write(tmp_path / "corpus")
    with pytest.raises(FileExistsError):
        CorpusWriter(out)
    writer = CorpusWriter(tmp_path / "other")
    writer.add(make_row(0, "Gato", "O gato dorme."))
    with pytest.raises(ValueError, match="build_id"):
        writer.finish({})
    writer.finish({"build_id": "x"})
    with pytest.raises(RuntimeError):
        writer.add(make_row(1, "Gato", "O gato dorme."))
    with pytest.raises(ValueError, match="hyphen"):
        CorpusWriter(tmp_path / "bad", tokenizer_hyphen="keep")


@pytest.mark.fast
@pytest.mark.req("REQ-COR-014")
def test_abort_on_error_leaves_nothing(tmp_path):
    with pytest.raises(KeyError), CorpusWriter(tmp_path / "corpus") as writer:
        writer.add({"title": "Gato"})  # no body
    assert not (tmp_path / "corpus").exists() and not (tmp_path / "corpus.tmp").exists()


def synthetic_body(rng: np.random.Generator, n_words: int) -> str:
    vocabulary = [f"palavra{i}" for i in range(5000)] + ["o", "a", "de", "que", "Brasil", "1990"]
    words = rng.choice(vocabulary, size=n_words)
    lines = [" ".join(words[i : i + 20]) + "." for i in range(0, n_words, 20)]
    return "\n".join(lines[:1] + ["== Seção ==", *lines[1:]])


@pytest.mark.perf
@pytest.mark.req("REQ-COR-014")
def test_perf_write_2000_documents(tmp_path):
    rng = np.random.default_rng(20261112)
    bodies = [synthetic_body(rng, int(rng.integers(300, 1500))) for _ in range(2000)]
    start = time.perf_counter()
    writer = CorpusWriter(tmp_path / "corpus")
    for i, body in enumerate(bodies):
        writer.add(make_row(i, f"Palavra{i} do Brasil", body))
    writer.add_lemma_view(get_lemmatizer("identity"))
    writer.finish({"build_id": "perf"})
    elapsed = time.perf_counter() - start
    n_tokens = json.loads((tmp_path / "corpus" / "manifest.json").read_text())["n_body_tokens"]
    print(f"\nwrite 2000 docs ({n_tokens} body tokens): {elapsed:.2f} s")
    assert elapsed < 300
