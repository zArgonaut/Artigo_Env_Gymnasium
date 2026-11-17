"""Construction of the minimal knowledge store (SDD 10 §4, F4 plan WP4.4)."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pytest

from artigo_env.cli import knowledge as knowledge_cli
from artigo_env.core.vocab import Vocabulary
from artigo_env.corpus.store import CorpusStore, CorpusWriter
from artigo_env.corpus.types import DOC_SCHEMA, Row
from artigo_env.corpus.vocab import VocabConfig, build_vocab
from artigo_env.engine.rulesets import Ruleset, get_ruleset
from artigo_env.services.knowledge import KnowledgeStore
from artigo_env.services.knowledge_build import WikimediaSourceError, build_knowledge
from artigo_env.text.lemmatizers import get_lemmatizer

pytestmark = pytest.mark.fast

# Training documents (title, body). Counts over title + body of the 10 "train" documents:
#   key      freq  doc_freq  title_doc_freq  mean_count_present
#   gato       7       4           1              1.75
#   casa       6       4           1              1.5
#   sol        5       3           1              5/3
#   rio        4       3           1              4/3
#   lua        4       3           1              4/3
#   n:1990     3       2           0              1.5
# "o" and "de" are stopwords; the two "w-val" documents never count ("zebra", "gelo" not in V).
DOCS: list[tuple[str | None, str, str]] = [
    ("Gato", "gato casa sol", "train"),
    ("Casa", "casa gato casa", "train"),
    ("Sol", "o sol de rio", "train"),
    (None, "gato gato gato 1990", "train"),
    (None, "rio lua", "train"),
    ("Lua", "lua casa", "train"),
    (None, "casa", "train"),
    (None, "sol sol 1990 1990", "train"),
    ("Rio", "rio", "train"),
    (None, "gato lua", "train"),
    ("Zebra", "zebra gato gato", "w-val"),
    ("Gelo", "gelo casa", "w-val"),
]
EXPECTED = {
    "gato": (7, 4, 1),
    "casa": (6, 4, 1),
    "sol": (5, 3, 1),
    "rio": (4, 3, 1),
    "lua": (4, 3, 1),
    "n:1990": (3, 2, 0),
}
CONFIG = VocabConfig(numeric_max=10, min_doc_freq=1, numeric_min_doc_freq=1)
RULESET = Ruleset(name="test-identity", lemmatizer="identity")


def run_cli(argv: list[str], stdout: io.StringIO) -> int:
    """``artigo-env <argv>`` with only the ``knowledge`` subcommand registered."""
    parser = argparse.ArgumentParser(prog="artigo-env")
    knowledge_cli.add_parser(parser.add_subparsers(dest="command", required=True))
    args = parser.parse_args(argv)
    return int(args.handler(args, io.StringIO(), stdout))


def write_corpus(out: Path, build_id: str, sources=None, views=("identity",)) -> Path:
    writer = CorpusWriter(out)
    for i, (title, body, split) in enumerate(DOCS):
        row: Row = dict.fromkeys(DOC_SCHEMA.names)
        row.update(
            source_id="synthetic",
            doc_id=f"doc-{i}",
            title=title,
            body=body,
            license_id="MIT",
            provenance="{}",
            domain="unknown",
            split=split,
        )
        writer.add(row)
    for name in views:
        writer.add_lemma_view(get_lemmatizer(name))
    manifest = {"build_id": build_id}
    if sources is not None:
        manifest["sources"] = sources
    writer.finish(manifest)
    return out


@pytest.fixture
def corpus(tmp_path):
    store = CorpusStore.open(write_corpus(tmp_path / "corpus", "synthetic-build"))
    vocab = Vocabulary.open(build_vocab(store, RULESET, CONFIG, tmp_path / "vocab"))
    yield store, vocab
    vocab.close()
    store.close()


def digest(path: Path) -> dict[str, str]:
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(path.iterdir())}


@pytest.mark.req("REQ-KNW-002")
def test_counts_checked_by_hand(corpus, tmp_path):
    store, vocab = corpus
    knowledge = KnowledgeStore.open(
        build_knowledge(store, RULESET, vocab, tmp_path / "knowledge"), vocab=vocab
    )
    assert knowledge.n_docs == 10 and knowledge.manifest["partition"] == "train"
    lemma, title = knowledge.lemma_stats, knowledge.title_stats["title_doc_freq"]
    for key, (freq, doc_freq, title_doc_freq) in EXPECTED.items():
        i = vocab.action_id(key)
        assert i >= 0, key
        assert lemma["freq"][i] == freq, key
        assert lemma["doc_freq"][i] == doc_freq, key
        assert title[i] == title_doc_freq, key
        assert lemma["mean_count_present"][i] == pytest.approx(freq / doc_freq), key
    assert vocab.action_id("zebra") == -1 and vocab.action_id("o") == -1
    unseen = vocab.action_id("n:7")
    assert lemma["freq"][unseen] == 0 and lemma["mean_count_present"][unseen] == 0
    assert lemma["freq"].dtype == np.float32 and title.dtype == np.int64
    assert lemma["freq"].shape == (vocab.size,)


@pytest.mark.req("REQ-KNW-002")
def test_deterministic_bytes_and_id(corpus, tmp_path):
    store, vocab = corpus
    a = build_knowledge(store, RULESET, vocab, tmp_path / "a")
    b = build_knowledge(store, RULESET, vocab, tmp_path / "b")
    assert a.name == b.name
    assert digest(a) == digest(b)
    manifest = json.loads((a / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["knowledge_id"] == a.name
    assert manifest["vocab_id"] == vocab.vocab_id and manifest["build_id"] == "synthetic-build"
    assert manifest["source_config"] == "X->W" and manifest["regime"] == "open"
    for name, sha in manifest["hashes"].items():
        assert hashlib.sha256((a / name).read_bytes()).hexdigest() == sha
    assert build_knowledge(store, RULESET, vocab, tmp_path / "a") == a  # existing: unchanged
    other = build_knowledge(store, RULESET, vocab, tmp_path / "c", source_config="W->W")
    assert other.name != a.name


@pytest.mark.req("REQ-KNW-001")
def test_wikimedia_source_refused_under_x_to_w(tmp_path):
    sources = [
        {"id": "news", "kind": "hf_parquet"},
        {"id": "ptwiki", "kind": "wikipedia_extracts"},
    ]
    path = write_corpus(tmp_path / "wiki", "wiki-build", sources)
    with CorpusStore.open(path) as store:
        vocab = Vocabulary.open(build_vocab(store, RULESET, CONFIG, tmp_path / "vocab"))
        try:
            with pytest.raises(WikimediaSourceError, match="ptwiki"):
                build_knowledge(store, RULESET, vocab, tmp_path / "knowledge")
            assert not (tmp_path / "knowledge").exists()
            built = build_knowledge(
                store, RULESET, vocab, tmp_path / "knowledge", source_config="W->W"
            )
            assert KnowledgeStore.open(built).source_config == "W->W"
        finally:
            vocab.close()


@pytest.mark.req("REQ-KNW-001")
def test_explicit_wikimedia_marker_refused(tmp_path):
    path = write_corpus(
        tmp_path / "w", "marked", [{"id": "x", "kind": "jsonl_gz", "wikimedia": True}]
    )
    with CorpusStore.open(path) as store:
        vocab = Vocabulary.open(build_vocab(store, RULESET, CONFIG, tmp_path / "vocab"))
        try:
            with pytest.raises(WikimediaSourceError):
                build_knowledge(store, RULESET, vocab, tmp_path / "knowledge")
        finally:
            vocab.close()


@pytest.mark.req("REQ-KNW-002")
def test_cli_build(tmp_path):
    corpus = write_corpus(tmp_path / "corpus", "cli-build", views=("open-tables",))
    with CorpusStore.open(corpus) as store:
        vocab_dir = build_vocab(store, get_ruleset("canonical"), VocabConfig(), tmp_path / "v")
    out = io.StringIO()
    code = run_cli(
        [
            "knowledge",
            "build",
            "--corpus",
            str(corpus),
            "--ruleset",
            "canonical",
            "--vocab",
            str(vocab_dir),
            "--out-root",
            str(tmp_path / "knowledge"),
        ],
        stdout=out,
    )
    assert code == 0, out.getvalue()
    assert "knowledge " in out.getvalue() and "10 training docs" in out.getvalue()
    out = io.StringIO()
    code = run_cli(
        ["knowledge", "build", "--corpus", str(corpus), "--ruleset", "no-lengths"]
        + ["--vocab", str(vocab_dir), "--out-root", str(tmp_path / "k2")],
        stdout=out,
    )
    assert code == 2 and "error:" in out.getvalue()
