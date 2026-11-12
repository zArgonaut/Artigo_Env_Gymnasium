"""Action vocabulary construction, build mapping and title coverage (SDD 09 §2, WP4.2)."""

from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

from artigo_env.core.errors import ArtigoEnvError, IncompatibleRulesetError
from artigo_env.core.vocab import Vocabulary
from artigo_env.corpus.store import CorpusStore, CorpusWriter
from artigo_env.corpus.types import DOC_SCHEMA, Row
from artigo_env.corpus.vocab import VocabConfig, build_vocab, map_store, title_coverage
from artigo_env.engine.rulesets import Ruleset
from artigo_env.text.lemmatizers import get_lemmatizer

pytestmark = pytest.mark.fast

# (title, body, split). Token frequencies in "train" count title and body tokens:
# gato 5 (df 3), casa 5 (df 4), sol 3 (df 2), rio 3 (df 2), lua 2 (df 2), gelo 2 (df 1);
# 1990 df 5, 2500 df 5, 777777 df 5 (6 digits), 1234567 df 5 (7 digits), 07 df 1.
TRAIN_DOCS: list[tuple[str | None, str, str]] = [
    ("Gato", "gato casa sol 1990 gato 1234567 gelo gelo", "train"),
    ("Casa", "casa gato rio 1990 1234567", "train"),
    ("Rio", "rio casa lua 777777 1234567", "train"),
    ("Sol", "sol gato o de casa 07 777777", "train"),
    (None, "lua 1990 2500", "train"),
    (None, "2500 1990", "train"),
    (None, "2500 1990", "train"),
    (None, "2500 777777", "train"),
    (None, "2500 777777 1234567", "train"),
    (None, "777777 1234567 1234567", "train"),
]
EVAL_DOCS: list[tuple[str | None, str, str]] = [
    ("Zebra", "zebra gato zebra zebra", "w-val"),
    ("Gato preto", "gato preto", "w-val"),
]
CONFIG = VocabConfig(numeric_max=10)
NUMERIC_FORCED = [f"n:{d}" for d in sorted(str(i) for i in range(11))]
EXPECTED_KEYS = [
    "casa",
    "gato",
    "n:1990",
    "n:2500",
    "n:777777",
    "rio",
    "sol",
    "lua",
    *NUMERIC_FORCED,
]


def make_row(i: int, title: str | None, body: str, split: str) -> Row:
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
    return row


def build_corpus(out: Path, docs, build_id: str, views=("identity",)) -> Path:
    writer = CorpusWriter(out)
    for i, (title, body, split) in enumerate(docs):
        writer.add(make_row(i, title, body, split))
    for name in views:
        writer.add_lemma_view(get_lemmatizer(name))
    writer.finish({"build_id": build_id})
    return out


def identity_ruleset(**changes) -> Ruleset:
    return Ruleset(name="test-identity", lemmatizer="identity", **changes)


def dir_digest(path: Path, skip: tuple[str, ...] = ()) -> dict[str, str]:
    return {
        p.relative_to(path).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(path.rglob("*"))
        if p.is_file() and not any(p.relative_to(path).as_posix().startswith(s) for s in skip)
    }


@pytest.fixture
def train_store(tmp_path):
    store = CorpusStore.open(
        build_corpus(tmp_path / "train", TRAIN_DOCS + EVAL_DOCS, "train-build")
    )
    yield store
    store.close()


def build(store, tmp_path, ruleset=None, config=CONFIG, lexicon=(), name="vocab") -> Vocabulary:
    path = build_vocab(
        store, ruleset or identity_ruleset(), config, tmp_path / name, lexicon_forms=lexicon
    )
    return Vocabulary.open(path)


# -- identity and determinism -------------------------------------------------------------------


@pytest.mark.req("REQ-ENV-004")
def test_same_inputs_give_identical_bytes_and_id(train_store, tmp_path):
    a = build_vocab(train_store, identity_ruleset(), CONFIG, tmp_path / "a", lexicon_forms=["x"])
    b = build_vocab(train_store, identity_ruleset(), CONFIG, tmp_path / "b", lexicon_forms=["x"])
    assert a.name == b.name and len(a.name) == 16
    assert dir_digest(a) == dir_digest(b)
    assert sorted(dir_digest(a)) == [
        "doc_freq.npy",
        "forms.parquet",
        "freq.npy",
        "is_numeric.npy",
        "lemmas.parquet",
        "length_mask.npy",
        "manifest.json",
        "stop.parquet",
    ]
    # Building again into the same root returns the existing artifact untouched.
    assert (
        build_vocab(train_store, identity_ruleset(), CONFIG, tmp_path / "a", lexicon_forms=["x"])
        == a
    )


@pytest.mark.req("REQ-ENV-004")
def test_ruleset_config_or_lexicon_change_the_id(train_store, tmp_path):
    base = build_vocab(train_store, identity_ruleset(), CONFIG, tmp_path / "v").name
    other_ruleset = build_vocab(
        train_store, identity_ruleset(lengths_visible="none"), CONFIG, tmp_path / "v"
    ).name
    other_config = build_vocab(
        train_store, identity_ruleset(), replace(CONFIG, max_size=500), tmp_path / "v"
    ).name
    other_lexicon = build_vocab(
        train_store, identity_ruleset(), CONFIG, tmp_path / "v", lexicon_forms=["árvore"]
    ).name
    assert len({base, other_ruleset, other_config, other_lexicon}) == 4
    with Vocabulary.open(tmp_path / "v" / base) as vocab:
        assert vocab.manifest["build_id"] == "train-build"
        assert vocab.manifest["format_version"] == 1
        assert vocab.manifest["ruleset_hash"] == identity_ruleset().content_hash()


@pytest.mark.req("REQ-ENV-003")
def test_vocabulary_ignores_evaluation_documents(tmp_path):
    other_eval = [("Ornitorrinco", "ornitorrinco ornitorrinco 1990 casa", "w-val")] * 3
    with (
        CorpusStore.open(build_corpus(tmp_path / "b1", TRAIN_DOCS + EVAL_DOCS, "same")) as s1,
        CorpusStore.open(build_corpus(tmp_path / "b2", other_eval + TRAIN_DOCS, "same")) as s2,
    ):
        p1 = build_vocab(s1, identity_ruleset(), CONFIG, tmp_path / "v1")
        p2 = build_vocab(s2, identity_ruleset(), CONFIG, tmp_path / "v2")
    assert p1.name == p2.name
    assert dir_digest(p1) == dir_digest(p2)


@pytest.mark.req("REQ-ENV-003")
def test_mapping_an_evaluation_build_changes_only_the_map(train_store, tmp_path):
    vocab = build(train_store, tmp_path)
    before = dir_digest(vocab.path)
    for i, docs in enumerate((EVAL_DOCS, [("Lua", "lua zebra", "w-val")])):
        with CorpusStore.open(build_corpus(tmp_path / f"w{i}", docs, f"w-build-{i}")) as w:
            mapping = map_store(vocab, w, identity_ruleset())
            assert mapping.build_id == f"w-build-{i}"
            assert (vocab.path / "maps" / f"w-build-{i}" / "lemma_obs.npy").is_file()
    assert dir_digest(vocab.path, skip=("maps/",)) == before
    vocab.close()


# -- construction rules -------------------------------------------------------------------------


@pytest.mark.req("REQ-ENV-004")
def test_rules_order_and_tie_break(train_store, tmp_path):
    with build(train_store, tmp_path) as vocab:
        assert vocab.keys() == EXPECTED_KEYS
        assert vocab.size == len(EXPECTED_KEYS)
        ids = {key: i for i, key in enumerate(EXPECTED_KEYS)}
        # rule (1): df 1 lemma out; evaluation-only words out; stopwords out
        for key in ("gelo", "zebra", "preto", "o", "de"):
            assert vocab.action_id(key) == -1
        # rule (3): n:0..n:10 always; 1990/2500/777777 by df >= 5; 7 digits and df 1 out
        for key in ("n:1234567", "n:07"):
            assert vocab.action_id(key) == -1
        assert vocab.freq[ids["gato"]] == 5 and vocab.doc_freq[ids["gato"]] == 3
        assert vocab.freq[ids["casa"]] == 5 and vocab.doc_freq[ids["casa"]] == 4
        assert vocab.doc_freq[ids["n:777777"]] == 5 and vocab.freq[ids["n:777777"]] == 5
        assert vocab.freq[ids["n:7"]] == 0  # "07" is a different number
        assert vocab.is_numeric.tolist() == [k.startswith("n:") for k in EXPECTED_KEYS]
        assert vocab.manifest["partition"] == "train"
        assert vocab.manifest["n_train_docs"] == len(TRAIN_DOCS)


@pytest.mark.req("REQ-ENV-004")
def test_stopwords_go_to_stop_table(train_store, tmp_path):
    with build(train_store, tmp_path) as vocab:
        stops = vocab.stopwords()
        assert stops == sorted(stops) and vocab.n_stop == len(stops) > 100
        assert vocab.stopword(vocab.stop_id("o")) == "o"
        assert vocab.stop_id("de") >= 0 and vocab.stop_id("casa") == -1
        table = pq.read_table(vocab.path / "stop.parquet")
        assert table.column_names == ["stop_id", "form"]


@pytest.mark.req("REQ-ENV-004")
def test_lexicon_adds_lemmas_but_not_stopwords_or_numbers(train_store, tmp_path):
    lexicon = ["Árvore", "gatos", "de", "gelo", "42", "  ", "gato"]
    with build(train_store, tmp_path, lexicon=lexicon) as vocab:
        keys = vocab.keys()
        # gelo (freq 2, df 1) enters through the lexicon with its training statistics
        assert keys[:8] == ["casa", "gato", "n:1990", "n:2500", "n:777777", "rio", "sol", "gelo"]
        assert keys[8] == "lua"
        assert keys[9:] == ["gatos", *NUMERIC_FORCED, "árvore"]
        assert vocab.freq[vocab.action_id("gelo")] == 2
        assert vocab.doc_freq[vocab.action_id("gelo")] == 1
        assert vocab.freq[vocab.action_id("árvore")] == 0
        assert vocab.action_id("de") == -1 and vocab.action_id("n:42") == -1
        assert vocab.action_id("42") == -1
        # F = training forms ∪ lexicon forms, same order
        forms = vocab.forms()
        assert forms[:6] == ["1234567", "1990", "2500", "777777", "casa", "gato"]
        assert forms[-3:] == ["42", "gatos", "árvore"]  # lexicon-only forms have frequency 0
        assert {"árvore", "gatos", "42", "o", "de", "1990", "07"} <= set(forms)
        assert "zebra" not in forms
        assert vocab.manifest["n_lexicon_forms"] == 6


@pytest.mark.req("REQ-ENV-004")
def test_cut_keeps_forced_numbers(train_store, tmp_path):
    with build(train_store, tmp_path, config=replace(CONFIG, max_size=12)) as vocab:
        assert vocab.keys() == ["casa", *NUMERIC_FORCED]
    with pytest.raises(ValueError, match="never cut"):
        build(train_store, tmp_path, config=replace(CONFIG, max_size=10), name="small")


@pytest.mark.req("REQ-ENV-004")
def test_form_cut(train_store, tmp_path):
    with build(train_store, tmp_path, config=replace(CONFIG, max_forms=3)) as vocab:
        # freq: 1990 5, 2500 5, 777777 5, casa 5, gato 5, 1234567 6
        assert vocab.forms() == ["1234567", "1990", "2500"]
        assert vocab.n_forms == 3


@pytest.mark.req("REQ-ENV-004")
def test_min_doc_freq_and_disabled_numbers(train_store, tmp_path):
    with build(train_store, tmp_path, config=replace(CONFIG, min_doc_freq=1)) as vocab:
        assert vocab.action_id("gelo") >= 0
    ruleset = identity_ruleset(numbers="disabled")
    with build(train_store, tmp_path, ruleset=ruleset, name="nonum") as vocab:
        assert not vocab.is_numeric.any()
        assert vocab.action_id("1990") >= 0 and vocab.action_id("n:1990") == -1
        assert vocab.action_id("07") == -1  # df 1


@pytest.mark.req("REQ-ENV-004")
def test_demo_partition_uses_min_doc_freq_one(tmp_path):
    docs = [("Gato", "gato casa", "demo"), ("Rio", "rio rio", "demo")]
    with (
        CorpusStore.open(build_corpus(tmp_path / "demo", docs, "demo")) as store,
        build(store, tmp_path) as vocab,
    ):
        assert vocab.manifest["partition"] == "demo"
        assert vocab.manifest["min_doc_freq"] == 1
        assert vocab.keys()[:3] == ["rio", "gato", "casa"]
    with (
        CorpusStore.open(build_corpus(tmp_path / "none", [("A", "b", "w-val")], "x")) as store,
        pytest.raises(ArtigoEnvError, match="no 'train'"),
    ):
        build(store, tmp_path, name="none")


@pytest.mark.req("REQ-ENV-004")
def test_length_mask(tmp_path):
    docs = [("Gatos", "gato gatos gata casa", "train"), ("Gato", "gato casa", "train")]
    ruleset = Ruleset(name="test-protocol", lemmatizer="test-protocol")
    with (
        CorpusStore.open(build_corpus(tmp_path / "c", docs, "lm", ("test-protocol",))) as store,
        build(store, tmp_path, ruleset=ruleset, lexicon=["gatinhos"]) as vocab,
    ):
        mask = np.array(vocab.length_mask)
        assert mask.dtype == np.uint64
        assert int(mask[vocab.action_id("gato")]) == (1 << 3) | (1 << 4)  # gato, gatos
        assert int(mask[vocab.action_id("casa")]) == 1 << 3
        assert int(mask[vocab.action_id("gatinho")]) == 1 << 7  # lexicon form "gatinhos"
        assert int(mask[vocab.action_id("n:5")]) == 2**64 - 1
    long_word = "x" * 70
    with (
        CorpusStore.open(build_corpus(tmp_path / "d", [("A", long_word, "demo")], "l")) as store,
        build(store, tmp_path, name="long") as vocab,
    ):
        assert int(vocab.length_mask[vocab.action_id(long_word)]) == 1 << 63


# -- mapping ------------------------------------------------------------------------------------


@pytest.mark.req("DEC-064", "DEC-065")
def test_map_store(train_store, tmp_path):
    ruleset = identity_ruleset()
    with build(train_store, tmp_path, config=replace(CONFIG, max_forms=8)) as vocab:
        mapping = map_store(vocab, train_store, ruleset)
        size = vocab.size
        assert mapping.lemma_obs.dtype == np.int32 and mapping.form_obs.dtype == np.int32
        assert mapping.lemma_obs.shape == (train_store.n_forms,)
        assert mapping.action_lemma.shape == (size,) and mapping.action_lemma.dtype == np.int64

        def obs(form: str) -> tuple[int, int]:
            i = train_store.form_id(form)
            return int(mapping.lemma_obs[i]), int(mapping.form_obs[i])

        assert obs("gato") == (vocab.action_id("gato"), vocab.form_id("gato"))
        assert obs("zebra") == (-1, -1)  # visible outside V ∪ S and outside F
        assert obs("o")[0] == size + vocab.stop_id("o")
        assert obs("de")[0] == size + vocab.stop_id("de")
        assert obs("1990")[0] == vocab.action_id("n:1990")
        assert obs("1234567")[0] == -1 and obs("07")[0] == -1
        assert obs("gelo") == (-1, -1)  # out of V and beyond the form cut
        view = train_store.lemma_view("identity")
        assert mapping.action_lemma[vocab.action_id("casa")] == view.lemma_id("casa")
        assert mapping.action_lemma[vocab.action_id("n:1990")] == -1
        assert mapping.action_lemma[vocab.action_id("n:3")] == -1

        # the cache is reused and gives the same arrays
        assert (vocab.path / "maps" / "train-build" / "manifest.json").is_file()
        again = map_store(vocab, train_store, ruleset)
        np.testing.assert_array_equal(again.lemma_obs, mapping.lemma_obs)
        np.testing.assert_array_equal(again.form_obs, mapping.form_obs)
        np.testing.assert_array_equal(again.action_lemma, mapping.action_lemma)
        fresh = map_store(vocab, train_store, ruleset, cache=False)
        np.testing.assert_array_equal(fresh.lemma_obs, mapping.lemma_obs)


@pytest.mark.req("DEC-064")
def test_map_store_rejects_another_ruleset(train_store, tmp_path):
    with build(train_store, tmp_path) as vocab, pytest.raises(IncompatibleRulesetError):
        map_store(vocab, train_store, identity_ruleset(numbers="disabled"))


# -- coverage -----------------------------------------------------------------------------------


@pytest.mark.req("REQ-ENV-005")
def test_title_coverage(train_store, tmp_path):
    w_docs = [
        ("Gato", "gato", "w-val"),
        ("Zebra", "zebra", "w-val"),
        ("O Rio", "rio", "w-val"),
        ("Rio 1990", "rio", "w-val"),
        ("Lua (zebra)", "lua", "w-val"),
        ("Zebra preto", "zebra preto zebra", "w-val"),
        ("Gato", "gato", "w-test"),
    ]
    with CorpusStore.open(build_corpus(tmp_path / "w", w_docs, "w-build")) as w:
        with build(train_store, tmp_path) as vocab:
            report = title_coverage(vocab, w, identity_ruleset(), "w-val")
            assert (report.n_articles, report.n_covered) == (6, 3)
            assert report.fraction == pytest.approx(0.5)
            assert report.uncovered == (1, 4, 5)
            assert report.missing == (("zebra", 3), ("preto", 1))
            assert report.to_dict()["uncovered"] == [1, 4, 5]
        ignored = identity_ruleset(title_parenthetical="ignored")
        with build(train_store, tmp_path, ruleset=ignored, name="ign") as vocab:
            report = title_coverage(vocab, w, ignored, "w-val")
            assert report.uncovered == (1, 5)
            empty = title_coverage(vocab, w, ignored, "nothing")
            assert (empty.n_articles, empty.fraction) == (0, 0.0)
