"""Reader of the action vocabulary artifact (``core.vocab.Vocabulary``, SDD 09 §2)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from artigo_env.core.errors import ArtigoEnvError
from artigo_env.core.vocab import Vocabulary, length_bit
from artigo_env.corpus.store import CorpusStore, CorpusWriter
from artigo_env.corpus.types import DOC_SCHEMA, Row
from artigo_env.corpus.vocab import VocabConfig, build_vocab
from artigo_env.engine.rulesets import Ruleset
from artigo_env.text.lemmatizers import get_lemmatizer

pytestmark = pytest.mark.fast

DOCS = [
    ("Gato", "o gato e a casa 1990", "train"),
    ("Casa", "casa gato rio", "train"),
    ("Rio", "rio gato", "train"),
]


def make_vocab(root: Path) -> Path:
    writer = CorpusWriter(root / "corpus")
    for i, (title, body, split) in enumerate(DOCS):
        row: Row = dict.fromkeys(DOC_SCHEMA.names)
        row.update(
            source_id="s",
            doc_id=f"d{i}",
            title=title,
            body=body,
            license_id="MIT",
            provenance="{}",
            split=split,
        )
        writer.add(row)
    writer.add_lemma_view(get_lemmatizer("identity"))
    writer.finish({"build_id": "core-vocab"})
    ruleset = Ruleset(name="t", lemmatizer="identity")
    with CorpusStore.open(root / "corpus") as store:
        return build_vocab(
            store, ruleset, VocabConfig(numeric_max=3), root / "vocab", lexicon_forms=["lua"]
        )


@pytest.mark.req("REQ-ENV-004")
def test_open_round_trip(tmp_path):
    path = make_vocab(tmp_path)
    with Vocabulary.open(path) as vocab:
        manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        assert vocab.manifest == manifest
        assert vocab.vocab_id == path.name == manifest["vocab_id"]
        # gato 4, casa 3, rio 3, n:1990 1 (df 1 < 5: not added), lua 0 (lexicon), n:0..n:3
        assert vocab.keys() == ["gato", "casa", "rio", "lua", "n:0", "n:1", "n:2", "n:3"]
        assert len(vocab) == vocab.size == 8
        assert [vocab.key(i) for i in range(vocab.size)] == vocab.keys()
        assert vocab.action_id("rio") == 2 and vocab.action_id("zebra") == -1
        assert vocab.action_id("n:2") == 6
        with pytest.raises(KeyError):
            vocab.key(vocab.size)
        assert vocab.freq.dtype == np.int64 and vocab.doc_freq.dtype == np.int64
        assert vocab.is_numeric.dtype == np.bool_ and vocab.length_mask.dtype == np.uint64
        assert vocab.freq.tolist() == [4, 3, 3, 0, 0, 0, 0, 0]
        assert vocab.doc_freq.tolist() == [3, 2, 2, 0, 0, 0, 0, 0]
        assert vocab.is_numeric.tolist() == [False] * 4 + [True] * 4
        assert int(vocab.length_mask[3]) == 1 << length_bit(3)
        assert vocab.n_stop == len(vocab.stopwords())
        i = vocab.stop_id("o")
        assert i >= 0 and vocab.stopword(i) == "o" and vocab.stop_id("gato") == -1
        assert vocab.n_forms == len(vocab.forms())
        assert vocab.form(vocab.form_id("1990")) == "1990"
        assert vocab.form_id("lua") >= 0 and vocab.form_id("zebra") == -1
        assert vocab.manifest["n_forms"] == vocab.n_forms
        assert isinstance(vocab.freq, np.memmap)


@pytest.mark.req("REQ-ENV-004")
def test_length_bit():
    assert [length_bit(n) for n in (1, 4, 64, 65, 300)] == [0, 3, 63, 63, 63]


@pytest.mark.req("REQ-ENV-004")
def test_close_releases_files(tmp_path):
    path = make_vocab(tmp_path)
    vocab = Vocabulary.open(path)
    assert int(vocab.freq.sum()) > 0 and vocab.length_mask.shape == (vocab.size,)
    assert vocab.key(0) == "gato" and vocab.form_id("gato") >= 0
    vocab.close()
    vocab.close()  # idempotent
    shutil.rmtree(path)  # fails on Windows while a file is still mapped
    assert not path.exists()
    with pytest.raises(ArtigoEnvError, match="closed"):
        _ = vocab.freq


@pytest.mark.req("REQ-ENV-004")
def test_open_rejects_bad_directories(tmp_path):
    with pytest.raises(ArtigoEnvError, match="not a vocabulary"):
        Vocabulary.open(tmp_path)
    path = make_vocab(tmp_path)
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    manifest["format_version"] = 99
    (path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ArtigoEnvError, match="format version"):
        Vocabulary.open(path)
