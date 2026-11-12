"""``artigo-env vocab build`` and ``artigo-env vocab coverage``."""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from artigo_env.cli.main import main
from artigo_env.core.vocab import Vocabulary
from artigo_env.corpus.store import CorpusWriter
from artigo_env.corpus.types import DOC_SCHEMA, Row
from artigo_env.text.lemmatizers import get_lemmatizer

pytestmark = pytest.mark.fast

TRAIN = [
    ("Gato", "gato casa rio", "train"),
    ("Casa", "casa gato sol", "train"),
    ("Rio", "rio sol gato", "train"),
]
W = [
    ("Gato", "gato", "w-val"),
    ("Zebra", "zebra", "w-val"),
    ("Árvore", "árvore", "w-val"),
    ("Sol", "sol", "w-test"),
]


def write_corpus(out: Path, docs, build_id: str) -> None:
    writer = CorpusWriter(out)
    for i, (title, body, split) in enumerate(docs):
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
    writer.finish({"build_id": build_id})


@pytest.fixture
def data_root(tmp_path, monkeypatch):
    monkeypatch.setenv("ARTIGO_DATA_ROOT", str(tmp_path))
    write_corpus(tmp_path / "corpus" / "cli-train", TRAIN, "cli-train")
    write_corpus(tmp_path / "corpus" / "cli-w", W, "cli-w")
    return tmp_path


def run(*argv: str) -> tuple[int, str]:
    out = io.StringIO()
    code = main(list(argv), stdin=io.StringIO(), stdout=out)
    return code, out.getvalue()


@pytest.mark.req("REQ-ENV-004", "REQ-ENV-005")
def test_build_then_coverage(data_root):
    lexicon = data_root / "lexicon.txt"
    lexicon.write_text("# open lexicon\nÁrvore\n\nde\n", encoding="utf-8")
    code, text = run(
        "vocab",
        "build",
        "--corpus",
        "cli-train",
        "--ruleset",
        "no-grouping",
        "--lexicon",
        str(lexicon),
        "--max-size",
        "5000",
    )
    assert code == 0, text
    (vocab_dir,) = (data_root / "vocab").iterdir()
    assert f"vocab {vocab_dir.name}:" in text
    with Vocabulary.open(vocab_dir) as vocab:
        assert vocab.manifest["config"]["max_size"] == 5000
        assert vocab.manifest["ruleset_name"] == "no-grouping"
        assert vocab.manifest["n_lexicon_forms"] == 2
        assert vocab.action_id("árvore") >= 0 and vocab.action_id("de") == -1
        assert vocab.keys()[:4] == ["gato", "casa", "rio", "sol"]

    code, text = run(
        "vocab", "coverage", "--vocab", vocab_dir.name, "--corpus", "cli-w", "--split", "w-val"
    )
    assert code == 0, text
    assert "2/3 = 66.67%" in text
    assert "missing 'zebra': 1 article(s)" in text
    report = json.loads((vocab_dir / "coverage" / "cli-w" / "w-val.json").read_text("utf-8"))
    assert report["n_articles"] == 3 and report["uncovered"] == [1]
    assert (vocab_dir / "maps" / "cli-w" / "lemma_obs.npy").is_file()


@pytest.mark.req("REQ-ENV-004")
def test_build_accepts_a_path_and_out_root(data_root, tmp_path):
    out_root = tmp_path / "elsewhere"
    code, text = run(
        "vocab",
        "build",
        "--corpus",
        str(data_root / "corpus" / "cli-train"),
        "--ruleset",
        "no-grouping",
        "--out-root",
        str(out_root),
    )
    assert code == 0, text
    assert len(list(out_root.iterdir())) == 1


@pytest.mark.req("REQ-ENV-004")
def test_errors_are_reported(data_root):
    code, text = run("vocab", "build", "--corpus", "missing", "--ruleset", "no-grouping")
    assert code == 2 and text.startswith("error:")
    code, text = run(
        "vocab",
        "build",
        "--corpus",
        "cli-train",
        "--ruleset",
        "no-grouping",
        "--lexicon",
        str(data_root / "nope.txt"),
    )
    assert code == 2 and "lexicon file not found" in text
    code, text = run("vocab", "coverage", "--vocab", "0123456789abcdef", "--corpus", "cli-w")
    assert code == 2 and "not a vocabulary" in text
