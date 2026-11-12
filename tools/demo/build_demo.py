"""Compile the packaged ``demo`` corpus and one action vocabulary per ruleset preset.

Usage (from the repository root)::

    uv run python -m tools.demo.build_demo [--out DIR]

Reads ``src/artigo_env/assets/demo/articles.jsonl`` (30 synthetic articles, split ``demo``) and
writes, below ``DIR`` (default ``src/artigo_env/assets/demo``):

- ``corpus/``: the compiled store (``CorpusWriter``) with the lemma views ``identity`` and
  ``open-tables``; ``build_id`` is derived from the content, and the manifest sets ``l_max`` to
  1,024 so that observations of the demo environment stay small;
- ``vocab/<vocab_id>/``: one vocabulary per ruleset preset (the vocabulary id includes the
  ruleset hash, so presets that share a lemmatizer still get distinct artifacts), each with the
  observation map of the demo build already cached under ``maps/<build_id>/`` so an installed
  (read-only) package never has to write it.

The output is byte-identical for the same inputs and library versions; the tool deletes and
rewrites ``corpus/`` and ``vocab/`` on every run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any

from artigo_env.core.config import canonical_json, sha256_hex
from artigo_env.core.vocab import Vocabulary
from artigo_env.corpus.store import CorpusStore, CorpusWriter
from artigo_env.corpus.store.writer import FORMAT_VERSION
from artigo_env.corpus.types import DOC_SCHEMA, Row
from artigo_env.corpus.vocab import VocabConfig, build_vocab, map_store
from artigo_env.engine.rulesets import PRESETS, get_ruleset
from artigo_env.text.guess import clean_guess
from artigo_env.text.lemmatizers import get_lemmatizer
from artigo_env.text.normalize import lower, normalize
from artigo_env.text.stopwords import StopwordSpec, load_stopwords
from artigo_env.text.tokenizer import tokenize

TOOL_VERSION = 1
SOURCE_ID = "demo"
SPLIT = "demo"
DOMAIN = "demo"
VARIANT = "pt-BR"
L_MAX = 1024
LEMMA_VIEWS = ("identity", "open-tables")
REPO_ROOT = Path(__file__).resolve().parents[2]
DEMO_DIR = REPO_ROOT / "src" / "artigo_env" / "assets" / "demo"


def read_articles(path: Path) -> list[dict[str, Any]]:
    """The articles of ``articles.jsonl`` in file order."""
    articles: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            item = json.loads(line)
            for key in ("doc_id", "title", "body"):
                if not isinstance(item.get(key), str) or not item[key].strip():
                    raise ValueError(f"{path.name}:{number}: missing or empty {key!r}")
            articles.append(item)
    return articles


def _title_stats(title: str, body: str) -> tuple[int, bool]:
    """``(n_title_content, title_in_body)`` under ``game227`` and ``open-tables``."""
    stopwords = load_stopwords(StopwordSpec())
    lemmatizer = get_lemmatizer("open-tables")

    def lemmas(text: str) -> list[str]:
        out: list[str] = []
        for token in tokenize(normalize(text), "split"):
            form = lower(token.text)
            if form not in stopwords:
                out.append(lemmatizer.lemma(clean_guess(form)))
        return out

    title_lemmas = lemmas(title)
    body_lemmas = set(lemmas(body))
    return len(title_lemmas), all(lemma in body_lemmas for lemma in title_lemmas)


def make_row(article: dict[str, Any]) -> Row:
    row: Row = dict.fromkeys(DOC_SCHEMA.names)
    n_title_content, title_in_body = _title_stats(article["title"], article["body"])
    row.update(
        source_id=SOURCE_ID,
        doc_id=article["doc_id"],
        title=article["title"],
        body=article["body"],
        license_id=article.get("license_id", "MIT"),
        provenance="{}",
        variant=VARIANT,
        n_title_content=n_title_content,
        title_in_body=title_in_body,
        domain=DOMAIN,
        split=SPLIT,
    )
    return row


def demo_build_id(source: bytes) -> str:
    """``demo-`` + 12 hex of SHA-256 over the source bytes and everything that shapes the build."""
    identity = {
        "source_sha256": hashlib.sha256(source).hexdigest(),
        "tool_version": TOOL_VERSION,
        "format_version": FORMAT_VERSION,
        "lemma_views": {name: get_lemmatizer(name).content_hash() for name in LEMMA_VIEWS},
        "l_max": L_MAX,
    }
    return "demo-" + sha256_hex(canonical_json(identity))[:12]


def build_corpus(source_path: Path, out_dir: Path) -> str:
    """Compile ``source_path`` into ``out_dir``; returns the build id."""
    source = source_path.read_bytes()
    build_id = demo_build_id(source)
    articles = read_articles(source_path)
    writer = CorpusWriter(out_dir)
    with writer:
        for article in articles:
            writer.add(make_row(article))
        for name in LEMMA_VIEWS:
            writer.add_lemma_view(get_lemmatizer(name))
        writer.finish(
            {
                "build_id": build_id,
                "name": "demo",
                "kind": "demo",
                "l_max": L_MAX,
                "source_file": source_path.name,
                "source_sha256": hashlib.sha256(source).hexdigest(),
                "tool": "tools/demo/build_demo.py",
                "tool_version": TOOL_VERSION,
            }
        )
    return build_id


def build_vocabularies(corpus_dir: Path, vocab_root: Path) -> dict[str, str]:
    """One vocabulary per ruleset preset, with the demo map cached; preset -> vocab id."""
    ids: dict[str, str] = {}
    with CorpusStore.open(corpus_dir) as store:
        for name in sorted(PRESETS.names()):
            ruleset = get_ruleset(name)
            path = build_vocab(store, ruleset, VocabConfig(), vocab_root)
            with Vocabulary.open(path) as vocab:
                map_store(vocab, store, ruleset, cache=True)
                ids[name] = vocab.vocab_id
    return ids


def build_demo(out: Path, source_path: Path | None = None) -> dict[str, Any]:
    """Rebuild ``out/corpus`` and ``out/vocab``; returns a summary."""
    source_path = source_path or DEMO_DIR / "articles.jsonl"
    out = Path(out)
    corpus_dir = out / "corpus"
    vocab_root = out / "vocab"
    for directory in (corpus_dir, vocab_root):
        if directory.exists():
            shutil.rmtree(directory)
    out.mkdir(parents=True, exist_ok=True)
    build_id = build_corpus(source_path, corpus_dir)
    vocabs = build_vocabularies(corpus_dir, vocab_root)
    files = [p for d in (corpus_dir, vocab_root) for p in d.rglob("*") if p.is_file()]
    size = sum(p.stat().st_size for p in files)
    return {"build_id": build_id, "vocabularies": vocabs, "bytes": size}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--out", type=Path, default=DEMO_DIR, help="output directory")
    args = parser.parse_args(argv)
    summary = build_demo(args.out)
    sys.stdout.write(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
