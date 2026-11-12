"""``artigo-env vocab``: build the action vocabulary and measure title coverage (SDD 09 §2)."""

from __future__ import annotations

import argparse
from collections.abc import Iterator
from pathlib import Path
from typing import TextIO

from artigo_env.core.config import canonical_json
from artigo_env.core.errors import ArtigoEnvError
from artigo_env.core.paths import data_path
from artigo_env.core.vocab import Vocabulary
from artigo_env.corpus.store import CorpusStore
from artigo_env.corpus.vocab import VocabConfig, build_vocab, title_coverage, with_max_size
from artigo_env.engine.rulesets import PRESETS, Ruleset, get_ruleset


def add_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("vocab", help="build the action vocabulary V")
    vocab_subparsers = parser.add_subparsers(dest="vocab_command", required=True)

    build = vocab_subparsers.add_parser(
        "build",
        help="build V for a training build and a ruleset",
        description="Build the action vocabulary from the training partition of a compiled "
        "corpus and an optional open lexicon; written to ARTIGO_DATA_ROOT/vocab/<vocab_id>/.",
    )
    build.add_argument("--corpus", required=True, help="build id under corpus/ or a path")
    build.add_argument("--ruleset", required=True, choices=PRESETS.names(), help="ruleset preset")
    build.add_argument(
        "--lexicon", type=Path, default=None, help="UTF-8 file with one lexicon form per line"
    )
    build.add_argument("--max-size", type=int, default=None, help="maximum |V|")
    build.add_argument(
        "--out-root", type=Path, default=None, help="output root (default: ARTIGO_DATA_ROOT/vocab)"
    )
    build.set_defaults(handler=run_build)

    coverage = vocab_subparsers.add_parser(
        "coverage",
        help="title coverage of a split (REQ-ENV-005)",
        description="Fraction of the articles of a split whose title content lemmas are all in "
        "V. The report is printed and written to <vocab>/coverage/<build_id>/<split>.json.",
    )
    coverage.add_argument("--vocab", required=True, help="vocabulary id under vocab/ or a path")
    coverage.add_argument("--corpus", required=True, help="build id under corpus/ or a path")
    coverage.add_argument("--split", default="w-val", help="partition to measure (default w-val)")
    coverage.add_argument(
        "--show", type=int, default=10, help="number of missing keys to print (default 10)"
    )
    coverage.set_defaults(handler=run_coverage)


def _resolve(value: str, kind: str) -> Path:
    path = Path(value)
    if path.is_dir():
        return path
    return data_path(kind, value)


def _lexicon_lines(path: Path) -> Iterator[str]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            word = line.strip()
            if word and not word.startswith("#"):
                yield word


def run_build(args: argparse.Namespace, stdin: TextIO, stdout: TextIO) -> int:
    try:
        ruleset = get_ruleset(args.ruleset)
        config = with_max_size(VocabConfig(), args.max_size)
        out_root = args.out_root if args.out_root is not None else data_path("vocab")
        if args.lexicon is not None and not args.lexicon.is_file():
            stdout.write(f"error: lexicon file not found: {args.lexicon}\n")
            return 2
        lexicon = _lexicon_lines(args.lexicon) if args.lexicon is not None else ()
        with CorpusStore.open(_resolve(args.corpus, "corpus")) as store:
            path = build_vocab(store, ruleset, config, out_root, lexicon_forms=lexicon)
        with Vocabulary.open(path) as vocab:
            manifest = vocab.manifest
            stdout.write(
                f"vocab {vocab.vocab_id}: |V|={vocab.size} ({manifest['n_numeric']} numeric), "
                f"|S|={vocab.n_stop}, |F|={vocab.n_forms}\n"
                f"  build {manifest['build_id']} ({manifest['partition']}, "
                f"{manifest['n_train_docs']} docs), ruleset {manifest['ruleset_name']}, "
                f"{manifest['n_lexicon_forms']} lexicon forms\n"
                f"  {path}\n"
            )
    except (ArtigoEnvError, ValueError, OSError) as error:
        stdout.write(f"error: {error}\n")
        return 2
    return 0


def run_coverage(args: argparse.Namespace, stdin: TextIO, stdout: TextIO) -> int:
    try:
        with Vocabulary.open(_resolve(args.vocab, "vocab")) as vocab:
            ruleset = Ruleset.from_dict(vocab.manifest["ruleset"])
            with CorpusStore.open(_resolve(args.corpus, "corpus")) as store:
                report = title_coverage(vocab, store, ruleset, args.split)
            out = vocab.path / "coverage" / report.build_id / f"{args.split}.json"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes((canonical_json(report.to_dict()) + "\n").encode("utf-8"))
    except (ArtigoEnvError, ValueError, OSError) as error:
        stdout.write(f"error: {error}\n")
        return 2
    stdout.write(
        f"title coverage of {report.split} in {report.build_id} by vocab {report.vocab_id}: "
        f"{report.n_covered}/{report.n_articles} = {report.fraction:.2%}\n"
    )
    for key, count in report.missing[: max(args.show, 0)]:
        stdout.write(f"  missing {key!r}: {count} article(s)\n")
    stdout.write(f"  report: {out}\n")
    return 0
