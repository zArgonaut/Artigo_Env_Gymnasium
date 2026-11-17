"""``artigo-env knowledge build``: the minimal knowledge store of phase F4 (SDD 10 §4.1)."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TextIO

from artigo_env.core.errors import ArtigoEnvError
from artigo_env.core.paths import data_path
from artigo_env.core.vocab import Vocabulary
from artigo_env.corpus.store import CorpusStore
from artigo_env.engine.rulesets import PRESETS, get_ruleset
from artigo_env.services.knowledge import REGIMES, SOURCE_CONFIGS, KnowledgeStore
from artigo_env.services.knowledge_build import build_knowledge


def add_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("knowledge", help="build the knowledge store")
    knowledge_subparsers = parser.add_subparsers(dest="knowledge_command", required=True)
    build = knowledge_subparsers.add_parser(
        "build",
        help="build the knowledge store of (training build, ruleset, vocabulary)",
        description="Count per-action statistics over the training partition of a compiled "
        "corpus; written to ARTIGO_DATA_ROOT/knowledge/<knowledge_id>/.",
    )
    build.add_argument("--corpus", required=True, help="build id under corpus/ or a path")
    build.add_argument("--ruleset", required=True, choices=PRESETS.names(), help="ruleset preset")
    build.add_argument("--vocab", required=True, help="vocabulary id under vocab/ or a path")
    build.add_argument("--source-config", default="X->W", choices=SOURCE_CONFIGS)
    build.add_argument("--regime", default="open", choices=REGIMES)
    build.add_argument(
        "--out-root",
        type=Path,
        default=None,
        help="output root (default: ARTIGO_DATA_ROOT/knowledge)",
    )
    build.set_defaults(handler=run_build)


def _resolve(value: str, kind: str) -> Path:
    path = Path(value)
    if path.is_dir():
        return path
    return data_path(kind, value)


def run_build(args: argparse.Namespace, stdin: TextIO, stdout: TextIO) -> int:
    try:
        ruleset = get_ruleset(args.ruleset)
        out_root = args.out_root if args.out_root is not None else data_path("knowledge")
        with (
            CorpusStore.open(_resolve(args.corpus, "corpus")) as store,
            Vocabulary.open(_resolve(args.vocab, "vocab")) as vocab,
        ):
            path = build_knowledge(
                store,
                ruleset,
                vocab,
                out_root,
                source_config=args.source_config,
                regime=args.regime,
            )
            knowledge = KnowledgeStore.open(path, vocab=vocab)
        stdout.write(
            f"knowledge {knowledge.knowledge_id}: {knowledge.n_docs} training docs "
            f"({knowledge.manifest['partition']}), |V|={knowledge.vocab_size}, "
            f"{knowledge.source_config}, {knowledge.regime}\n"
            f"  build {knowledge.build_id}, ruleset {knowledge.manifest['ruleset_name']}, "
            f"vocab {knowledge.vocab_id}\n"
            f"  {path}\n"
        )
    except (ArtigoEnvError, ValueError, NotImplementedError, OSError) as error:
        stdout.write(f"error: {error}\n")
        return 2
    return 0
