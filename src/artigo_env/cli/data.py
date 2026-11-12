"""``artigo-env data``: raw data and corpus builds (SDD 02 §B.8, SDD 06).

Subcommands register themselves on the ``data`` group through ``add_*_parser`` functions, so
``build`` (WP3.5) is added next to ``fetch`` without changing the group.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from typing import Any, TextIO

from artigo_env.core.errors import ArtigoEnvError
from artigo_env.core.paths import data_path
from artigo_env.corpus.decontam.windex import directory_hash
from artigo_env.corpus.fetch import ManualFetchRequired, fetch
from artigo_env.corpus.manifest import get_source, load_sources
from artigo_env.corpus.pipeline import (
    BuildError,
    CorpusConfig,
    build,
    build_w_index,
    load_config,
)
from artigo_env.corpus.wset import (
    N_TEST,
    N_VAL,
    SEED,
    SPLITS,
    default_context,
    eligible,
    l_max,
    load_title_list,
    make_splits,
    measure_snapshot,
    write_w_splits,
)


def add_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("data", help="fetch raw data and build corpora")
    data_subparsers = parser.add_subparsers(dest="data_command", required=True)
    add_fetch_parser(data_subparsers)
    add_build_parser(data_subparsers)
    add_report_parser(data_subparsers)
    add_w_splits_parser(data_subparsers)
    add_windex_parser(data_subparsers)


def add_fetch_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser(
        "fetch",
        help="download the pinned raw files of a source and verify their SHA-256",
        description="Download the files pinned in the source manifest to "
        "ARTIGO_DATA_ROOT/raw/<source>/ and verify their SHA-256 and size. A file that does not "
        "match is deleted. Sources that need registration or a request only print instructions.",
    )
    parser.add_argument("source", help="source id in the manifest (e.g. cc_news_pt)")
    parser.add_argument(
        "--files",
        nargs="+",
        metavar="FILE",
        help="only these pinned files (path, file name or stem, e.g. 0003)",
    )
    parser.add_argument(
        "--accept-terms", action="store_true", help="accept the source's terms of use"
    )
    parser.add_argument(
        "--manifest", type=Path, default=None, help="source manifest (default: the package's)"
    )
    parser.set_defaults(handler=run_fetch)


def run_fetch(args: argparse.Namespace, stdin: TextIO, stdout: TextIO) -> int:
    def say(message: str) -> None:
        stdout.write(message + "\n")
        stdout.flush()

    try:
        entry = get_source(args.source, load_sources(args.manifest))
        say(f"{entry.id}: state {entry.state}, licence {entry.license_id}")
        if entry.license_note:
            say(f"  {entry.license_note}")
        paths = fetch(
            entry,
            data_path("raw"),
            accept_terms=args.accept_terms,
            files=args.files,
            log=say,
        )
    except ManualFetchRequired as error:
        say(str(error))
        return 1
    except ArtigoEnvError as error:
        say(f"error: {error}")
        return 2
    if paths:
        say(f"{len(paths)} file(s) verified")
    return 0


# --- build -----------------------------------------------------------------------------------


def add_build_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser(
        "build",
        help="build a compiled corpus from a config (configs/corpus/<name>.yaml)",
        description="Run the corpus pipeline (SDD 06) for a YAML config and compile the result "
        "under ARTIGO_DATA_ROOT/corpus/<build_id>/. Finished shards are reused, so an "
        "interrupted build resumes. Options override fields of the config; the effective "
        "config is recorded in the manifest and in report.json.",
    )
    parser.add_argument("config", type=Path, help="corpus build config (YAML)")
    parser.add_argument("--shards", nargs="+", metavar="SHARD", help="only these shard ids")
    parser.add_argument("--lemmatizer", help="playability lemmatizer (e.g. identity)")
    parser.add_argument("--lemma-views", nargs="+", metavar="NAME", help="lemma views to compile")
    parser.add_argument("--w-index", help="build id of the W corpus, or 'none' to skip")
    parser.add_argument("--workers", type=int, help="worker processes (output does not change)")
    parser.add_argument(
        "--manifest", type=Path, default=None, help="source manifest (default: the package one)"
    )
    parser.add_argument(
        "--no-resume", action="store_true", help="discard finished shards and start over"
    )
    parser.add_argument(
        "--recompile",
        action="store_true",
        help="compile again even if the corpus exists (finished shards are still reused)",
    )
    parser.set_defaults(handler=run_build)


def _apply_overrides(config: CorpusConfig, args: argparse.Namespace) -> CorpusConfig:
    changes: dict[str, Any] = {}
    if args.shards:
        by_source: dict[str, list[str]] = {}
        for shard in args.shards:
            owner = next(
                (s for s in config.sources if shard == s or shard.startswith(f"{s}__")),
                config.sources[0] if len(config.sources) == 1 else None,
            )
            if owner is None:
                raise BuildError(f"cannot tell which source shard {shard!r} belongs to")
            by_source.setdefault(owner, []).append(shard)
        changes["shards"] = {k: tuple(v) for k, v in by_source.items()}
    if args.lemmatizer:
        changes["lemmatizer"] = args.lemmatizer
    if args.lemma_views:
        changes["lemma_views"] = tuple(args.lemma_views)
    if args.workers is not None:
        changes["workers"] = args.workers
    if args.w_index is not None:
        changes["w_index"] = None if args.w_index.lower() == "none" else args.w_index
    try:
        return replace(config, **changes) if changes else config
    except (TypeError, ValueError) as error:
        raise BuildError(str(error)) from None


def run_build(args: argparse.Namespace, stdin: TextIO, stdout: TextIO) -> int:
    def say(message: str) -> None:
        stdout.write(message + "\n")
        stdout.flush()

    try:
        config = _apply_overrides(load_config(args.config), args)
        sources = load_sources(args.manifest) if args.manifest else None
        result = build(
            config,
            data_root=data_path(),
            sources=sources,
            resume=not args.no_resume,
            recompile=args.recompile,
            log=say,
        )
    except ArtigoEnvError as error:
        say(f"error: {error}")
        return 2
    say(f"build_id {result.build_id}")
    say(f"{result.n_docs} document(s) at {result.out_dir}")
    return 0


# --- report ----------------------------------------------------------------------------------


def add_report_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("report", help="print the report of a compiled corpus build")
    parser.add_argument("build_id", help="build id (directory under ARTIGO_DATA_ROOT/corpus/)")
    parser.add_argument("--json", action="store_true", help="print report.json instead")
    parser.set_defaults(handler=run_report)


def run_report(args: argparse.Namespace, stdin: TextIO, stdout: TextIO) -> int:
    try:
        directory = data_path("corpus", args.build_id)
    except (ArtigoEnvError, ValueError) as error:
        stdout.write(f"error: {error}\n")
        return 2
    name = "report.json" if args.json else "report.md"
    path = directory / name
    if not path.is_file():
        stdout.write(f"error: no {name} for build {args.build_id} under corpus/\n")
        return 2
    stdout.write(path.read_text(encoding="utf-8"))
    return 0


# --- W splits and W index --------------------------------------------------------------------


def add_w_splits_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser(
        "w-splits",
        help="compute the W-test / W-val / W-pool split file from a W snapshot",
        description="Read a TextExtracts snapshot and the title list, apply the W eligibility "
        "rules (SDD 06 §12), draw the stratified splits (seed 20261109) and write the split "
        "TSV. Prints the counts, the largest body in W-test + W-val and the resulting L_max.",
    )
    parser.add_argument("--snapshot", type=Path, required=True, help="snapshot directory")
    parser.add_argument("--titles", type=Path, required=True, help="title list TSV")
    parser.add_argument("--out", type=Path, required=True, help="split file to write")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--n-test", type=int, default=N_TEST)
    parser.add_argument("--n-val", type=int, default=N_VAL)
    parser.set_defaults(handler=run_w_splits)


def run_w_splits(args: argparse.Namespace, stdin: TextIO, stdout: TextIO) -> int:
    def say(message: str) -> None:
        stdout.write(message + "\n")
        stdout.flush()

    try:
        ctx = default_context()
        titles = load_title_list(args.titles)
        measure = measure_snapshot(args.snapshot, ctx)
        result = eligible(measure, titles, ctx)
        rows = make_splits(result.pages, args.seed, args.n_test, args.n_val)
    except ArtigoEnvError as error:
        say(f"error: {error}")
        return 2
    tokens = {page.title: page.n_body_tokens for page in result.pages}
    evaluated = [tokens[row.title] for row in rows if row.split in ("w-test", "w-val")]
    max_tokens = max(evaluated, default=0)
    write_w_splits(args.out, rows, max_tokens=max_tokens, seed=args.seed)
    main_titles = sum(1 for entry in titles if entry.in_complete)
    say(f"title list: {len(titles)} titles, {main_titles} in the main list")
    say(f"snapshot: {measure.n_index_rows} index rows, {len(measure.pages)} pages")
    for reason, excluded in result.excluded.items():
        say(f"excluded ({reason}): {len(excluded)}")
        for title in excluded[:10]:
            say(f"  {title}")
        if len(excluded) > 10:
            say(f"  ... {len(excluded) - 10} more")
    counts = {split: sum(1 for row in rows if row.split == split) for split in SPLITS[:3]}
    say(f"eligible: {len(result.pages)}; " + ", ".join(f"{k} {v}" for k, v in counts.items()))
    all_max = max((page.n_body_tokens for page in result.pages), default=0)
    say(f"max body tokens: eligible {all_max}, w-test+w-val {max_tokens}")
    say(f"L_max: {l_max(max_tokens)}")
    say(f"written: {args.out}")
    return 0


def add_windex_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser(
        "windex",
        help="(re)build the decontamination index of a compiled W corpus",
        description="Build corpus/<build_id>/windex/ from a compiled evaluation build plus every "
        "requested and resolved title of its snapshot (data build does this automatically).",
    )
    parser.add_argument("build_id", help="build id of the W corpus")
    parser.add_argument(
        "--manifest", type=Path, default=None, help="source manifest (default: the package one)"
    )
    parser.set_defaults(handler=run_windex)


def run_windex(args: argparse.Namespace, stdin: TextIO, stdout: TextIO) -> int:
    try:
        sources = load_sources(args.manifest) if args.manifest else None
        target = build_w_index(args.build_id, data_root=data_path(), sources=sources)
    except ArtigoEnvError as error:
        stdout.write(f"error: {error}\n")
        return 2
    stdout.write(f"W index written to {target}\n")
    stdout.write(f"content hash {directory_hash(target)}\n")
    return 0
