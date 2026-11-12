"""Entry point of the ``artigo-env`` command."""

from __future__ import annotations

import argparse
import io
import sys
from collections.abc import Sequence
from typing import TextIO

from artigo_env.cli import data, play, vocab


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="artigo-env", description="ArtigoEnv: offline word-reveal game and research tools."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    play.add_parser(subparsers)
    data.add_parser(subparsers)
    vocab.add_parser(subparsers)
    return parser


def main(
    argv: Sequence[str] | None = None,
    stdin: TextIO | None = None,
    stdout: TextIO | None = None,
) -> int:
    """Run the command line; returns the process exit code."""
    out = sys.stdout if stdout is None else stdout
    if stdout is None and isinstance(out, io.TextIOWrapper):
        # Windows consoles default to a legacy code page that cannot print "█".
        out.reconfigure(encoding="utf-8", errors="replace")
    args = build_parser().parse_args(argv)
    handler = args.handler
    return int(handler(args, sys.stdin if stdin is None else stdin, out))


if __name__ == "__main__":
    raise SystemExit(main())
