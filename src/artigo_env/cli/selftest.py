"""``artigo-env selftest``: check that every part of the installed package works here."""

from __future__ import annotations

import argparse
import time
from typing import TextIO


def add_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser(
        "selftest",
        help="check that the installation works on this machine",
        description="Run the product self-test against the packaged data: installation, text, "
        "engine, corpus, environment, determinism, leakage and evaluation. No network and no "
        "ARTIGO_DATA_ROOT needed. Exit code 0 only if no check failed.",
    )
    parser.add_argument("--quick", action="store_true", help="skip the slower checks")
    parser.add_argument("--json", action="store_true", help="print a JSON report instead")
    parser.add_argument(
        "--only", nargs="+", default=None, metavar="GROUP", help="run only these groups"
    )
    parser.set_defaults(handler=run_selftest)


def run_selftest(args: argparse.Namespace, stdin: TextIO, stdout: TextIO) -> int:
    from artigo_env.selftest import report
    from artigo_env.selftest.registry import UnknownGroupError, run_checks

    start = time.perf_counter()
    try:
        results = run_checks(quick=args.quick, only=args.only)
    except UnknownGroupError as exc:
        stdout.write(f"error: {exc}\n")
        return 2
    seconds = time.perf_counter() - start
    if args.json:
        stdout.write(report.to_json_text(results, quick=args.quick, seconds=seconds) + "\n")
    else:
        stdout.write(report.table(results, quick=args.quick, seconds=seconds) + "\n")
    return report.exit_code(results)
