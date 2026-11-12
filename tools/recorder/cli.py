"""Command line: ``python -m tools.recorder record --mode {daily,random,history,free}``."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from artigo_env.core.paths import data_path
from tools.recorder.session import RecorderConfig, run_session
from tools.recorder.writer import MODES

START_URLS = {
    "daily": "https://artigo.app/",
    "history": "https://artigo.app/",
    "free": "https://artigo.app/",
    "random": "https://artigo.app/aleatorio",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m tools.recorder")
    commands = parser.add_subparsers(dest="command", required=True)
    record = commands.add_parser("record", help="record one passive session")
    record.add_argument("--mode", required=True, choices=MODES)
    record.add_argument("--url", help="start URL (default depends on the mode)")
    record.add_argument("--no-screens", action="store_true", help="do not take screenshots")
    record.add_argument("--out-root", type=Path, help="default: <ARTIGO_DATA_ROOT>/recordings")
    record.add_argument(
        "--profile-dir", type=Path, help="default: <ARTIGO_DATA_ROOT>/recorder-profile"
    )
    record.add_argument(
        "--allow-host",
        action="append",
        metavar="HOST",
        help="host recorded in detail; repeatable; replaces the default (artigo.app)",
    )
    record.add_argument("--headless", action="store_true", help="run without a visible window")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = RecorderConfig(
        mode=args.mode,
        start_url=args.url or START_URLS[args.mode],
        out_root=args.out_root or data_path("recordings"),
        profile_dir=args.profile_dir or data_path("recorder-profile"),
        allowed_hosts=tuple(args.allow_host) if args.allow_host else ("artigo.app",),
        screenshots=not args.no_screens,
        headless=args.headless,
    )
    session = run_session(config)
    print(f"session saved to {session}")
    return 0
