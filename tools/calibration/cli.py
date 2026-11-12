"""Command line: ``python -m tools.calibration process <session-dir> --version cal-v1``."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

from artigo_env.core.paths import data_path
from tools.calibration.align import align, article_tokens
from tools.calibration.dom import dom_tokens
from tools.calibration.fixtures import build_fixture, dump_fixture, store_article
from tools.calibration.report import build_report
from tools.calibration.session import RecordedSession, extract_article, load_session

DEFAULT_FIXTURES_DIR = Path("tests/fidelity/recordings/fixtures")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(text.encode("utf-8"))  # bytes: no newline translation on Windows


def _first_snapshot_html(session: RecordedSession) -> str:
    for event in session.events:
        payload = event.get("payload", {})
        if event.get("type") == "snapshot" and payload.get("file"):
            return session.read_snapshot(payload["file"])
    return ""


def process(session_dir: Path, version: str, fixtures_dir: Path, calibration_root: Path) -> None:
    session = load_session(session_dir)
    article = extract_article(session)
    if article is None:
        raise SystemExit("error: the session has no recorded article response")
    title, text = article
    fixture = build_fixture(session, version)
    alignment = align(dom_tokens(_first_snapshot_html(session)), article_tokens(title, text))

    version_dir = calibration_root / version
    store_article(title, text, version_dir)
    out_dir = version_dir / session_dir.name
    divergences = "".join(
        json.dumps(asdict(item), sort_keys=True, ensure_ascii=False) + "\n"
        for item in alignment.divergences
    )
    _write_text(out_dir / "divergences.jsonl", divergences)
    _write_text(out_dir / "report.md", build_report(session, alignment, fixture))
    _write_text(fixtures_dir / f"{session_dir.name}.json", dump_fixture(fixture))
    print(
        f"processed {session_dir.name}: {len(fixture['guesses'])} guesses, "
        f"{len(alignment.divergences)} divergences"
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m tools.calibration")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("process", help="turn a recorded session into a fidelity fixture")
    run.add_argument("session_dir", type=Path)
    run.add_argument("--version", required=True, help="calibration version, e.g. cal-v1")
    run.add_argument("--fixtures-dir", type=Path, default=DEFAULT_FIXTURES_DIR)
    run.add_argument(
        "--data-root-dir",
        type=Path,
        default=None,
        help="calibration root overriding ARTIGO_DATA_ROOT/calibration (used by tests)",
    )
    args = parser.parse_args(argv)
    root = args.data_root_dir if args.data_root_dir is not None else data_path("calibration")
    try:
        process(args.session_dir, args.version, args.fixtures_dir, root)
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0
