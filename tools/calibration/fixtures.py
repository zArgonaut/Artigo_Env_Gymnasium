"""Fidelity fixtures (SDD 03 §7.3): expected game behaviour without the article text."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from tools.calibration.align import revealed_indices
from tools.calibration.dom import dom_tokens, read_counter, read_message
from tools.calibration.segment import GuessSegment, segment_guesses
from tools.calibration.session import RecordedSession, extract_article

FIXTURE_VERSION = 1


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _expected(session: RecordedSession, segment: GuessSegment) -> dict[str, Any]:
    unreliable = segment.overlapped or segment.quiescent_seq is None
    if segment.snapshot_after is None or segment.snapshot_before is None:
        # Without both snapshots nothing can be derived reliably.
        return {
            "counted": False,
            "count": 0,
            "revealed_token_indices": [],
            "message": None,
            "unreliable": True,
        }
    before_html = session.read_snapshot(segment.snapshot_before)
    after_html = session.read_snapshot(segment.snapshot_after)
    counter_before, counter_after = read_counter(before_html), read_counter(after_html)
    revealed = revealed_indices(dom_tokens(before_html), dom_tokens(after_html))
    expected: dict[str, Any] = {
        "counted": (
            counter_before is not None
            and counter_after is not None
            and counter_after > counter_before
        ),
        "count": len(revealed),
        "revealed_token_indices": revealed,
        "message": read_message(after_html),
    }
    if unreliable:
        expected["unreliable"] = True
    return expected


def build_fixture(session: RecordedSession, calibration_version: str) -> dict[str, Any]:
    """Build the fixture for ``session``; raises ``ValueError`` if the article was not recorded."""
    article = extract_article(session)
    if article is None:
        raise ValueError("the session has no recorded article response")
    title, text = article
    return {
        "fixture_version": FIXTURE_VERSION,
        "calibration_version": calibration_version,
        "bundle_hash": session.meta.get("bundle_hash"),
        "source_session": session.path.name,
        "article_ref": {"sha256": sha256_text(text), "title_sha256": sha256_text(title)},
        "guesses": [
            {"raw": segment.raw, "expected": _expected(session, segment)}
            for segment in segment_guesses(session.events)
        ],
    }


def dump_fixture(fixture: dict[str, Any]) -> str:
    """Deterministic JSON: sorted keys, no timestamps, trailing newline."""
    return json.dumps(fixture, sort_keys=True, indent=2, ensure_ascii=False) + "\n"


def store_article(title: str, text: str, version_dir: Path) -> str:
    """Write ``articles/<sha256>.json`` atomically below ``version_dir``; return the hash."""
    digest = sha256_text(text)
    target = version_dir / "articles" / f"{digest}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps({"title": title, "text": text}, sort_keys=True, ensure_ascii=False)
    temporary = target.with_name(target.name + ".tmp")
    with temporary.open("wb") as handle:
        handle.write(payload.encode("utf-8"))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, target)
    return digest


def load_fixtures(version: str, fixtures_dir: Path) -> list[dict[str, Any]]:
    """Fixtures of exactly ``version`` (versions never mix, REQ-CAL-004), by source session."""
    if not fixtures_dir.is_dir():
        return []
    loaded = [json.loads(path.read_text(encoding="utf-8")) for path in fixtures_dir.glob("*.json")]
    selected = [item for item in loaded if item.get("calibration_version") == version]
    return sorted(selected, key=lambda item: str(item.get("source_session", "")))
