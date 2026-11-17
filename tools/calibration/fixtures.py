"""Fidelity fixtures (SDD 03 §7.3): expected game behaviour without the article text."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from tools.calibration.align import partial_indices, revealed_indices
from tools.calibration.dom import dom_tokens, read_guess_list
from tools.calibration.segment import GuessSegment, segment_guesses
from tools.calibration.session import RecordedSession, extract_article

FIXTURE_VERSION = 2


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _expected(session: RecordedSession, segment: GuessSegment) -> dict[str, Any]:
    """What the game showed for one guess, read from the snapshot taken after it.

    ``count`` is the number the game shows in the guess list (0 for warnings), ``status`` the
    row status (``found``, ``not-found`` or ``warning``), ``revealed_token_indices`` the tokens
    that became fully visible (diff of the snapshots), ``partial_token_indices`` those that only
    gained some visible characters (numeric guesses) and ``matched_token_indices`` every token
    carrying the game's match class after the guess (including tokens already revealed, which
    the shown count includes).
    """
    unreliable = segment.overlapped or segment.quiescent_seq is None
    if segment.snapshot_after is None:
        return {
            "counted": False,
            "count": 0,
            "revealed_token_indices": [],
            "matched_token_indices": [],
            "partial_token_indices": [],
            "message": None,
            "status": None,
            "unreliable": True,
        }
    after_html = session.read_snapshot(segment.snapshot_after)
    after_list = read_guess_list(after_html)
    after_tokens = dom_tokens(after_html)
    newest = after_list.newest
    if segment.snapshot_before is not None:
        before_html = session.read_snapshot(segment.snapshot_before)
        before_total = read_guess_list(before_html).total
        before_tokens = dom_tokens(before_html)
        revealed = revealed_indices(before_tokens, after_tokens)
        partial = partial_indices(before_tokens, after_tokens)
        counted = (
            after_list.total > before_total
            if after_list.total is not None and before_total is not None
            else newest is not None and newest.status != "warning"
        )
    else:
        # No snapshot precedes the guess (the page loaded after it was typed): the diff is
        # unknown, the rest still comes from the guess list and the match class.
        unreliable = True
        counted = newest is not None and newest.status != "warning"
        matched = [t.index for t in after_tokens if t.matched] if counted else []
        revealed = [i for i in matched if after_tokens[i].revealed]
        partial = [i for i in matched if not after_tokens[i].revealed]
    expected: dict[str, Any] = {
        "counted": counted,
        "count": (newest.count or 0) if counted and newest is not None else 0,
        "revealed_token_indices": revealed,
        "partial_token_indices": partial,
        "matched_token_indices": [t.index for t in after_tokens if t.matched] if counted else [],
        "message": newest.value if newest is not None and newest.status == "warning" else None,
        "status": newest.status
        if newest is not None and (counted or newest.status == "warning")
        else None,
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
