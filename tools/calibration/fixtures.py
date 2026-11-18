"""Fidelity fixtures (SDD 03 §7.3): expected game behaviour without the article text."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from tools.calibration.align import partial_indices, revealed_indices
from tools.calibration.dom import DomToken, dom_tokens
from tools.calibration.effects import TokenEffects, replay_effects
from tools.calibration.outcomes import (
    ListObservation,
    Outcome,
    clean_label,
    observations,
    read_outcomes,
    read_prior_guesses,
)
from tools.calibration.segment import GuessSegment, segment_guesses
from tools.calibration.session import RecordedSession, extract_article

FIXTURE_VERSION = 3


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


_EMPTY: dict[str, Any] = {
    "counted": False,
    "count": 0,
    "revealed_token_indices": [],
    "matched_token_indices": [],
    "partial_token_indices": [],
    "message": None,
    "status": None,
}


def _snapshot_tokens(
    session: RecordedSession,
    items: Sequence[ListObservation],
    segment: GuessSegment,
    outcome: Outcome,
    *,
    after: bool,
) -> list[DomToken] | None:
    """Tokens of the snapshot that shows the page right after (or before) ``segment``, chosen by
    its guess list (the header total, and for ``after`` the newest row), never by position."""
    total = outcome.total_after
    if total is None:
        return None
    if not after:
        total -= 1 if outcome.counted else 0
    for item in reversed(items) if not after else items:
        if (item.seq > segment.submit_seq) != after:
            continue
        newest = item.guess_list.newest
        right_total = item.guess_list.total == total
        right_row = not after or (newest is not None and clean_label(newest.label) == outcome.label)
        if right_total and right_row:
            return dom_tokens(session.read_snapshot(item.file))
    return None


def _effects(
    session: RecordedSession,
    items: Sequence[ListObservation],
    segment: GuessSegment,
    outcome: Outcome,
    replayed: TokenEffects | None,
) -> tuple[dict[str, list[int]], str | None]:
    """Token effects of a counted guess: from the mutation stream, else from snapshots."""
    if replayed is not None and replayed.touched > 0:
        return (
            {
                "revealed_token_indices": list(replayed.revealed),
                "partial_token_indices": list(replayed.partial),
                "matched_token_indices": list(replayed.matched),
            },
            None,
        )
    before = _snapshot_tokens(session, items, segment, outcome, after=False)
    after = _snapshot_tokens(session, items, segment, outcome, after=True)
    if before is None or after is None:
        return {}, "token effects were not observed"
    return (
        {
            "revealed_token_indices": revealed_indices(before, after),
            "partial_token_indices": partial_indices(before, after),
            "matched_token_indices": [t.index for t in after if t.matched],
        },
        None,
    )


def _expectations(
    session: RecordedSession, segments: Sequence[GuessSegment], prior: Sequence[str]
) -> tuple[list[dict[str, Any]], int]:
    """The expected outcome of every guess, and the number of guesses the list held at load."""
    items = observations(session)
    first_total = items[0].guess_list.total if items else None
    initial_total = first_total if first_total is not None else len(prior)
    outcomes = read_outcomes(segments, items, initial_total)
    first_html = next(
        (session.read_snapshot(item.file) for item in items), ""
    )  # the load snapshot: token state before any guess
    initial = dom_tokens(first_html)
    replayed = replay_effects(initial, session.events, len(segments))
    seen: set[str] = set()
    expectations: list[dict[str, Any]] = []
    for index, (segment, outcome) in enumerate(zip(segments, outcomes, strict=True)):
        expected: dict[str, Any] = {**_EMPTY, "label": outcome.label}
        reason = outcome.reason
        if reason is None:
            expected["counted"] = outcome.counted
            expected["status"] = outcome.status
            expected["count"] = outcome.count
            expected["message"] = outcome.message
            if outcome.counted and outcome.count > 0:
                effects, reason = _effects(
                    session, items, segment, outcome, replayed[index] if replayed else None
                )
                expected.update(effects)
            if outcome.message == "já foi" and outcome.label not in seen:
                expected["prior_history"] = True  # repeat of a guess made in an earlier game
            if outcome.counted and outcome.label is not None:
                seen.add(outcome.label)
        if reason is not None:
            expected["unreliable"] = True
            expected["reason"] = reason
        expectations.append(expected)
    return expectations, initial_total


def build_fixture(session: RecordedSession, calibration_version: str) -> dict[str, Any]:
    """Build the fixture for ``session``; raises ``ValueError`` if the article was not recorded."""
    article = extract_article(session)
    if article is None:
        raise ValueError("the session has no recorded article response")
    title, text = article
    segments = segment_guesses(session.events)
    prior = read_prior_guesses(session)
    expectations, prior_total = _expectations(session, segments, prior)
    return {
        "fixture_version": FIXTURE_VERSION,
        "calibration_version": calibration_version,
        "bundle_hash": session.meta.get("bundle_hash"),
        "source_session": session.path.name,
        "article_ref": {"sha256": sha256_text(text), "title_sha256": sha256_text(title)},
        "prior_guesses": prior,
        "prior_total": prior_total,
        "guesses": [
            {"raw": segment.raw, "expected": expected}
            for segment, expected in zip(segments, expectations, strict=True)
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
