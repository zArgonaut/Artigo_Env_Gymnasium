"""Guess segmentation (SDD 03 §7.1, REQ-CAL-001)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any


@dataclass(frozen=True)
class GuessSegment:
    """Everything observed between one ``input.submit`` and its ``dom.quiescent``."""

    index: int
    raw: str
    submit_seq: int
    quiescent_seq: int | None
    mutations: tuple[dict[str, Any], ...]
    snapshot_before: str | None
    snapshot_after: str | None
    overlapped: bool


def segment_guesses(events: Sequence[dict[str, Any]]) -> list[GuessSegment]:
    """Split the event stream into guesses.

    A new submit before the previous quiescence closes the previous segment as overlapped
    (its mutations cannot be attributed reliably). A truncated session leaves the last segment
    without ``quiescent_seq``.
    """
    segments: list[GuessSegment] = []
    last_snapshot: str | None = None
    current: GuessSegment | None = None
    mutations: list[dict[str, Any]] = []
    awaiting_after: int | None = None  # index in ``segments`` waiting for its guess snapshot

    def close(segment: GuessSegment, quiescent_seq: int | None, overlapped: bool) -> int:
        segments.append(
            replace(
                segment,
                quiescent_seq=quiescent_seq,
                mutations=tuple(mutations),
                overlapped=overlapped,
            )
        )
        return len(segments) - 1

    for event in events:
        kind = event.get("type")
        payload = event.get("payload", {}) or {}
        if kind == "input.submit":
            if current is not None:
                close(current, None, True)
            current = GuessSegment(
                index=len(segments),
                raw=str(payload.get("value", "")),
                submit_seq=event["seq"],
                quiescent_seq=None,
                mutations=(),
                snapshot_before=last_snapshot,
                snapshot_after=None,
                overlapped=False,
            )
            mutations = []
            awaiting_after = None
        elif kind == "dom.mutation" and current is not None:
            mutations.append(event)
        elif kind == "dom.quiescent" and current is not None:
            awaiting_after = close(current, event["seq"], False)
            current = None
            mutations = []
        elif kind == "snapshot":
            name = payload.get("file")
            if payload.get("reason") == "guess" and awaiting_after is not None and name:
                segments[awaiting_after] = replace(segments[awaiting_after], snapshot_after=name)
                awaiting_after = None
            if name:
                last_snapshot = name
    if current is not None:
        close(current, None, False)
    return segments
