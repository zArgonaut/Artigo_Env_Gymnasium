from __future__ import annotations

from typing import Any

import pytest

from tools.calibration.segment import segment_guesses

pytestmark = [pytest.mark.fast, pytest.mark.req("REQ-CAL-001")]


class Stream:
    """Tiny builder of synthetic event dicts with increasing ``seq``."""

    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def add(self, event_type: str, /, **payload: Any) -> int:
        seq = len(self.events)
        self.events.append({"seq": seq, "type": event_type, "payload": payload})
        return seq

    def guess(self, raw: str, mutations: int = 0, *, snapshot: bool = True) -> None:
        self.add("input.submit", value=raw)
        for _ in range(mutations):
            self.add("dom.mutation", kind="attributes")
        self.add("dom.quiescent", after_ms=300, mutations=mutations)
        if snapshot:
            self.add("snapshot", file=f"snapshots/{len(self.events)}.html", reason="guess")


def loaded() -> Stream:
    stream = Stream()
    stream.add("session.start")
    stream.add("snapshot", file="snapshots/1.html", reason="load")
    return stream


def test_three_ordinary_guesses() -> None:
    stream = loaded()
    stream.guess("rio", 3)
    stream.guess("vale", 2)
    stream.guess("ponte", 1)
    segments = segment_guesses(stream.events)
    assert [s.raw for s in segments] == ["rio", "vale", "ponte"]
    assert [s.index for s in segments] == [0, 1, 2]
    assert [len(s.mutations) for s in segments] == [3, 2, 1]
    assert all(s.quiescent_seq is not None and not s.overlapped for s in segments)
    assert all(s.submit_seq < s.quiescent_seq for s in segments if s.quiescent_seq)
    assert all(m["type"] == "dom.mutation" for s in segments for m in s.mutations)


def test_miss_has_zero_mutations() -> None:
    stream = loaded()
    stream.guess("zzz")
    (segment,) = segment_guesses(stream.events)
    assert segment.mutations == ()
    assert segment.quiescent_seq is not None
    assert not segment.overlapped


def test_overlapping_submits_mark_the_first_one() -> None:
    stream = loaded()
    stream.add("input.submit", value="rio")
    stream.add("dom.mutation", kind="attributes")
    stream.add("input.submit", value="vale")  # before the page stopped changing
    stream.add("dom.mutation", kind="attributes")
    stream.add("dom.quiescent", after_ms=300, mutations=2)
    stream.add("snapshot", file="snapshots/9.html", reason="guess")
    first, second = segment_guesses(stream.events)
    assert first.overlapped and first.quiescent_seq is None
    assert first.snapshot_after is None
    assert len(first.mutations) == 1
    assert not second.overlapped and second.quiescent_seq is not None
    assert len(second.mutations) == 1
    assert second.snapshot_after == "snapshots/9.html"


def test_truncated_session_leaves_last_segment_open() -> None:
    stream = loaded()
    stream.guess("rio", 1)
    stream.add("input.submit", value="vale")
    stream.add("dom.mutation", kind="attributes")
    done, cut = segment_guesses(stream.events)
    assert done.quiescent_seq is not None
    assert cut.quiescent_seq is None and not cut.overlapped
    assert cut.snapshot_after is None
    assert len(cut.mutations) == 1


def test_snapshot_before_and_after_association() -> None:
    stream = loaded()
    stream.guess("rio", 1)
    stream.guess("vale", 1)
    stream.add("snapshot", file="snapshots/end.html", reason="end")
    first, second = segment_guesses(stream.events)
    assert first.snapshot_before == "snapshots/1.html"
    assert first.snapshot_after is not None and first.snapshot_after != "snapshots/end.html"
    assert second.snapshot_before == first.snapshot_after
    assert second.snapshot_after is not None
    assert second.snapshot_after not in {first.snapshot_after, "snapshots/end.html"}


def test_events_before_first_submit_and_empty_stream() -> None:
    assert segment_guesses([]) == []
    stream = loaded()
    stream.add("dom.mutation", kind="attributes")
    stream.add("dom.quiescent", after_ms=300, mutations=1)
    assert segment_guesses(stream.events) == []
