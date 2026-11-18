"""Reading the game's outcome of every guess from the recorded guess lists (REQ-CAL-001).

The guess list shows the *newest* guess first, one row per counted guess, below a header with
the number of counted guesses. A snapshot may be taken long after the guess it was meant for
(the recorder's queue can lag behind the page), so the outcome is never read from "the
snapshot after the guess". Instead every row of every snapshot is placed in a ledger by its
ordinal (the counted-guess number, from the header total and the row position), and the
recorded guesses are aligned to the ledger by their label: a guess that owns a ledger row was
counted, one that does not was not (its warning message is read from the transient row when
some snapshot caught it).
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass

from tools.calibration.dom import GuessList, read_guess_list
from tools.calibration.segment import GuessSegment
from tools.calibration.session import RecordedSession

_EDGE = re.compile(r"^[^\w#]+|[^\w]+$")


def clean_label(raw: str) -> str:
    """What the game shows in the list for a typed guess: lower case, surrounding punctuation
    and spaces dropped (``"Sendo]"`` is shown as ``sendo``)."""
    return _EDGE.sub("", unicodedata.normalize("NFC", raw.strip()).lower())


@dataclass(frozen=True)
class ListObservation:
    """The guess list of one snapshot."""

    seq: int
    file: str
    guess_list: GuessList


@dataclass(frozen=True)
class LedgerRow:
    label: str
    status: str
    count: int | None


@dataclass(frozen=True)
class Outcome:
    """The game's reaction to one guess; ``reason`` is set when it could not be read."""

    label: str | None
    counted: bool
    status: str | None
    count: int
    message: str | None
    total_after: int | None
    ordinal: int | None = None
    reason: str | None = None
    prior_history: bool = False


def observations(session: RecordedSession) -> list[ListObservation]:
    """Guess lists of every snapshot, in event order."""
    found: list[ListObservation] = []
    for event in session.events:
        payload = event.get("payload", {}) or {}
        if event.get("type") == "snapshot" and payload.get("file"):
            html = session.read_snapshot(payload["file"])
            found.append(ListObservation(event["seq"], payload["file"], read_guess_list(html)))
    return found


def read_prior_guesses(session: RecordedSession) -> list[str]:
    """Keys of the guesses the game already holds for this game when the page loads.

    The client keeps ``games.list[<mode or number>].guesses`` in ``localStorage``
    (``persist:root``), keyed by lemma, or ``#<digits>`` for numeric guesses, in guess order.
    The first ``storage`` event is the state at load. A game that is absent or unreadable
    yields an empty list.
    """
    mode = str(session.meta.get("mode", ""))
    for event in session.events:
        if event.get("type") != "storage":
            continue
        items = (event.get("payload", {}) or {}).get("items", {}) or {}
        raw = items.get("persist:root")
        if raw is None:
            continue
        try:
            games = json.loads(json.loads(raw)["games"])["list"]
        except (ValueError, KeyError, TypeError):
            return []
        game = games.get(mode) if isinstance(games, dict) else None
        if game is None and isinstance(games, dict) and len(games) == 1:
            game = next(iter(games.values()))
        guesses = game.get("guesses") if isinstance(game, dict) else None
        return [str(key) for key in guesses] if isinstance(guesses, dict) else []
    return []


def _list_total(guess_list: GuessList) -> int | None:
    if guess_list.total is not None:
        return guess_list.total
    if guess_list.container_rows is not None and not guess_list.has_warning:
        return guess_list.container_rows - 1  # header row, no warning row
    return None


def build_ledger(lists: Sequence[GuessList]) -> dict[int, LedgerRow]:
    """Counted-guess ordinal -> row, from every list (first observation wins)."""
    ledger: dict[int, LedgerRow] = {}
    for guess_list in lists:
        total = _list_total(guess_list)
        if total is None:
            continue
        shift = 1 if guess_list.has_warning else 0
        for entry in guess_list.entries:
            if entry.status == "warning" or entry.index < 1:
                continue
            ordinal = total - (entry.index - 1 - shift)
            if ordinal >= 1 and ordinal not in ledger:
                ledger[ordinal] = LedgerRow(clean_label(entry.label), entry.status, entry.count)
    return ledger


def _warning_message(
    label: str, total: int | None, after_seq: int, items: Sequence[ListObservation]
) -> str | None:
    for item in items:
        if item.seq < after_seq or _list_total(item.guess_list) != total:
            continue
        for entry in item.guess_list.entries:
            if entry.status == "warning" and clean_label(entry.label) == label:
                return entry.value
    return None


def _lcs_pairs(segments: Sequence[str], rows: Sequence[str | None]) -> list[tuple[int, int]]:
    """Longest common subsequence of labels, taking the earliest guess for a repeated label."""
    m, k = len(segments), len(rows)
    table = [[0] * (k + 1) for _ in range(m + 1)]
    for i in range(m - 1, -1, -1):
        for j in range(k - 1, -1, -1):
            table[i][j] = (
                table[i + 1][j + 1] + 1
                if rows[j] is not None and rows[j] == segments[i]
                else max(table[i + 1][j], table[i][j + 1])
            )
    pairs: list[tuple[int, int]] = []
    i = j = 0
    while i < m and j < k:
        if (
            rows[j] is not None
            and rows[j] == segments[i]
            and table[i][j] == table[i + 1][j + 1] + 1
        ):
            pairs.append((i, j))
            i += 1
            j += 1
        elif table[i + 1][j] >= table[i][j + 1]:
            i += 1
        else:
            j += 1
    return pairs


def read_outcomes(
    segments: Sequence[GuessSegment],
    items: Sequence[ListObservation],
    initial_total: int,
) -> list[Outcome]:
    """One outcome per segment; see the module docstring for the method."""
    n = len(segments)
    unread = [Outcome(None, False, None, 0, None, None, reason="no guess list observed")] * n
    totals = [t for t in (_list_total(item.guess_list) for item in items) if t is not None]
    if not totals:
        return unread
    last_seq = max((item.seq for item in items), default=-1)
    observable = [i for i, s in enumerate(segments) if s.submit_seq < last_seq]
    m = len(observable)
    final = max(totals)
    rows_total = final - initial_total
    results: list[Outcome | None] = [None] * n
    for i in range(m, n):
        results[i] = Outcome(
            clean_label(segments[i].raw), False, None, 0, None, None, reason="no snapshot after it"
        )
    ledger = build_ledger([item.guess_list for item in items])
    labels = [clean_label(segments[i].raw) for i in range(m)]
    if rows_total < 0 or rows_total > m:
        reason = "the list total does not fit the number of guesses"
        return [
            Outcome(clean_label(s.raw), False, None, 0, None, None, reason=reason) for s in segments
        ]
    rows = [ledger[o].label if o in ledger else None for o in range(initial_total + 1, final + 1)]
    pairs = _lcs_pairs(labels, rows)
    anchors = [(-1, -1), *pairs, (m, rows_total)]
    for (a, x), (b, y) in zip(anchors, anchors[1:], strict=False):
        between, owned = b - a - 1, y - x - 1
        for offset in range(between):
            i = a + 1 + offset
            label = labels[i]
            if owned == 0:
                total = initial_total + x + 1
                message = _warning_message(label, total, segments[i].submit_seq, items)
                results[i] = Outcome(label, False, "warning", 0, message, total)
            elif owned == between:
                results[i] = _counted(label, initial_total + x + 2 + offset, ledger)
            else:
                results[i] = Outcome(
                    label, False, None, 0, None, None, reason="ambiguous: rows cannot be paired"
                )
        if b < m:
            results[b] = _counted(labels[b], initial_total + y + 1, ledger)
    return [r for r in results if r is not None]


def _counted(label: str, ordinal: int, ledger: dict[int, LedgerRow]) -> Outcome:
    row = ledger.get(ordinal)
    if row is None:
        return Outcome(
            label, True, None, 0, None, ordinal, ordinal, reason="its row was never observed"
        )
    return Outcome(row.label, True, row.status, row.count or 0, None, ordinal, ordinal)
