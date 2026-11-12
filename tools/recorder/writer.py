"""Session writer: one folder per recording session, tolerant to interruption (REQ-REC-006).

``events.jsonl`` is append-only and flushed to disk per event, so a killed process loses at most
a partially written last line (ignored by :func:`read_events`). Every other file is written to
a temporary name and renamed.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

from tools.recorder.events import SCHEMA_VERSION, RedactedEvent
from tools.recorder.redact import redact_text

RECORDER_VERSION = "0.1.0"
MODES = ("daily", "random", "history", "free")
EVENTS_FILE = "events.jsonl"
META_FILE = "meta.json"


def _atomic_write(path: Path, data: bytes) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


class SessionWriter:
    """Writes redacted events and attachments for one recording session."""

    def __init__(
        self,
        root: Path,
        mode: str,
        *,
        now: Callable[[], datetime] = datetime.now,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        if mode not in MODES:
            raise ValueError(f"unknown mode {mode!r}; expected one of {MODES}")
        self._now = now
        self._monotonic = monotonic
        self._origin = monotonic()
        self._seq = 0
        started = now()
        self.path = root / f"{started.strftime('%Y-%m-%d_%H%M%S')}_{mode}"
        for name in ("bodies", "snapshots", "screens"):
            (self.path / name).mkdir(parents=True, exist_ok=False)
        self._meta: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "recorder_version": RECORDER_VERSION,
            "playwright_version": None,
            "browser_version": None,
            "mode": mode,
            "bundle_hash": None,
            "started_at": started.astimezone().isoformat(timespec="seconds"),
            "ended_at": None,
            "n_guesses": None,
            "complete": False,
        }
        self._events = (self.path / EVENTS_FILE).open("a", encoding="utf-8", newline="\n")
        self.write_meta()

    @property
    def next_seq(self) -> int:
        """Sequence number the next emitted event will get (for naming attachments)."""
        return self._seq

    def write_meta(self, **fields: Any) -> None:
        """Update ``meta.json`` atomically; unknown fields are rejected."""
        unknown = sorted(set(fields) - set(self._meta))
        if unknown:
            raise ValueError(f"unknown meta fields: {unknown}")
        self._meta.update(fields)
        text = json.dumps(self._meta, indent=2, ensure_ascii=False) + "\n"
        _atomic_write(self.path / META_FILE, text.encode("utf-8"))

    def emit(self, event: RedactedEvent) -> int:
        """Append one event and return its sequence number."""
        if not isinstance(event, RedactedEvent):
            raise TypeError("the writer only accepts events built by make_event()")
        seq = self._seq
        self._seq += 1
        record = {
            "seq": seq,
            "t_mono_ms": round((self._monotonic() - self._origin) * 1000, 3),
            "t_wall": self._now().astimezone().isoformat(timespec="milliseconds"),
            "type": event.type,
            "payload": event.payload,
        }
        self._events.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._events.flush()
        os.fsync(self._events.fileno())
        return seq

    def save_body(self, seq: int, data: bytes, ext: str, *, textual: bool = True) -> str:
        """Store a response body; textual bodies are redacted first. Returns the relative name."""
        if textual:
            data = redact_text(data.decode("utf-8", errors="replace")).encode("utf-8")
        name = f"bodies/{seq}.{ext.lstrip('.') or 'bin'}"
        _atomic_write(self.path / name, data)
        return name

    def save_snapshot(self, seq: int, html: str) -> str:
        name = f"snapshots/{seq}.html"
        _atomic_write(self.path / name, redact_text(html).encode("utf-8"))
        return name

    def save_screen(self, seq: int, png: bytes) -> str:
        name = f"screens/{seq}.png"
        _atomic_write(self.path / name, png)
        return name

    def close(self, n_guesses: int) -> None:
        """Mark the session as complete."""
        if self._events.closed:
            return
        self._events.close()
        self.write_meta(
            ended_at=self._now().astimezone().isoformat(timespec="seconds"),
            n_guesses=n_guesses,
            complete=True,
        )


def read_events(session_dir: Path) -> list[dict[str, Any]]:
    """Read ``events.jsonl``; a truncated last line (interrupted session) is ignored."""
    events: list[dict[str, Any]] = []
    lines = (session_dir / EVENTS_FILE).read_text(encoding="utf-8").split("\n")
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            events.append(json.loads(line))
        except ValueError:
            if index >= len(lines) - 2:
                break
            raise
    return events


def read_meta(session_dir: Path) -> dict[str, Any]:
    meta: dict[str, Any] = json.loads((session_dir / META_FILE).read_text(encoding="utf-8"))
    return meta
