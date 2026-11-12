"""Loading of recorded sessions (the consumer side of the recorder contract)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tools.recorder import selectors
from tools.recorder.writer import read_events, read_meta


@dataclass(frozen=True)
class RecordedSession:
    """A recorded session folder; attachments are read lazily."""

    path: Path
    meta: dict[str, Any]
    events: list[dict[str, Any]]

    def read_body(self, name: str) -> bytes:
        """Read a file below ``bodies/`` given its relative name (e.g. ``bodies/12.json``)."""
        return (self.path / name).read_bytes()

    def read_snapshot(self, name: str) -> str:
        """Read a file below ``snapshots/`` given its relative name."""
        return (self.path / name).read_text(encoding="utf-8")


def load_session(path: Path) -> RecordedSession:
    """Load ``meta.json`` and ``events.jsonl`` (a truncated last line is tolerated)."""
    return RecordedSession(path=path, meta=read_meta(path), events=read_events(path))


def extract_article(session: RecordedSession) -> tuple[str, str] | None:
    """Return ``(title, text)`` served by the game, or ``None`` if it was not recorded.

    The format is provisional (JSON ``{"title", "text"}``, see the recorder contract) and is
    deliberately confined to this function until the first real recording fixes it.
    """
    for event in session.events:
        if event.get("type") != "net.response":
            continue
        payload = event.get("payload", {})
        body_file = payload.get("body_file")
        if selectors.ARTICLE_PATH_MARKER not in str(payload.get("url", "")) or not body_file:
            continue
        try:
            data = json.loads(session.read_body(body_file).decode("utf-8"))
            return str(data["title"]), str(data["text"])
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return None
