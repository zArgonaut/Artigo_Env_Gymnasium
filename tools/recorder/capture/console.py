"""Console capture: browser console messages and uncaught page errors."""

from __future__ import annotations

from typing import Any

from tools.recorder.events import make_event
from tools.recorder.writer import SessionWriter


def on_console(writer: SessionWriter, message: Any) -> None:
    location = message.location or {}
    where = f"{location.get('url', '')}:{location.get('lineNumber', 0)}"
    writer.emit(
        make_event("console", {"level": message.type, "text": message.text, "location": where})
    )


def on_page_error(writer: SessionWriter, error: Any) -> None:
    writer.emit(make_event("console", {"level": "error", "text": str(error), "location": ""}))
