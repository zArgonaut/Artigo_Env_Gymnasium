"""Recorder event model (SDD 03 §6): one JSON object per line in ``events.jsonl``."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from tools.recorder.redact import redact_headers, redact_mapping, redact_text, redact_url

SCHEMA_VERSION = 1

EVENT_TYPES = (
    "session.start",
    "session.end",
    "net.request",
    "net.response",
    "input.click",
    "input.key",
    "input.submit",
    "dom.mutation",
    "dom.quiescent",
    "snapshot",
    "console",
    "storage",
    "screen",
    "third_party.count",
)


@dataclass(frozen=True)
class RedactedEvent:
    """An event whose payload already went through redaction.

    Build instances only with :func:`make_event`; the writer refuses anything else.
    """

    type: str
    payload: Mapping[str, Any]


def _scrub(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _scrub_field(str(key), item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_scrub(item) for item in value]
    if isinstance(value, str):
        return redact_text(value)
    return value


def _scrub_field(key: str, value: Any) -> Any:
    if key == "headers" and isinstance(value, Mapping):
        return redact_headers({str(name): str(item) for name, item in value.items()})
    if key == "url" and isinstance(value, str):
        return redact_text(redact_url(value))
    if key == "items" and isinstance(value, Mapping):
        # localStorage contents: sensitive key names are redacted wholesale.
        return redact_mapping(value)
    return _scrub(value)


def make_event(event_type: str, payload: Mapping[str, Any]) -> RedactedEvent:
    """Redact ``payload`` in memory and wrap it; the only way to obtain a writable event."""
    if event_type not in EVENT_TYPES:
        raise ValueError(f"unknown event type {event_type!r}")
    return RedactedEvent(event_type, _scrub(payload))
