"""localStorage capture (read-only ``evaluate``); sensitive keys are redacted by make_event."""

from __future__ import annotations

from typing import Any

from tools.recorder.events import make_event
from tools.recorder.writer import SessionWriter

_DUMP = """() => {
  try {
    const items = {};
    for (let i = 0; i < localStorage.length; i += 1) {
      const key = localStorage.key(i);
      items[key] = localStorage.getItem(key);
    }
    return items;
  } catch (error) {
    return {};
  }
}"""


async def capture_storage(writer: SessionWriter, page: Any, reason: str) -> None:
    """Emit one ``storage`` event with the current localStorage contents."""
    items = await page.evaluate(_DUMP)
    writer.emit(make_event("storage", {"items": items, "reason": reason}))
