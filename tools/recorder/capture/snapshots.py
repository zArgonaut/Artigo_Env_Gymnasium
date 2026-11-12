"""Game-container snapshots and screenshots (read-only ``evaluate`` / ``screenshot``)."""

from __future__ import annotations

import asyncio
from typing import Any

from tools.recorder.capture.storage import capture_storage
from tools.recorder.events import make_event
from tools.recorder.selectors import GAME_CONTAINER
from tools.recorder.writer import SessionWriter

_OUTER_HTML = (
    "(selector) => { const el = document.querySelector(selector); "
    "return el ? el.outerHTML : null; }"
)
_BODY_HTML = "() => (document.body ? document.body.outerHTML : '')"


async def wait_for_container(page: Any, timeout_s: float = 10.0) -> bool:
    """Poll (read-only) until the game container exists; False on timeout."""
    deadline = asyncio.get_running_loop().time() + timeout_s
    while True:
        if await page.evaluate("(s) => !!document.querySelector(s)", GAME_CONTAINER):
            return True
        if asyncio.get_running_loop().time() >= deadline:
            return False
        await asyncio.sleep(0.1)


async def capture_snapshot(writer: SessionWriter, page: Any, reason: str) -> None:
    """Emit ``snapshot`` with the container's outerHTML (whole body if the container is absent,
    which helps fixing the provisional selectors, LOCAL-REC-1)."""
    html = await page.evaluate(_OUTER_HTML, GAME_CONTAINER)
    if html is None:
        html = await page.evaluate(_BODY_HTML)
    # No await between reading next_seq and emitting: the file name matches the event seq.
    name = writer.save_snapshot(writer.next_seq, html)
    writer.emit(make_event("snapshot", {"file": name, "reason": reason}))


async def capture_screen(writer: SessionWriter, page: Any, reason: str) -> None:
    # caret="initial" keeps Playwright from hiding the text caret, which it does by changing the
    # focused input's style: the recorder must not mutate the page, not even transiently.
    png = await page.screenshot(caret="initial", animations="allow")
    name = writer.save_screen(writer.next_seq, png)
    writer.emit(make_event("screen", {"file": name, "reason": reason}))


async def capture_all(
    writer: SessionWriter, page: Any, reason: str, *, screenshots: bool = True
) -> None:
    """Snapshot, then storage, then screen: the order the contract guarantees."""
    await capture_snapshot(writer, page, reason)
    await capture_storage(writer, page, reason)
    if screenshots:
        await capture_screen(writer, page, reason)
