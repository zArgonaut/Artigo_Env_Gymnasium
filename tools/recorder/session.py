"""Passive recording session (SDD 03 §4): observe the game, never act on it (REQ-REC-001).

The only Playwright calls made here are those allowed by the plan: ``launch_persistent_context``,
``pages``/``new_page``, a single ``goto`` of the start URL, ``add_init_script``,
``expose_binding``, ``on``, read-only ``evaluate``, ``screenshot`` and ``close``.
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import json
import re
from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any

from tools.recorder import selectors
from tools.recorder.capture.console import on_console, on_page_error
from tools.recorder.capture.network import NetworkCapture
from tools.recorder.capture.snapshots import capture_all, wait_for_container
from tools.recorder.events import make_event
from tools.recorder.writer import SessionWriter

INJECT_JS = Path(__file__).parent / "capture" / "inject.js"
BINDING_NAME = "__artigoRecorder"
PAGE_EVENTS = frozenset(
    {"input.click", "input.key", "input.submit", "dom.mutation", "dom.quiescent"}
)
_CHROME_VERSION = re.compile(r"Chrome/([\d.]+)")


@dataclass(frozen=True)
class RecorderConfig:
    mode: str
    start_url: str
    out_root: Path
    profile_dir: Path
    allowed_hosts: tuple[str, ...] = ("artigo.app",)
    screenshots: bool = True
    quiescence_ms: int = 300
    headless: bool = False


def build_init_script(config: RecorderConfig) -> str:
    """The injected script: a JSON config prefix followed by ``inject.js``."""
    page_config = {
        "gameContainer": selectors.GAME_CONTAINER,
        "guessInput": selectors.GUESS_INPUT,
        "guessForm": selectors.GUESS_FORM,
        "sendButton": selectors.SEND_BUTTON,
        "quiescenceMs": config.quiescence_ms,
    }
    prefix = f"window.__artigoRecorderConfig = {json.dumps(page_config)};\n"
    return prefix + INJECT_JS.read_text(encoding="utf-8")


def run_session(config: RecorderConfig, on_ready: Callable[[Any], Any] | None = None) -> Path:
    """Record one session and return its folder.

    Without ``on_ready`` the session lasts until the user closes the window or presses Ctrl+C.
    With ``on_ready`` (tests only) the callback receives the Playwright *async* ``Page`` once the
    initial snapshot is taken, may be a plain function or a coroutine function, and the session
    is finalized as soon as it returns. The recorder itself stays passive: any typing is done by
    the callback, through its own use of the page object.
    """
    writer = SessionWriter(config.out_root, config.mode)
    with contextlib.suppress(KeyboardInterrupt):  # _record already finalized the session
        asyncio.run(_record(config, writer, on_ready))
    return writer.path


async def _record(
    config: RecorderConfig,
    writer: SessionWriter,
    on_ready: Callable[[Any], Any] | None,
) -> None:
    from playwright.async_api import async_playwright

    tasks: set[asyncio.Task[Any]] = set()
    capture_lock = asyncio.Lock()
    first_quiet = asyncio.Event()
    closed = asyncio.Event()
    guesses = 0
    awaiting_quiet = False
    network = NetworkCapture(writer, config.allowed_hosts)

    def spawn(coro: Coroutine[Any, Any, Any]) -> None:
        task = asyncio.ensure_future(coro)
        tasks.add(task)
        task.add_done_callback(tasks.discard)

    async def capture(page: Any, reason: str) -> None:
        async with capture_lock:
            # The page may go away mid-capture (window closed): best effort.
            with contextlib.suppress(Exception):
                await capture_all(writer, page, reason, screenshots=config.screenshots)

    def on_binding(source: Any, event_type: str, payload: Any) -> None:
        nonlocal guesses, awaiting_quiet
        if event_type not in PAGE_EVENTS:
            return
        writer.emit(make_event(event_type, payload if isinstance(payload, dict) else {}))
        if event_type == "input.submit":
            guesses += 1
            awaiting_quiet = True
        elif event_type == "dom.quiescent":
            first_quiet.set()
            if awaiting_quiet:
                awaiting_quiet = False
                spawn(capture(source["page"], "guess"))

    reason = "window-closed"
    async with async_playwright() as playwright:
        context = await playwright.chromium.launch_persistent_context(
            str(config.profile_dir), headless=config.headless
        )
        try:
            await context.expose_binding(BINDING_NAME, on_binding)
            await context.add_init_script(script=build_init_script(config))
            context.on("request", network.on_request)
            context.on("response", lambda response: spawn(network.on_response(response)))
            context.on("console", lambda message: on_console(writer, message))
            context.on("weberror", lambda web_error: on_page_error(writer, web_error.error))
            context.on("close", lambda _: closed.set())
            context.on("page", lambda page: page.on("close", lambda _: _maybe_closed()))

            def _maybe_closed() -> None:
                if not [p for p in context.pages if not p.is_closed()]:
                    closed.set()

            page = context.pages[0] if context.pages else await context.new_page()
            agent = await page.evaluate("() => navigator.userAgent")
            found = _CHROME_VERSION.search(agent)
            writer.write_meta(
                playwright_version=metadata.version("playwright"),
                browser_version=found.group(1) if found else None,
            )
            writer.emit(make_event("session.start", {"url": config.start_url, "mode": config.mode}))
            await page.goto(config.start_url)

            await wait_for_container(page)
            # Wait for a quiet period after the article was rendered, not for an earlier one.
            first_quiet.clear()
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(first_quiet.wait(), timeout=5.0)
            await capture(page, "load")

            try:
                if on_ready is not None:
                    outcome = on_ready(page)
                    if inspect.isawaitable(outcome):
                        await outcome
                else:
                    await closed.wait()
            except (asyncio.CancelledError, KeyboardInterrupt):
                reason = "interrupted"

            while tasks:
                await asyncio.gather(*list(tasks), return_exceptions=True)
            live = [p for p in context.pages if not p.is_closed()]
            if live:
                await capture(live[-1], "end")
            writer.emit(make_event("third_party.count", {"hosts": network.third_party_counts()}))
            writer.emit(make_event("session.end", {"reason": reason}))
            writer.close(guesses)
        finally:
            with contextlib.suppress(Exception):
                await context.close()
