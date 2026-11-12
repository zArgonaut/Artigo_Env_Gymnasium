"""Builds synthetic recorded sessions on disk, following the recorder/calibration contract."""

from __future__ import annotations

import html
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from artigo_env.text.tokenizer import Token, tokenize
from tools.recorder.events import make_event
from tools.recorder.writer import SessionWriter

ARTICLE_URL = "https://artigo.app/articles/synthetic.json"

TITLE = "Rio Sintético"
TEXT = (
    "O rio sintético corre pelo vale antigo e passa por pontes de pedra.\n"
    "== História ==\n"
    "Os moradores do vale usam o rio para plantar milho e trigo todos os anos."
)


@dataclass(frozen=True)
class Guess:
    """A guess and what the (synthetic) game did with it."""

    raw: str
    counted: bool = True
    reveal: tuple[int, ...] = ()  # unified token positions revealed by this guess
    message: str | None = None


def _span(token: Token, position: int, revealed: set[int]) -> str:
    if position in revealed:
        return f'<span class="token" data-token-length="{len(token.text)}">{token.text}</span>'
    return f'<span class="token hidden" data-token-length="{len(token.text)}"></span>'


def _line_html(line: str, start: int, revealed: set[int]) -> tuple[str, int]:
    tokens = tokenize(line)
    position = start
    if tokens and tokens[0].header:
        spans = " ".join(_span(token, start + i, revealed) for i, token in enumerate(tokens))
        return f"<h2>{spans}</h2>", start + len(tokens)
    parts: list[str] = []
    cursor = 0
    for token in tokens:
        parts.append(html.escape(line[cursor : token.start]))
        parts.append(_span(token, position, revealed))
        position += 1
        cursor = token.end
    parts.append(html.escape(line[cursor:]))
    return f"<p>{''.join(parts)}</p>", position


def render_game(
    title: str, text: str, revealed: set[int], counter: int, message: str | None
) -> str:
    """The ``outerHTML`` of ``#game`` in the contract's structure."""
    title_tokens = tokenize(title)
    title_html = " ".join(_span(token, i, revealed) for i, token in enumerate(title_tokens))
    position = len(title_tokens)
    body: list[str] = []
    for line in text.split("\n"):
        rendered, position = _line_html(line, position, revealed)
        body.append(rendered)
    return (
        '<div id="game">'
        '<form id="guess-form"><input id="guess-input"><button type="submit">Enviar</button></form>'
        f'<span id="guess-count">{counter}</span>'
        f'<div id="message">{html.escape(message or "")}</div>'
        f'<div id="article"><h1 id="title">{title_html}</h1>{"".join(body)}</div>'
        "</div>"
    )


def build_session(
    root: Path,
    guesses: list[Guess],
    *,
    title: str = TITLE,
    text: str = TEXT,
    mode: str = "free",
    complete: bool = True,
) -> Path:
    """Record ``guesses`` through the real writer and return the session folder."""
    writer = SessionWriter(root, mode, now=lambda: datetime(2026, 11, 12, 21, 3, 1))
    emitted = 0

    def emit(kind: str, payload: dict[str, object]) -> int:
        nonlocal emitted
        emitted += 1
        return writer.emit(make_event(kind, payload))

    writer.write_meta(bundle_hash="0a1b2c3d")
    emit("session.start", {"url": "https://artigo.app/", "mode": mode})
    body = json.dumps({"title": title, "text": text}, ensure_ascii=False).encode("utf-8")
    body_file = writer.save_body(emitted, body, "json")
    emit("net.response", {"url": ARTICLE_URL, "status": 200, "headers": {}, "body_file": body_file})

    revealed: set[int] = set()
    counter = 0
    snapshot_seq = emitted
    emit(
        "snapshot",
        {
            "file": writer.save_snapshot(snapshot_seq, render_game(title, text, revealed, 0, None)),
            "reason": "load",
        },
    )
    for guess in guesses:
        emit("input.submit", {"value": guess.raw, "target": "#guess-form"})
        for position in guess.reveal:
            emit(
                "dom.mutation",
                {"kind": "attributes", "target": f"span.token:nth-of-type({position + 1})"},
            )
        emit("dom.quiescent", {"after_ms": 300, "mutations": len(guess.reveal)})
        revealed.update(guess.reveal)
        counter += int(guess.counted)
        page = render_game(title, text, revealed, counter, guess.message)
        snapshot_seq = emitted
        emit("snapshot", {"file": writer.save_snapshot(snapshot_seq, page), "reason": "guess"})
    if complete:
        emit("session.end", {"reason": "window-closed"})
        writer.close(len(guesses))
    return writer.path
