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


def _span(
    kind: str, token: Token, base: int, revealed: set[int], position: int, matched: bool
) -> str:
    offset = base + token.start
    classes = "token" + (" guess-match" if matched else "")
    if position in revealed:
        shown = token.text
    else:
        classes = "token obfuscated"
        shown = "█" * len(token.text)
    return (
        f'<span id="token-{kind}-{offset}" data-token-length="{len(token.text)}" '
        f'class="{classes}">{shown}</span>'
    )


def _line_html(
    line: str, base: int, start: int, revealed: set[int], matched: set[int]
) -> tuple[str, int]:
    tokens = tokenize(line)
    position = start
    spans = [
        _span("text", token, base, revealed, start + i, start + i in matched)
        for i, token in enumerate(tokens)
    ]
    if tokens and tokens[0].header:
        return f"<h2>{' '.join(spans)}</h2>", start + len(tokens)
    parts: list[str] = []
    cursor = 0
    for token, span in zip(tokens, spans, strict=True):
        parts.append(html.escape(line[cursor : token.start]))
        parts.append(span)
        position += 1
        cursor = token.end
    parts.append(html.escape(line[cursor:]))
    return f"<p>{''.join(parts)}</p>", position


def _style(top: int | None) -> str:
    if top is None:
        return ""
    return f' style="position: absolute; left: 0px; top: {top * 40}px; height: 40px; width: 100%;"'


def _row(label: str, status: str, value: str, selected: bool, top: int | None = None) -> str:
    inner = (
        f'<div class="{status} list-item-left">{html.escape(label)}</div>'
        f'<div class="{status} list-item-right">{html.escape(value)}</div>'
    )
    css = "list-item Mui-selected" if selected else "list-item"
    return f'<li{_style(top)}><div class="{css}">{inner}</div></li>'


def render_game(
    title: str,
    text: str,
    revealed: set[int],
    rows: list[tuple[str, int]],
    warning: tuple[str, str] | None = None,
    matched: set[int] | None = None,
    *,
    positioned: bool = False,
    scroll: int = 0,
) -> str:
    """The ``#root`` HTML in the real page's structure.

    ``rows`` are the counted guesses (label, shown count), newest first, prior history last;
    ``warning`` is the transient (label, message) row; ``matched`` the positions carrying
    ``guess-match``. ``positioned`` gives every row the absolute ``top`` of the virtualized list
    and ``scroll`` drops that many leading list items (header first), as a scrolled list does.
    """
    matched = matched or set()
    title_tokens = tokenize(title)
    title_html = " ".join(
        _span("title", token, 0, revealed, i, i in matched) for i, token in enumerate(title_tokens)
    )
    position = len(title_tokens)
    body: list[str] = []
    base = 0
    for line in text.split("\n"):
        rendered, position = _line_html(line, base, position, revealed, matched)
        body.append(rendered)
        base += len(line) + 1
    count = len(rows)
    header = f"{count} {'palpite' if count == 1 else 'palpites'}"
    items = [f"<li{_style(0 if positioned else None)}><span>{header}</span></li>"]
    if warning is not None:
        items.append(
            _row(warning[0], "guess-warning", warning[1], False, 1 if positioned else None)
        )
    first = len(items)
    for index, (label, shown) in enumerate(rows):
        status = "guess-found" if shown else "guess-not-found"
        top = first + index if positioned else None
        items.append(_row(label, status, str(shown), index == 0, top))
    items = items[scroll:]
    inner = "".join(items)
    if positioned:
        height = 40 * (count + 1 + (warning is not None))
        inner = f'<div style="height: {height}px; width: 100%;">{inner}</div>'
        listing = f'<section style="position: relative; height: 720px;">{inner}</section>'
    else:
        listing = f"<section><ul>{inner}</ul></section>"
    return (
        '<div id="root"><div id="single-mode">'
        f'<section id="app-text-section"><h1 id="article-title">{title_html}</h1>'
        f'<div id="article-text">{"".join(body)}</div></section>'
        '<div id="app-controls-section"><form><input id="guess-input">'
        '<button type="button" title="Enviar"></button></form>'
        f"{listing}</div>"
        "</div></div>"
    )


def token_ids(title: str, text: str) -> list[tuple[str, Token]]:
    """Element id and token of every unified position (title first)."""
    found = [(f"token-title-{token.start}", token) for token in tokenize(title)]
    base = 0
    for line in text.split("\n"):
        found += [(f"token-text-{base + token.start}", token) for token in tokenize(line)]
        base += len(line) + 1
    return found


def build_session(
    root: Path,
    guesses: list[Guess],
    *,
    title: str = TITLE,
    text: str = TEXT,
    mode: str = "free",
    complete: bool = True,
    positioned: bool = False,
    snapshots: str = "each",
    scroll_at: dict[int, int] | None = None,
    token_events: bool = False,
    prior: tuple[tuple[str, int], ...] = (),
) -> Path:
    """Record ``guesses`` through the real writer and return the session folder.

    ``positioned`` renders the list like the real virtualized one (rows with ``top``);
    ``snapshots="last"`` takes no snapshot per guess, only one after the last (a recorder that
    lagged behind the page); ``scroll_at`` maps a guess index to the number of leading list
    items missing from its snapshot; ``token_events`` makes the mutations carry the token ids
    and the shown text as the real page does; ``prior`` are the (label, count) rows the game
    already holds when the page loads (oldest last), stored in ``localStorage`` too.
    """
    writer = SessionWriter(root, mode, now=lambda: datetime(2026, 11, 12, 21, 3, 1))
    emitted = 0
    scroll_at = scroll_at or {}

    def emit(kind: str, payload: dict[str, object]) -> int:
        nonlocal emitted
        emitted += 1
        return writer.emit(make_event(kind, payload))

    def page_html(
        revealed: set[int],
        rows: list[tuple[str, int]],
        warning: tuple[str, str] | None,
        matched: set[int],
        scroll: int = 0,
    ) -> str:
        return render_game(
            title,
            text,
            revealed,
            [*rows, *prior],
            warning,
            matched,
            positioned=positioned,
            scroll=scroll,
        )

    writer.write_meta(bundle_hash="0a1b2c3d")
    emit("session.start", {"url": "https://artigo.app/", "mode": mode})
    body = json.dumps({"title": title, "text": text}, ensure_ascii=False).encode("utf-8")
    body_file = writer.save_body(emitted, body, "json")
    emit("net.response", {"url": ARTICLE_URL, "status": 200, "headers": {}, "body_file": body_file})

    revealed: set[int] = set()
    rows: list[tuple[str, int]] = []
    matched: set[int] = set()
    ids = token_ids(title, text)
    snapshot_seq = emitted
    emit(
        "snapshot",
        {
            "file": writer.save_snapshot(snapshot_seq, page_html(revealed, rows, None, matched)),
            "reason": "load",
        },
    )
    if prior:
        games = {
            mode: {"guesses": {label: {"matchCount": shown} for label, shown in reversed(prior)}}
        }
        state = json.dumps({"games": json.dumps({"current": None, "list": games})})
        emit("storage", {"items": {"persist:root": state}, "reason": "load"})

    def snapshot(index: int) -> None:
        page = page_html(revealed, rows, warning, matched, scroll_at.get(index, 0))
        seq = emitted
        emit("snapshot", {"file": writer.save_snapshot(seq, page), "reason": "guess"})

    warning: tuple[str, str] | None = None
    for index, guess in enumerate(guesses):
        emit("input.submit", {"value": guess.raw, "target": "#guess-form"})
        previous = matched
        matched = set(guess.reveal) if guess.counted else matched
        for position in guess.reveal:
            if token_events:
                token_id, token = ids[position]
                emit(
                    "dom.mutation",
                    {
                        "kind": "attributes",
                        "target": f"#{token_id}",
                        "attribute": "class",
                        "old": "token",
                        "new": "token guess-match",
                    },
                )
                emit(
                    "dom.mutation",
                    {
                        "kind": "childList",
                        "target": f"#{token_id}",
                        "added": [f"#text:{token.text}"],
                        "removed": ["#text:" + "█" * len(token.text)],
                    },
                )
            else:
                emit(
                    "dom.mutation",
                    {"kind": "attributes", "target": f"span.token:nth-of-type({position + 1})"},
                )
        if token_events and guess.counted:
            for position in previous - matched:
                emit(
                    "dom.mutation",
                    {
                        "kind": "attributes",
                        "target": f"#{ids[position][0]}",
                        "attribute": "class",
                        "old": "token guess-match",
                        "new": "token",
                    },
                )
        emit("dom.quiescent", {"after_ms": 300, "mutations": len(guess.reveal)})
        revealed.update(guess.reveal)
        warning = None
        if guess.counted:
            rows.insert(0, (guess.raw, len(guess.reveal)))
        else:
            warning = (guess.raw, guess.message or "não conta")
        if snapshots == "each":
            snapshot(index)
    if snapshots == "last":
        snapshot(len(guesses) - 1)
    if complete:
        emit("session.end", {"reason": "window-closed"})
        writer.close(len(guesses))
    return writer.path
