"""Parsing of game snapshots into token elements and the guess list (SDD 03 §7.2)."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from html.parser import HTMLParser
from typing import Literal

from tools.recorder import selectors

_VOID = frozenset(
    {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "path", "source"}
)
_TOKEN_ID = re.compile(
    rf"^(?:{re.escape(selectors.TOKEN_ID_TITLE_PREFIX)}(?P<title>\d+)"
    rf"|{re.escape(selectors.TOKEN_ID_TEXT_PREFIX)}(?P<text>\d+))$"
)
_HEADER = re.compile(rf"^\s*{selectors.GUESS_HEADER_PATTERN}\s*$")

GuessStatus = Literal["found", "not-found", "warning"]
_STATUS_CLASSES: dict[str, GuessStatus] = {
    selectors.GUESS_FOUND_CLASS: "found",
    selectors.GUESS_NOT_FOUND_CLASS: "not-found",
    selectors.GUESS_WARNING_CLASS: "warning",
}


@dataclass(frozen=True)
class DomToken:
    """One token element; ``index`` is its position in the unified title+body order.

    ``offset`` is the character offset encoded in the element id (relative to the title for
    title tokens, to the body for body tokens) and ``matched`` tells that the element carries
    the class of the latest counted guess. ``revealed`` is true only for a fully visible token;
    a token partly shown by a numeric guess (for example ``1██0``; the game drops the hidden
    class for it) has ``revealed`` false and ``mask`` with a bit per visible character.
    """

    index: int
    length: int
    text: str | None
    revealed: bool
    in_title: bool
    header: bool
    offset: int | None = None
    matched: bool = False
    mask: int = 0


@dataclass(frozen=True)
class GuessEntry:
    """One guess row: what the player typed (as shown), its status and the right-hand text."""

    label: str
    status: GuessStatus
    value: str
    selected: bool

    @property
    def count(self) -> int | None:
        """Number shown on the right (matching tokens); ``None`` for warnings."""
        return int(self.value) if self.value.isdecimal() else None


@dataclass(frozen=True)
class GuessList:
    """The (virtualized) guess list: header total and the rows present in the DOM."""

    total: int | None
    entries: tuple[GuessEntry, ...]

    @property
    def newest(self) -> GuessEntry | None:
        """The first row after the header: the latest guess or warning."""
        return self.entries[0] if self.entries else None


@dataclass
class _Frame:
    tag: str
    token: int | None = None
    row: int | None = None
    side: str | None = None


@dataclass
class _Record:
    offset: int
    hidden: bool
    matched: bool
    length: int | None
    in_title: bool
    header: bool
    parts: list[str] = field(default_factory=list)


@dataclass
class _Row:
    parts: list[str] = field(default_factory=list)
    left: list[str] = field(default_factory=list)
    right: list[str] = field(default_factory=list)
    status: GuessStatus | None = None
    selected: bool = False


class _Parser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.records: list[_Record] = []
        self.rows: list[_Row] = []
        self._stack: list[_Frame] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in _VOID:
            return
        values = {name: (value or "") for name, value in attrs}
        classes = values.get("class", "").split()
        frame = _Frame(tag)
        if tag == selectors.TOKEN_TAG and selectors.TOKEN_CLASS in classes:
            found = _TOKEN_ID.match(values.get("id", ""))
            if found is not None:
                in_title = found.group("title") is not None
                raw_length = values.get(selectors.TOKEN_LENGTH_ATTR, "")
                self.records.append(
                    _Record(
                        offset=int(found.group("title" if in_title else "text")),
                        hidden=selectors.TOKEN_HIDDEN_CLASS in classes,
                        matched=selectors.TOKEN_MATCH_CLASS in classes,
                        length=int(raw_length) if raw_length.isdecimal() else None,
                        in_title=in_title,
                        header=any(f.tag in selectors.HEADING_TAGS for f in self._stack),
                    )
                )
                frame.token = len(self.records) - 1
        if tag == selectors.GUESS_ROW_TAG:
            self.rows.append(_Row())
            frame.row = len(self.rows) - 1
        self._open_row(frame, classes)
        self._stack.append(frame)

    def _open_row(self, frame: _Frame, classes: list[str]) -> None:
        row = next((self.rows[f.row] for f in reversed(self._stack) if f.row is not None), None)
        if frame.row is not None:
            row = self.rows[frame.row]
        if row is None:
            return
        if selectors.GUESS_SELECTED_CLASS in classes:
            row.selected = True
        for name, status in _STATUS_CLASSES.items():
            if name in classes:
                row.status = status
        if selectors.GUESS_LEFT_CLASS in classes:
            frame.side = "left"
        elif selectors.GUESS_RIGHT_CLASS in classes:
            frame.side = "right"

    def handle_endtag(self, tag: str) -> None:
        if tag in _VOID:
            return
        for position in range(len(self._stack) - 1, -1, -1):
            if self._stack[position].tag == tag:
                del self._stack[position:]
                return

    def handle_data(self, data: str) -> None:
        for frame in self._stack:
            if frame.token is not None:
                self.records[frame.token].parts.append(data)
            if frame.row is not None:
                self.rows[frame.row].parts.append(data)
            if frame.side is not None:
                row = next(self.rows[f.row] for f in reversed(self._stack) if f.row is not None)
                (row.left if frame.side == "left" else row.right).append(data)


def _parse(snapshot_html: str) -> _Parser:
    parser = _Parser()
    parser.feed(snapshot_html)
    parser.close()
    return parser


def dom_tokens(snapshot_html: str) -> list[DomToken]:
    """Token elements in document order; hidden ones carry no text."""
    tokens: list[DomToken] = []
    for index, record in enumerate(_parse(snapshot_html).records):
        text = None if record.hidden else "".join(record.parts)
        length = record.length if record.length is not None else len(text or "")
        if text is None:
            mask = 0
        else:
            mask = sum(1 << i for i, char in enumerate(text) if char != selectors.HIDDEN_CHAR)
        full = text is not None and mask == (1 << len(text)) - 1
        tokens.append(
            DomToken(
                index=index,
                length=length,
                text=text,
                revealed=full,
                in_title=record.in_title,
                header=record.header,
                offset=record.offset,
                matched=record.matched,
                mask=mask,
            )
        )
    return tokens


def read_guess_list(snapshot_html: str) -> GuessList:
    """Header total and guess rows present in the snapshot (the list is virtualized, so rows
    scrolled out of view are absent; the header total still counts every counted guess)."""
    total: int | None = None
    entries: list[GuessEntry] = []
    for row in _parse(snapshot_html).rows:
        if row.status is not None:
            entries.append(
                GuessEntry(
                    label="".join(row.left).strip(),
                    status=row.status,
                    value="".join(row.right).strip(),
                    selected=row.selected,
                )
            )
            continue
        header = _HEADER.match(" ".join("".join(row.parts).split()))
        if header is not None:
            total = int(header.group(1))
    return GuessList(total, tuple(entries))
