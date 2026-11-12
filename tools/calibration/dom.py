"""Parsing of ``#game`` snapshots into token elements (SDD 03 §7.2)."""

from __future__ import annotations

from dataclasses import dataclass, field
from html.parser import HTMLParser

from tools.recorder import selectors

_VOID = frozenset({"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta"})
_HEADING_TAGS = frozenset({"h2", "h3", "h4", "h5", "h6"})


def _id_of(selector: str) -> str:
    return selector.removeprefix("#")


@dataclass(frozen=True)
class DomToken:
    """One token element; ``index`` is its position in the unified title+body order."""

    index: int
    length: int
    text: str | None
    revealed: bool
    in_title: bool
    header: bool


@dataclass
class _Frame:
    tag: str
    token: int | None = None
    capture: str | None = None
    is_title: bool = False


@dataclass
class _Record:
    hidden: bool
    length: int | None
    in_title: bool
    header: bool
    parts: list[str] = field(default_factory=list)


class _Parser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.records: list[_Record] = []
        self.captured: dict[str, list[str]] = {"counter": [], "message": []}
        self._stack: list[_Frame] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in _VOID:
            return
        values = {name: (value or "") for name, value in attrs}
        classes = values.get("class", "").split()
        element_id = values.get("id", "")
        frame = _Frame(tag, is_title=element_id == _id_of(selectors.TITLE))
        if element_id == _id_of(selectors.COUNTER):
            frame.capture = "counter"
        elif element_id == _id_of(selectors.MESSAGE):
            frame.capture = "message"
        if tag == selectors.TOKEN_TAG and selectors.TOKEN_CLASS in classes:
            raw_length = values.get(selectors.TOKEN_LENGTH_ATTR, "")
            self.records.append(
                _Record(
                    hidden=selectors.TOKEN_HIDDEN_CLASS in classes,
                    length=int(raw_length) if raw_length.isdecimal() else None,
                    in_title=any(f.is_title for f in self._stack),
                    header=any(f.tag in _HEADING_TAGS for f in self._stack),
                )
            )
            frame.token = len(self.records) - 1
        self._stack.append(frame)

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
            if frame.capture is not None:
                self.captured[frame.capture].append(data)


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
        tokens.append(
            DomToken(
                index=index,
                length=length,
                text=text,
                revealed=not record.hidden,
                in_title=record.in_title,
                header=record.header,
            )
        )
    return tokens


def read_counter(snapshot_html: str) -> int | None:
    """Integer shown by the guess counter, or ``None`` when absent or not a number."""
    value = "".join(_parse(snapshot_html).captured["counter"]).strip()
    return int(value) if value.isdecimal() else None


def read_message(snapshot_html: str) -> str | None:
    """Message shown by the game; empty or absent becomes ``None``."""
    value = "".join(_parse(snapshot_html).captured["message"]).strip()
    return value or None
