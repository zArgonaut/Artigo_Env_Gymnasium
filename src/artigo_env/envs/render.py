"""Plain-text rendering of a ``ScreenView`` (SDD 09 §7, DEC-071).

Rendering reads the screen view only, so it cannot show anything a player must not see. The
environment's ``ansi`` mode and the ``artigo-env play`` CLI share these functions (the CLI
re-exports them from ``artigo_env.cli.render``).
"""

from __future__ import annotations

from collections.abc import Sequence

from artigo_env.engine.screen import HIDDEN_CHAR, ScreenToken, ScreenView
from artigo_env.engine.types import FLAG_HEADER, FLAG_PARAGRAPH_BREAK, GuessRecord, Outcome

__all__ = ["describe_outcome", "describe_record", "render_ansi", "render_token"]

_HISTORY_SHOWN = 12


def render_token(token: ScreenToken) -> str:
    """Visible spelling, partial reveal, or one ``█`` per hidden character.

    A hidden token whose length the ruleset hides is a single ``█``.
    """
    if token.text is not None:
        return token.text
    if token.partial is not None:
        return token.partial
    return HIDDEN_CHAR * (token.length if token.length is not None else 1)


def _render_tokens(tokens: Sequence[ScreenToken]) -> str:
    lines: list[str] = []
    current: list[str] = []
    in_header = False

    def flush() -> None:
        if current:
            text = " ".join(current)
            lines.append(f"== {text} ==" if in_header else text)
            current.clear()

    for token in tokens:
        header = bool(token.flags & FLAG_HEADER)
        if current and (header != in_header or token.flags & FLAG_PARAGRAPH_BREAK):
            flush()
            if not header and not in_header:
                lines.append("")
        in_header = header
        current.append(render_token(token))
    flush()
    return "\n".join(lines)


def describe_record(record: GuessRecord) -> str:
    """One history line: the guess as typed and what it did."""
    if record.kind == "noop_stopword":
        return f"{record.form}: not counted (stopword)"
    if record.kind == "noop_repeat":
        return f"{record.form}: already guessed"
    return f"{record.form}: {record.count}"


def describe_outcome(outcome: Outcome, form: str) -> str:
    """Message for the guess just applied."""
    kind = outcome.kind.value
    if kind == "invalid":
        return "invalid guess"
    if kind == "noop_stopword":
        return f'"{form}" is a stopword: not counted'
    if kind == "noop_repeat":
        return f'"{form}" was already guessed'
    if kind == "miss":
        return f'"{form}": no occurrences'
    plural = "occurrence" if outcome.count == 1 else "occurrences"
    return f'"{form}": {outcome.count} {plural}'


def render_ansi(view: ScreenView, message: str | None = None) -> str:
    """The whole screen as text: title, body, counter, recent history and a message."""
    counted = [record for record in view.history if record.counted]
    parts = [
        _render_tokens(view.title),
        "",
        _render_tokens(view.body),
        "",
        f"Guesses: {view.n_counted}" + ("  (solved)" if view.status == "won" else ""),
    ]
    if counted:
        recent = counted[-_HISTORY_SHOWN:]
        parts.append("History: " + "; ".join(describe_record(record) for record in recent))
    if message:
        parts.append(message)
    return "\n".join(parts)
