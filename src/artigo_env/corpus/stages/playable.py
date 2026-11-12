"""S5 playable: playability criteria (SDD 06 §5-§6, DEC-048).

REQ-COR-006 (configurable criteria), REQ-COR-017 (too-long bodies are removed, never truncated).
"""

from __future__ import annotations

import pyarrow as pa
import regex

from artigo_env.core.errors import ArtigoEnvError
from artigo_env.corpus.stages.context import check_batch, finish
from artigo_env.corpus.types import PlayabilityConfig, StageContext, StageResult
from artigo_env.text.guess import clean_guess
from artigo_env.text.normalize import lower
from artigo_env.text.tokenizer import tokenize

__all__ = ["PlayableStage", "title_in_body", "title_informative"]

_LETTER = regex.compile(r"\p{L}")
_EDGES = regex.compile(r"^[\W_]+|[\W_]+$")
_MONTH = (
    r"(?:janeiro|fevereiro|mar[çc]o|abril|maio|junho|julho|agosto|setembro|outubro|novembro"
    r"|dezembro)"
)
_WEEKDAY = r"(?:(?:segunda|ter[çc]a|quarta|quinta|sexta)(?:[- ]feira)?|s[áa]bado|domingo),?\s+"
_DAY = r"\d{1,2}(?:º|°|o)?"
_NOT_INFORMATIVE = tuple(
    regex.compile(pattern, regex.IGNORECASE)
    for pattern in (
        # numeric dates (normally already caught by the "no letter" rule)
        r"\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}",
        r"\d{4}[/.-]\d{1,2}[/.-]\d{1,2}",
        # textual dates: "12 de março de 2020", "12 de março", "março de 2020"
        rf"(?:{_WEEKDAY})?{_DAY}\s+de\s+{_MONTH}(?:\s+de\s+\d{{4}})?",
        rf"{_MONTH}\s+de\s+\d{{4}}",
        # placeholder titles, optionally numbered: "Sem título", "Untitled 3", "Post #12"
        r"(?:sem\s+t[íi]tulo|untitled|post)(?:\s*[-#nº°.:]*\s*\d+)?",
        # bare part markers: "Parte 2", "Capítulo IV", "Episódio nº 10"
        r"(?:parte|cap[íi]tulo|epis[óo]dio)\s*(?:n[º°.o]?\s*)?(?:\d+|[ivxlc]+)",
        # opaque identifiers (SDD 06 §5): hexadecimal ids and UUIDs
        r"(?=[0-9a-f]*[0-9])(?=[0-9a-f]*[a-f])[0-9a-f]{8,}",
        r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
    )
)


def title_informative(title: str) -> bool:
    """False for titles with no letter, only a date, a placeholder, a part marker or an id."""
    if _LETTER.search(title) is None:
        return False
    core = _EDGES.sub("", title.strip())
    return not any(pattern.fullmatch(core) for pattern in _NOT_INFORMATIVE)


def title_in_body(title: str, body: str, ctx: StageContext) -> bool:
    """Whether every content lemma of the title is a lemma of a non-stopword body token."""
    hyphen = ctx.tokenizer_hyphen
    missing: set[str] = set()
    for tok in tokenize(title, hyphen):
        form = lower(tok.text)
        if not ctx.is_stopword(form):
            missing.add(ctx.lemma(clean_guess(form)))
    for tok in tokenize(body, hyphen):
        if not missing:
            break
        form = lower(tok.text)
        if not ctx.is_stopword(form):
            missing.discard(ctx.lemma(clean_guess(form)))
    return not missing


def _required(value: int | None, column: str) -> int:
    if value is None:
        raise ArtigoEnvError(f"playable stage needs {column!r}; run the tokenize stage first")
    return value


def _reason(row_title: str, n_body: int, n_content: int, cfg: PlayabilityConfig) -> str | None:
    if not title_informative(row_title):
        return "title_not_informative"
    if not cfg.min_title_content <= n_content <= cfg.max_title_content:
        return "title_content_out_of_range"
    if n_body < cfg.min_body_tokens:
        return "body_too_short"
    if n_body > cfg.max_body_tokens:
        return "body_too_long"
    return None


class PlayableStage:
    """Removes unplayable documents in a fixed reason order; fills ``title_in_body``.

    With ``require_title_lemmas_in_body == "none"`` the title-in-body test only records its
    outcome in ``title_in_body`` instead of removing the document.
    """

    name = "playable"
    version = "1"

    def run(self, batch: pa.RecordBatch, ctx: StageContext) -> StageResult:
        check_batch(batch, self.name)
        cfg = ctx.playability
        require_all = cfg.require_title_lemmas_in_body == "all"
        titles = batch.column("title").to_pylist()
        bodies = batch.column("body").to_pylist()
        n_bodies = batch.column("n_body_tokens").to_pylist()
        n_contents = batch.column("n_title_content").to_pylist()
        in_body: list[bool | None] = []
        reasons: list[str | None] = []
        for title, body, n_body, n_content in zip(
            titles, bodies, n_bodies, n_contents, strict=True
        ):
            reason = _reason(
                title or "",
                _required(n_body, "n_body_tokens"),
                _required(n_content, "n_title_content"),
                cfg,
            )
            found: bool | None = None
            if reason is None:
                found = title_in_body(title or "", body or "", ctx)
                if require_all and not found:
                    reason = "title_not_in_body"
            in_body.append(found)
            reasons.append(reason)
        return finish(self.name, batch, reasons, {"title_in_body": in_body})
