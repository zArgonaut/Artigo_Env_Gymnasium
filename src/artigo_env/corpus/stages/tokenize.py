"""S4 tokenize: token counts with the ``text`` core (SDD 06 §3; tokens are not persisted)."""

from __future__ import annotations

import pyarrow as pa

from artigo_env.corpus.stages.context import check_batch, finish
from artigo_env.corpus.types import StageContext, StageResult
from artigo_env.text.normalize import lower
from artigo_env.text.tokenizer import tokenize

__all__ = ["TokenizeStage", "count_tokens"]


def count_tokens(title: str, body: str, ctx: StageContext) -> tuple[int, int, int]:
    """``(n_body_tokens, n_title_tokens, n_title_content)`` of one document."""
    title_tokens = tokenize(title, ctx.tokenizer_hyphen)
    content = sum(1 for tok in title_tokens if not ctx.is_stopword(lower(tok.text)))
    return len(tokenize(body, ctx.tokenizer_hyphen)), len(title_tokens), content


class TokenizeStage:
    """Fills ``n_body_tokens``, ``n_title_tokens`` and ``n_title_content``; removes nothing."""

    name = "tokenize"
    version = "1"

    def run(self, batch: pa.RecordBatch, ctx: StageContext) -> StageResult:
        check_batch(batch, self.name)
        n_body: list[int] = []
        n_title: list[int] = []
        n_content: list[int] = []
        titles = batch.column("title").to_pylist()
        bodies = batch.column("body").to_pylist()
        for title, body in zip(titles, bodies, strict=True):
            body_count, title_count, content_count = count_tokens(title or "", body or "", ctx)
            n_body.append(body_count)
            n_title.append(title_count)
            n_content.append(content_count)
        return finish(
            self.name,
            batch,
            [None] * batch.num_rows,
            {"n_body_tokens": n_body, "n_title_tokens": n_title, "n_title_content": n_content},
        )
