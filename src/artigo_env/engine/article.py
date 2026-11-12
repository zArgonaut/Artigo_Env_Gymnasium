"""Article construction from raw text (SDD 07 §1)."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from artigo_env.core.config import sha256_hex
from artigo_env.core.errors import UnplayableArticleError
from artigo_env.engine.lexicon import InMemoryLexicon, Lexicon
from artigo_env.engine.rulesets.components import TITLE_RULES
from artigo_env.engine.rulesets.ruleset import Ruleset
from artigo_env.engine.types import (
    FLAG_HAS_DIGIT,
    FLAG_HEADER,
    FLAG_INSIDE_PARENTHESES,
    FLAG_PARAGRAPH_BREAK,
    Article,
    TokenSeq,
)
from artigo_env.text.guess import clean_guess
from artigo_env.text.lemmatizers.registry import get_lemmatizer
from artigo_env.text.normalize import lower, normalize
from artigo_env.text.stopwords import load_stopwords
from artigo_env.text.tokenizer import tokenize

_DIGITS = frozenset("0123456789")


class ArticleBuilder:
    """Builds ``Article`` objects under one ruleset (stopwords, lemmatizer and hyphen mode)."""

    def __init__(self, ruleset: Ruleset) -> None:
        self.ruleset = ruleset
        self._stopwords = load_stopwords(ruleset.stopwords)
        self._lemmatizer = get_lemmatizer(ruleset.lemmatizer)

    def _sequence(self, text: str, lexicon: InMemoryLexicon) -> TokenSeq:
        text = normalize(text)
        tokens = tokenize(text, self.ruleset.tokenizer_hyphen)
        n = len(tokens)
        forms = np.zeros(n, dtype=np.uint32)
        lemmas = np.zeros(n, dtype=np.int64)
        lengths = np.zeros(n, dtype=np.uint16)
        flags = np.zeros(n, dtype=np.uint8)
        stop = np.zeros(n, dtype=np.bool_)
        depth = 0
        previous_end = 0
        for i, token in enumerate(tokens):
            gap = text[previous_end : token.start]
            depth = max(0, depth + gap.count("(") - gap.count(")"))
            flag = 0
            if token.header:
                flag |= FLAG_HEADER
            if i > 0 and "\n" in gap:
                flag |= FLAG_PARAGRAPH_BREAK
            if depth > 0:
                flag |= FLAG_INSIDE_PARENTHESES
            if not _DIGITS.isdisjoint(token.text):
                flag |= FLAG_HAS_DIGIT
            low = lower(token.text)
            forms[i] = lexicon.intern_form(low)
            # A joined compound keeps its hyphen on screen, but a guess never has one.
            lemmas[i] = lexicon.intern_lemma(self._lemmatizer.lemma(clean_guess(low)))
            lengths[i] = len(token.text)
            flags[i] = flag
            stop[i] = low in self._stopwords
            previous_end = token.end
        return TokenSeq(forms, lemmas, lengths, flags, stop, tuple(t.text for t in tokens))

    def from_text(
        self,
        title: str,
        body: str,
        *,
        lexicon: InMemoryLexicon | None = None,
        article_ref: str | None = None,
    ) -> tuple[Article, Lexicon]:
        """Tokenize ``title`` and ``body`` separately and index them.

        Raises ``UnplayableArticleError`` when no title token is required for victory.
        """
        lex = InMemoryLexicon() if lexicon is None else lexicon
        title_seq = self._sequence(title, lex)
        body_seq = self._sequence(body, lex)
        ref = article_ref if article_ref is not None else sha256_hex(title + "\n" + body)
        article = assemble_article(ref, title_seq, body_seq, self.ruleset.title_parenthetical)
        return article, lex


def assemble_article(
    article_ref: str, title: TokenSeq, body: TokenSeq, title_rule: str = "required"
) -> Article:
    """Index two token sequences into a playable ``Article``.

    Shared by every way of building an article (raw text here, the compiled corpus elsewhere).
    Raises ``UnplayableArticleError`` when no title token is required for victory under the
    ``title_rule`` lever.
    """
    required = TITLE_RULES.get(title_rule).required_mask(title)
    if not required.any():
        raise UnplayableArticleError("the title has no content token required for victory")
    lemmas = np.concatenate([title.lemmas, body.lemmas])
    stop = np.concatenate([title.stop, body.stop])
    flags = np.concatenate([title.flags, body.flags])
    return Article(
        article_ref,
        title,
        body,
        _lemma_index(np.where(stop, -1, lemmas)),
        np.flatnonzero((flags & FLAG_HAS_DIGIT) != 0).astype(np.int64),
    )


def _lemma_index(lemmas: npt.NDArray[np.int64]) -> dict[int, npt.NDArray[np.int64]]:
    """Group positions by lemma id in O(L log L); ``-1`` is not indexed.

    Stopword tokens come in as ``-1``: the game indexes content tokens only, so a guess whose
    lemma matches a stopword token neither counts nor hits it (DEC-110).
    """
    index: dict[int, npt.NDArray[np.int64]] = {}
    if len(lemmas) == 0:
        return index
    order = np.argsort(lemmas, kind="stable").astype(np.int64)
    sorted_ids = lemmas[order]
    starts = [0, *(np.flatnonzero(np.diff(sorted_ids)) + 1).tolist(), len(order)]
    ids = sorted_ids[starts[:-1]].tolist()
    # Plain slices of one sorted array: much cheaper than ``np.split`` for ~1,500 groups.
    for lemma_id, begin, end in zip(ids, starts[:-1], starts[1:], strict=True):
        if lemma_id >= 0:
            index[lemma_id] = order[begin:end]
    return index
