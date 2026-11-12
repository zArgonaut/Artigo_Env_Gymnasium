from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from artigo_env.core.errors import UnplayableArticleError
from artigo_env.engine import ArticleBuilder, Engine, InMemoryLexicon
from artigo_env.engine.types import (
    FLAG_HAS_DIGIT,
    FLAG_HEADER,
    FLAG_INSIDE_PARENTHESES,
    FLAG_PARAGRAPH_BREAK,
)

pytestmark = pytest.mark.fast

BODY = "O gato (animal doméstico) dorme.\n\n== Hábitos ==\nEm 1990 o gato caçava ratos."


@pytest.fixture
def article(protocol_ruleset):
    built, _ = ArticleBuilder(protocol_ruleset).from_text("Gato", BODY)
    return built


def _pos(article, word: str) -> int:
    return article.surfaces.index(word)


@pytest.mark.req("REQ-ENG-004")
def test_positions_are_unified_title_then_body(article):
    assert article.n_title == 1
    assert article.n_total == len(article.surfaces)
    assert article.surface_at(0) == "Gato"
    assert article.surface_at(1) == "O"
    assert article.stop.tolist()[:3] == [False, True, False]


@pytest.mark.req("REQ-ENG-004")
def test_flags(article):
    flags = article.flags
    assert flags[_pos(article, "animal")] & FLAG_INSIDE_PARENTHESES
    assert flags[_pos(article, "doméstico")] & FLAG_INSIDE_PARENTHESES
    assert not flags[_pos(article, "dorme")] & FLAG_INSIDE_PARENTHESES
    assert flags[_pos(article, "Hábitos")] & FLAG_HEADER
    assert flags[_pos(article, "Hábitos")] & FLAG_PARAGRAPH_BREAK
    assert flags[_pos(article, "Em")] & FLAG_PARAGRAPH_BREAK
    assert not flags[_pos(article, "dorme")] & FLAG_PARAGRAPH_BREAK
    assert flags[_pos(article, "1990")] & FLAG_HAS_DIGIT
    assert article.digit_positions.tolist() == [_pos(article, "1990")]
    assert article.lengths[_pos(article, "doméstico")] == 9


@pytest.mark.req("REQ-ENG-003")
def test_lemma_index_groups_title_and_body(article):
    lemma_id = int(article.lemmas[0])
    assert article.lemma_index[lemma_id].tolist() == [
        pos for pos, surface in enumerate(article.surfaces) if surface.lower() == "gato"
    ]
    plural = int(article.lemmas[_pos(article, "ratos")])
    assert article.lemma_index[plural].tolist() == [_pos(article, "ratos")]


@pytest.mark.req("REQ-ENG-003", "DEC-110")
def test_lemma_index_skips_stopword_tokens(article):
    indexed = {int(p) for positions in article.lemma_index.values() for p in positions}
    assert indexed == {pos for pos in range(article.n_total) if not article.stop[pos]}


@pytest.mark.req("REQ-ENG-002")
def test_arrays_are_read_only(article):
    with pytest.raises(ValueError):
        article.title.lemmas[0] = 5
    with pytest.raises(ValueError):
        article.stop[0] = True
    with pytest.raises(ValueError):
        next(iter(article.lemma_index.values()))[0] = 0


@pytest.mark.req("REQ-ENG-010")
@pytest.mark.parametrize("title", ["De", "o que é", "", "?!"])
def test_title_without_content_is_unplayable(protocol_ruleset, title):
    with pytest.raises(UnplayableArticleError):
        ArticleBuilder(protocol_ruleset).from_text(title, "Um texto qualquer.")


@pytest.mark.req("REQ-ENG-010")
def test_title_only_in_parentheses_is_unplayable_when_ignored(protocol_ruleset):
    ignored = replace(protocol_ruleset, title_parenthetical="ignored")
    with pytest.raises(UnplayableArticleError):
        ArticleBuilder(ignored).from_text("(Gato)", "Um texto qualquer.")
    article, _ = ArticleBuilder(protocol_ruleset).from_text("(Gato)", "Um texto qualquer.")
    assert article.n_title == 1


@pytest.mark.req("REQ-ENG-011")
def test_article_ref_is_a_content_hash_unless_given(protocol_ruleset):
    builder = ArticleBuilder(protocol_ruleset)
    first, _ = builder.from_text("Gato", BODY)
    second, _ = builder.from_text("Gato", BODY)
    other, _ = builder.from_text("Gato", BODY + " Fim.")
    named, _ = builder.from_text("Gato", BODY, article_ref="doc-1")
    assert first.article_ref == second.article_ref != other.article_ref
    assert len(first.article_ref) == 64
    assert named.article_ref == "doc-1"


@pytest.mark.req("REQ-ENG-003")
def test_shared_lexicon_keeps_ids_stable_across_articles(protocol_ruleset):
    builder = ArticleBuilder(protocol_ruleset)
    lexicon = InMemoryLexicon()
    first, _ = builder.from_text("Gato", "O gato dorme.", lexicon=lexicon)
    second, returned = builder.from_text("Rato", "O rato teme o gato.", lexicon=lexicon)
    assert returned is lexicon
    gato = lexicon.lemma_id("gato")
    assert gato == int(first.lemmas[0])
    assert gato in second.lemma_index
    assert lexicon.lemma_text(gato) == "gato"
    assert lexicon.lemma_id("zebra") == -1
    assert lexicon.form_id("gato") >= 0 and lexicon.form_id("zebra") == -1


@pytest.mark.req("REQ-TXT-009")
def test_hyphen_join_mode_builds_one_token(protocol_ruleset):
    joined = replace(protocol_ruleset, tokenizer_hyphen="join")
    article, lexicon = ArticleBuilder(joined).from_text("Guarda-chuva", "Um guarda-chuva azul.")
    assert article.surfaces[0] == "Guarda-chuva"
    assert np.count_nonzero(~article.stop) == 3
    # the guess loses its hyphen in the clean-up and must still match the joined token
    engine = Engine(joined, lexicon)
    state, outcome = engine.apply(
        engine.new_game(article), article, engine.parse_guess("guarda-chuva")
    )
    assert outcome.count == 2 and outcome.won
    assert engine.screen(state, article).title[0].text == "Guarda-chuva"
