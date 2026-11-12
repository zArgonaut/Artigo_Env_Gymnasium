"""Compiled corpus -> engine ``Article``: equivalence with articles built from raw text."""

from __future__ import annotations

import unicodedata
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from artigo_env.core.errors import IncompatibleRulesetError
from artigo_env.corpus.store import CorpusStore, CorpusWriter, StoreLexicon
from artigo_env.corpus.types import DOC_SCHEMA, Row
from artigo_env.engine import ArticleBuilder, Engine
from artigo_env.engine.rulesets import Ruleset
from artigo_env.engine.types import Article
from artigo_env.text.lemmatizers import get_lemmatizer

pytestmark = pytest.mark.fast

TEXTS: list[tuple[str, str]] = [
    (
        "História do Brasil",
        "A história do Brasil tem muitos anos. O primeiro ano foi difícil.\n\n"
        "Em 1822 o país se tornou independente, e 19 províncias se tornaram estados. "
        "O 1º imperador teve apoio norte-americano. Cerca de 1.500 famílias chegaram.\n\n"
        "== Economia ==\nO café nasceu forte (a primeira do país) no sul.",
    ),
    (
        "Gato-do-mato (felino)",
        "O gato-do-mato (Leopardus tigrinus, ou (às vezes) gato) vive na Mata Atlântica).\n"
        "== Hábitos ==\nEm 1990, 12 gatos-do-mato foram vistos ) em Minas (e 3 no Rio.\n"
        "=== Dieta ===\nCome ratos e aves; o gato caça à noite.",
    ),
    (
        "ONU e as Nações Unidas",
        "A ONU (Organização das Nações Unidas) foi fundada em 1945. A ONU tem 193 membros.\n"
        "O McDonald e o iPhone são nomes MiStOs; a sigla UNESCO também é da ONU.\n\n"
        "== Agências ==\nAs Nações Unidas mantêm agências em Genebra e Nova Iorque.",
    ),
    (
        unicodedata.normalize("NFD", "Ação e reação"),
        unicodedata.normalize(
            "NFD",
            "A ação gera uma reação. Toda ação (física) tem reação igual.\r\n"
            "== Leis de Newton ==\r\nA 3ª lei vale desde 1687; π ≈ 3,14 não é token.",
        ),
    ),
    (
        "İstanbul e Ancara",
        "== Cidades ==\nİstanbul é a maior cidade; Ancara é a capital desde 1923.\n\n"
        "A cidade de İstanbul tem 15 milhões (aprox.) de habitantes e o Bósforo.",
    ),
    (
        "Rio Grande do Sul",
        "O Rio Grande do Sul é o estado sul-rio-grandense. Porto Alegre (capital) tem "
        "1,4 milhão de habitantes.\n\nO rio Guaíba banha a cidade -- e o lago-rio é grande.",
    ),
]

IDENTITY = Ruleset(name="store-identity", lemmatizer="identity")


def make_row(i: int, title: str | None, body: str) -> Row:
    row: Row = dict.fromkeys(DOC_SCHEMA.names)
    row.update(
        source_id="synthetic",
        doc_id=f"doc-{i}",
        title=title,
        body=body,
        license_id="MIT",
        provenance="{}",
        split="train" if i % 2 == 0 else "ext-val",
    )
    return row


def build(out: Path, texts: list[tuple[str, str]], hyphen: str = "split") -> CorpusStore:
    writer = CorpusWriter(out, tokenizer_hyphen=hyphen)
    for i, (title, body) in enumerate(texts):
        writer.add(make_row(i, title, body))
    writer.add_lemma_view(get_lemmatizer("identity"))
    writer.add_lemma_view(get_lemmatizer("test-protocol"))
    writer.finish({"build_id": "synthetic-build"})
    return CorpusStore.open(out)


@pytest.fixture
def store(tmp_path):
    opened = build(tmp_path / "corpus", TEXTS)
    yield opened
    opened.close()


def lemma_texts(article: Article, lexicon) -> list[str]:
    return [lexicon.lemma_text(int(lemma)) for lemma in article.lemmas]


def grouping(article: Article, lexicon) -> dict[str, tuple[int, ...]]:
    return {
        lexicon.lemma_text(key): tuple(positions.tolist())
        for key, positions in article.lemma_index.items()
    }


def assert_equivalent(stored: Article, stored_lex, built: Article, built_lex) -> None:
    assert stored.title.surface == built.title.surface
    assert stored.body.surface == built.body.surface
    for name in ("flags", "stop", "lengths"):
        np.testing.assert_array_equal(getattr(stored, name), getattr(built, name), err_msg=name)
        assert getattr(stored, name).dtype == getattr(built, name).dtype
    np.testing.assert_array_equal(stored.digit_positions, built.digit_positions)
    assert lemma_texts(stored, stored_lex) == lemma_texts(built, built_lex)
    assert grouping(stored, stored_lex) == grouping(built, built_lex)


@pytest.mark.req("REQ-COR-014", "REQ-TXT-017")
@pytest.mark.parametrize("which", ["identity", "protocol"])
def test_round_trip_matches_article_builder(store, protocol_ruleset, which):
    ruleset = IDENTITY if which == "identity" else protocol_ruleset
    lexicon = StoreLexicon(store, store.lemma_view(ruleset.lemmatizer))
    for i, (title, body) in enumerate(TEXTS):
        stored = store.article(i, ruleset)
        built, built_lex = ArticleBuilder(ruleset).from_text(title, body)
        assert_equivalent(stored, lexicon, built, built_lex)


@pytest.mark.req("REQ-COR-014")
def test_round_trip_protocol_text(tmp_path, protocol_ruleset, protocol_text):
    title, body = protocol_text
    with build(tmp_path / "corpus", [protocol_text]) as stored_corpus:
        lexicon = StoreLexicon(stored_corpus, stored_corpus.lemma_view(protocol_ruleset.lemmatizer))
        built, built_lex = ArticleBuilder(protocol_ruleset).from_text(title, body)
        assert_equivalent(stored_corpus.article(0, protocol_ruleset), lexicon, built, built_lex)


@pytest.mark.req("REQ-COR-014")
def test_round_trip_join_build(tmp_path, protocol_ruleset):
    ruleset = replace(protocol_ruleset, tokenizer_hyphen="join")
    with build(tmp_path / "corpus", TEXTS, hyphen="join") as joined:
        lexicon = StoreLexicon(joined, joined.lemma_view(ruleset.lemmatizer))
        stored = joined.article(5, ruleset)
        assert "sul-rio-grandense" in stored.surfaces
        built, built_lex = ArticleBuilder(ruleset).from_text(*TEXTS[5])
        assert_equivalent(stored, lexicon, built, built_lex)


@pytest.mark.req("REQ-COR-014")
def test_article_ref_is_build_and_doc_id(store):
    from artigo_env.core.config import sha256_hex

    assert store.article(2, IDENTITY).article_ref == sha256_hex("synthetic-build:doc-2")


@pytest.mark.req("REQ-COR-014")
def test_lengths_follow_the_spelling(store):
    """ "İ" lowercases to two characters; the length shown is the spelling's (one)."""
    article = store.article(4, IDENTITY)
    pos = article.surfaces.index("İstanbul")
    assert article.lengths[pos] == len("İstanbul") == 8


def _guesses(article: Article) -> list[str]:
    words = sorted({surface.lower() for surface in article.surfaces})
    extra = ["xyzzy", "o", "de", "1", "19", "9", "brasil", "BRASIL", "a-b", ""]
    return extra[:6] + words[: len(words) // 2] + extra[6:] + words[len(words) // 2 :]


@pytest.mark.req("REQ-ENG-003", "REQ-COR-014")
@pytest.mark.parametrize("which", ["identity", "protocol"])
def test_game_outcomes_match_in_memory_article(store, protocol_ruleset, which):
    ruleset = IDENTITY if which == "identity" else protocol_ruleset
    store_engine = Engine(ruleset, StoreLexicon(store, store.lemma_view(ruleset.lemmatizer)))
    for i, (title, body) in enumerate(TEXTS):
        stored = store.article(i, ruleset)
        built, built_lex = ArticleBuilder(ruleset).from_text(title, body)
        mem_engine = Engine(ruleset, built_lex)
        s_state, m_state = store_engine.new_game(stored), mem_engine.new_game(built)
        np.testing.assert_array_equal(s_state.revealed, m_state.revealed)
        for raw in _guesses(built):
            if m_state.status == "won":
                break
            s_state, s_out = store_engine.apply(s_state, stored, store_engine.parse_guess(raw))
            m_state, m_out = mem_engine.apply(m_state, built, mem_engine.parse_guess(raw))
            assert (s_out.kind, s_out.counted, s_out.count, s_out.won) == (
                m_out.kind,
                m_out.counted,
                m_out.count,
                m_out.won,
            ), raw
            np.testing.assert_array_equal(s_out.newly_revealed, m_out.newly_revealed)
            assert dict(s_out.newly_partial) == dict(m_out.newly_partial)
            assert s_out.title_progress == m_out.title_progress
            np.testing.assert_array_equal(s_state.revealed, m_state.revealed)
            assert s_state.n_counted == m_state.n_counted
        assert s_state.status == m_state.status == "won"


@pytest.mark.req("REQ-RUL-004")
def test_split_build_refuses_join_ruleset(store, protocol_ruleset):
    with pytest.raises(IncompatibleRulesetError, match="join"):
        store.article(0, replace(protocol_ruleset, tokenizer_hyphen="join"))
