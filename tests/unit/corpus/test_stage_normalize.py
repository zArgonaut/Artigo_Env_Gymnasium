import sys
import unicodedata

import pytest

from artigo_env.core.errors import ArtigoEnvError
from artigo_env.corpus.stages import BuildContext, Cleaner, NormalizeStage
from artigo_env.corpus.stages import normalize as normalize_module
from artigo_env.corpus.stages.normalize import clean_body, clean_title, strip_portal_suffix
from artigo_env.corpus.types import DOC_SCHEMA, REASONS, PlayabilityConfig, to_batch, to_rows

pytestmark = pytest.mark.fast

CLEANER = Cleaner()


def row(i: int, title: str | None, body: str, domain: str | None = None) -> dict[str, object]:
    out: dict[str, object] = dict.fromkeys(DOC_SCHEMA.names)
    out.update(
        source_id="src",
        doc_id=f"d{i}",
        title=title,
        body=body,
        license_id="unknown",
        provenance="{}",
        url_domain=domain,
    )
    return out


def run(rows, stage=None):
    ctx = BuildContext(PlayabilityConfig(), lemmatizer="identity")
    result = (stage or NormalizeStage()).run(to_batch(rows), ctx)
    assert result.batch.schema.equals(DOC_SCHEMA)
    assert set(result.removed) <= set(REASONS["normalize"])
    assert sum(result.removed.values()) == len(rows) - result.batch.num_rows
    return result


@pytest.mark.req("REQ-COR-004")
def test_mojibake_is_repaired():
    assert clean_body("NINGUÃ‰M sabe", CLEANER) == "NINGUÉM sabe"
    assert clean_body("aÃ§Ã£o", CLEANER) == "ação"
    assert clean_title("A aÃ§Ã£o", None, CLEANER) == "A ação"


@pytest.mark.req("REQ-COR-004")
def test_decomposed_text_becomes_nfc():
    decomposed = unicodedata.normalize("NFD", "ação e coração")
    assert decomposed != "ação e coração"
    out = clean_body(decomposed, CLEANER)
    assert out == "ação e coração"
    assert unicodedata.is_normalized("NFC", out)


@pytest.mark.req("REQ-COR-004")
def test_quotes_whitespace_punctuation_lines_and_breaks():
    body = (
        "“Aspas” e ‘simples’ «angulares»\r\n"
        "  muitos\t\t espaços aqui  \n"
        "***\n"
        "\n\n\n\n"
        "== Seção ==\n"
        "fim"
    )
    assert clean_body(body, CLEANER) == (
        '"Aspas" e \'simples\' "angulares"\nmuitos espaços aqui\n\n== Seção ==\nfim'
    )


@pytest.mark.req("REQ-COR-004")
def test_boilerplate_lines_are_removed_without_case():
    body = (
        "Primeiro parágrafo.\n\n"
        "LEIA TAMBÉM: outra notícia\n\n"
        "Publicidade\n\n"
        "Continua depois da publicidade\n\n"
        "Compartilhe!\n\n"
        "Leia também\n\n"
        "Leiamente não é boilerplate.\n\n"
        "Segundo parágrafo."
    )
    assert clean_body(body, CLEANER) == (
        "Primeiro parágrafo.\n\nLeiamente não é boilerplate.\n\nSegundo parágrafo."
    )
    custom = Cleaner(boilerplate_prefixes=("Assine já",))
    assert clean_body("texto\nAssine já o jornal\nLeia mais: x", custom) == "texto\nLeia mais: x"


@pytest.mark.req("REQ-COR-004")
def test_portal_suffix_is_removed_only_when_it_names_the_portal():
    assert clean_title("Chuva forte no Rio - G1", "g1.globo.com", CLEANER) == "Chuva forte no Rio"
    assert clean_title("Fla - Flu", "g1.globo.com", CLEANER) == "Fla - Flu"
    assert clean_title("Fla - Flu", "lance.com.br", CLEANER) == "Fla - Flu"
    assert (
        clean_title("Governo anuncia plano | Folha de S.Paulo", "folha.uol.com.br", CLEANER)
        == "Governo anuncia plano"
    )
    assert clean_title("Eleições — PÚBLICO", "publico.pt", CLEANER) == "Eleições"
    # More than four tokens: never a portal suffix.
    assert (
        clean_title("Dia a dia - G1 de hoje em dia", "g1.globo.com", CLEANER)
        == "Dia a dia - G1 de hoje em dia"
    )
    # The remaining title must keep a token.
    assert strip_portal_suffix("... - G1", "g1.globo.com", CLEANER) == "... - G1"
    # No URL domain: only configured suffixes apply.
    assert clean_title("Chuva forte no Rio - G1", None, CLEANER) == "Chuva forte no Rio - G1"
    custom = Cleaner(title_suffixes=("Portal Exemplo",))
    assert clean_title("Notícia - portal exemplo", None, custom) == "Notícia"


@pytest.mark.req("REQ-COR-004")
def test_stage_cleans_columns_and_removes_empty_documents():
    rows = [
        row(0, "Chuva forte no Rio - G1", "Choveu   muito.\n\n\n\nFim.", "g1.globo.com"),
        row(1, "Título", "  \n***\n---\n"),
        row(2, None, "Corpo com texto."),
        row(3, "   ", "Corpo com texto."),
        row(4, "", ""),
    ]
    result = run(rows)
    assert result.removed == {"empty_body": 2, "empty_title": 2}
    (out,) = to_rows(result.batch)
    assert out["title"] == "Chuva forte no Rio"
    assert out["body"] == "Choveu muito.\n\nFim."
    assert out["url_domain"] == "g1.globo.com"


@pytest.mark.req("REQ-COR-004")
def test_missing_ftfy_raises_with_install_hint(monkeypatch):
    monkeypatch.setattr(normalize_module, "_FTFY", None)
    monkeypatch.setitem(sys.modules, "ftfy", None)
    with pytest.raises(ArtigoEnvError, match=r"artigo-env\[data\]"):
        run([row(0, "Título", "Corpo")])
