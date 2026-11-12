import pytest

from artigo_env.core.errors import ArtigoEnvError, UnknownNameError
from artigo_env.core.resources import resource_path
from artigo_env.text.lemmatizers import LEMMATIZERS, get_lemmatizer
from artigo_env.text.lemmatizers.base import RuleLemmatizer
from artigo_env.text.lemmatizers.tables import load_tables

pytestmark = pytest.mark.fast


@pytest.mark.req("REQ-TXT-016")
def test_identity_returns_the_form():
    lemmatizer = get_lemmatizer("identity")
    assert lemmatizer.lemma("Casas") == "Casas"
    assert lemmatizer.content_hash() == get_lemmatizer("identity").content_hash()
    assert get_lemmatizer("identity") is lemmatizer


@pytest.mark.req("REQ-TXT-016")
def test_registered_names():
    assert {"identity", "open-tables", "reference-local"} <= set(LEMMATIZERS.names())


@pytest.mark.req("REQ-TXT-016")
def test_unknown_name():
    with pytest.raises(UnknownNameError):
        get_lemmatizer("nope")


@pytest.mark.req("REQ-TXT-016")
def test_reference_local_without_file(tmp_path, monkeypatch):
    monkeypatch.setenv("ARTIGO_DATA_ROOT", str(tmp_path))
    with pytest.raises(ArtigoEnvError, match="reference/lemmatizer/tables.json"):
        LEMMATIZERS.get("reference-local")()


@pytest.mark.req("REQ-TXT-016")
def test_open_tables_missing_is_a_clear_error():
    if resource_path("lemmatizers", "open-tables", "tables.json").is_file():
        pytest.skip("open-tables already shipped")
    with pytest.raises(ArtigoEnvError, match="open-tables"):
        LEMMATIZERS.get("open-tables")()


@pytest.mark.req("REQ-TXT-016", "DEC-040")
def test_mini_table_loads_and_lemmatizes(repo_root):
    tables = load_tables(repo_root / "tests" / "fixtures" / "lemma_tables" / "mini.json")
    lemmatizer = RuleLemmatizer("mini", tables)
    expected = {
        "tem": "ter",
        "teve": "ter",
        "tornou": "tornar",
        "tornando": "tornar",
        "nasceu": "nascer",
        "torna": "torna",
        "anos": "ano",
        "regiões": "região",
        "lápis": "lápis",
        "pequenas": "pequeno",
        "casa": "casa",
        "alta": "alta",
        "alto": "alto",
    }
    for form, lemma in expected.items():
        assert lemmatizer.lemma(form) == lemma, form
    assert tables.name == "mini"


@pytest.mark.req("REQ-TXT-016")
def test_other_format_version_is_rejected(tmp_path):
    path = tmp_path / "t.json"
    path.write_text('{"format_version": 2, "name": "x", "source": "y", "stages": {}}', "utf-8")
    with pytest.raises(ValueError, match="format_version"):
        load_tables(path)
