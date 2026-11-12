import pytest

from artigo_env.core.errors import ArtigoEnvError
from artigo_env.core.resources import resource_path
from artigo_env.text.stopwords import StopwordSet, StopwordSpec, load_stopwords

pytestmark = pytest.mark.fast


def file_words() -> list[str]:
    text = resource_path("stopwords", "game227.txt").read_text(encoding="utf-8")
    return [line for line in text.splitlines() if line and not line.startswith("#")]


@pytest.mark.req("REQ-TXT-010")
def test_game227_has_227_unique_words_in_order():
    words = file_words()
    assert len(words) == 227 == len(set(words))
    assert (words[0], words[-1]) == ("o", "poucas")
    assert len(load_stopwords(StopwordSpec())) == 227


@pytest.mark.req("REQ-TXT-010")
def test_game227_header_cites_the_bundle():
    text = resource_path("stopwords", "game227.txt").read_text(encoding="utf-8")
    header = [line for line in text.splitlines() if line.startswith("#")]
    assert any("be54b82b" in line for line in header)
    assert any("227" in line for line in header)


@pytest.mark.req("REQ-TXT-010")
def test_membership():
    stop = load_stopwords(StopwordSpec())
    for word in [
        "é",
        "está",
        "estão",
        "foi",
        "foram",
        "vez",
        "caso",
        "logo",
        "ora",
        "quer",
        "connosco",
        "conosco",
    ]:
        assert word in stop
    for word in [
        "ser",
        "ter",
        "há",
        "são",
        "era",
        "seu",
        "sua",
        "meu",
        "minha",
        "nosso",
        "outro",
        "mesmo",
        "cada",
        "lhes",
    ]:
        assert word not in stop


@pytest.mark.req("REQ-TXT-011")
def test_comparison_is_on_lowercase_literals_only():
    stop = load_stopwords(StopwordSpec())
    assert "de" in stop
    assert "De" not in stop  # callers lowercase the token first


@pytest.mark.req("REQ-TXT-010")
def test_content_hash_changes_with_a_word():
    base = StopwordSet(frozenset({"a", "b"}))
    assert base.content_hash() == StopwordSet(frozenset({"b", "a"})).content_hash()
    assert base.content_hash() != StopwordSet(frozenset({"a", "c"})).content_hash()


@pytest.mark.req("REQ-TXT-010")
def test_none_and_custom(tmp_path):
    assert len(load_stopwords(StopwordSpec(kind="none"))) == 0
    path = tmp_path / "custom.txt"
    path.write_text("# comment\nFoo\n\nbar\n", encoding="utf-8")
    custom = load_stopwords(StopwordSpec(kind="custom", path=str(path)))
    assert custom.words == frozenset({"foo", "bar"})
    with pytest.raises(ValueError):
        load_stopwords(StopwordSpec(kind="custom"))
    with pytest.raises(ArtigoEnvError):
        load_stopwords(StopwordSpec(kind="custom", path=str(tmp_path / "missing.txt")))


@pytest.mark.req("REQ-TXT-010")
def test_unknown_kind():
    with pytest.raises(ValueError):
        load_stopwords(StopwordSpec(kind="nope"))


@pytest.mark.req("REQ-TXT-010")
def test_closed_class_without_file_is_a_clear_error():
    if resource_path("stopwords", "closed-class-all.txt").is_file():
        pytest.skip("closed-class list already shipped")
    with pytest.raises(ArtigoEnvError, match="closed-class-all"):
        load_stopwords(StopwordSpec(kind="closed-class-all"))
