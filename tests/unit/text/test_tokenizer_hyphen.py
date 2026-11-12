import pytest

from artigo_env.text.tokenizer import tokenize

pytestmark = pytest.mark.fast


def texts(value: str, hyphen: str = "split") -> list[str]:
    return [token.text for token in tokenize(value, hyphen=hyphen)]


@pytest.mark.req("DEC-043")
def test_default_is_split():
    assert texts("guarda-chuva") == ["guarda", "chuva"]
    assert tokenize("guarda-chuva") == tokenize("guarda-chuva", hyphen="split")


@pytest.mark.req("DEC-043")
@pytest.mark.parametrize(
    "value, expected",
    [
        ("guarda-chuva", ["guarda-chuva"]),
        ("sul-rio-grandense", ["sul-rio-grandense"]),
        ("a - b", ["a", "b"]),
        ("-x", ["x"]),
        ("x-", ["x"]),
        ("a--b", ["a", "b"]),
        ("1-2", ["1", "2"]),
        ("a-1", ["a", "1"]),
        ("1-a", ["1", "a"]),
        ("norte-americano e sul-americano", ["norte-americano", "e", "sul-americano"]),
        ("a–b", ["a", "b"]),
    ],
)
def test_join_only_between_two_letters(value, expected):
    assert texts(value, "join") == expected


@pytest.mark.req("DEC-043")
def test_join_offsets_cover_the_whole_compound():
    (token,) = tokenize("guarda-chuva", hyphen="join")
    assert (token.start, token.end) == (0, 12)


@pytest.mark.req("DEC-043")
@pytest.mark.parametrize("value", ["strip", "", "JOIN"])
def test_unknown_hyphen_mode_raises(value):
    with pytest.raises(ValueError):
        tokenize("a-b", hyphen=value)
