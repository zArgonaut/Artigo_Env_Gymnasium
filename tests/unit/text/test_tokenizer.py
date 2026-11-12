import re

import pytest

from artigo_env.text.normalize import lower, normalize
from artigo_env.text.tokenizer import Token, is_header_line, tokenize

pytestmark = pytest.mark.fast


def texts(value: str) -> list[str]:
    return [token.text for token in tokenize(value)]


@pytest.mark.req("REQ-TXT-003")
@pytest.mark.parametrize(
    "value, expected",
    [
        ("guarda-chuva", ["guarda", "chuva"]),
        ("d'água", ["d", "água"]),
        ("Olá, mundo!", ["Olá", "mundo"]),
        ("a_b", ["a", "b"]),
        ("século XXI (2001–2100)", ["século", "XXI", "2001", "2100"]),
        ("abc123def", ["abc123def"]),
        ("3,14 e 1.000", ["3", "14", "e", "1", "000"]),
        ("", []),
        ("... — !", []),
    ],
)
def test_token_is_maximal_run_of_letters_and_ascii_digits(value, expected):
    assert texts(value) == expected


@pytest.mark.req("REQ-TXT-004")
@pytest.mark.parametrize("value", ["٣", "४२", "_", "__init__"[:2]])
def test_non_ascii_digits_and_underscore_are_not_tokens(value):
    assert texts(value) == []


@pytest.mark.req("REQ-TXT-005")
def test_ordinal_indicators_are_letters():
    assert texts("1º lugar, 2ª vez") == ["1º", "lugar", "2ª", "vez"]


@pytest.mark.req("REQ-TXT-001", "REQ-TXT-002")
def test_nfc_composes_and_keeps_diacritics():
    decomposed = "café ação"
    assert normalize(decomposed) == "café ação"
    assert texts(decomposed) == ["café", "ação"]
    assert texts("avó avô avo") == ["avó", "avô", "avo"]


@pytest.mark.req("REQ-TXT-001")
def test_orphan_combining_mark_separates_tokens():
    # A combining mark that cannot compose with its base stays a separator (R4b).
    assert texts("x́y") != ["x́y"]
    assert all("́" not in token for token in texts("x́y"))


@pytest.mark.req("REQ-TXT-006")
def test_lower_matches_javascript():
    assert lower("ÁGUA") == "água"
    assert lower("ß") == "ß"
    assert lower("İ") == "i̇"


@pytest.mark.req("REQ-TXT-007")
@pytest.mark.parametrize(
    "line, expected",
    [
        ("== História ==", True),
        ("=== Sub seção ===", True),
        ("==História==", True),
        ("== Ver também ==  ", True),
        ("a == b", False),
        # MediaWiki reads an unbalanced header at the lower level; the extra "=" joins the title.
        ("== desbalanceado ===", True),
        ("= Nível um =", False),
        ("====", False),
        ("texto comum", False),
    ],
)
def test_header_lines(line, expected):
    assert is_header_line(line) is expected


@pytest.mark.req("REQ-TXT-007")
def test_header_tokens_are_marked():
    tokens = tokenize("Intro aqui.\n== História antiga ==\nCorpo == não é.\r\n=== Sub ===\r\nfim")
    marked = [(token.text, token.header) for token in tokens]
    assert marked == [
        ("Intro", False),
        ("aqui", False),
        ("História", True),
        ("antiga", True),
        ("Corpo", False),
        ("não", False),
        ("é", False),
        ("Sub", True),
        ("fim", False),
    ]


@pytest.mark.req("REQ-TXT-017")
def test_offsets_roundtrip_on_normalized_text():
    raw = "São Paulo — 1º!\n== Seção ==\nguarda-chuva d'água"
    normalized = normalize(raw)
    tokens = tokenize(raw)
    assert all(normalized[t.start : t.end] == t.text for t in tokens)
    assert [t.start for t in tokens] == sorted(t.start for t in tokens)
    assert all(a.end <= b.start for a, b in zip(tokens, tokens[1:], strict=False))
    assert tokens[0] == Token("São", 0, 3, False)


@pytest.mark.req("REQ-TXT-004", "REQ-TXT-006")
def test_text_package_avoids_forbidden_constructs(repo_root):
    for path in (repo_root / "src" / "artigo_env" / "text").rglob("*.py"):
        code = "\n".join(
            line.split("#", 1)[0] for line in path.read_text(encoding="utf-8").splitlines()
        )
        code = re.sub(r'""".*?"""', "", code, flags=re.S)
        assert "casefold" not in code, path
        assert r"\w" not in code, path
        assert not re.search(r"^\s*import re\b|^\s*from re\b", code, flags=re.M), path
