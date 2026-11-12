"""Behaviour cases of the shipped ``open-tables`` lemmatizer (SDD 05 §5-§6, DEC-041)."""

from __future__ import annotations

import pytest

from artigo_env.text.lemmatizers import get_lemmatizer

pytestmark = [pytest.mark.fast, pytest.mark.req("REQ-TXT-016", "DEC-041")]

CASES = [
    ("anos", "ano"),
    ("autora", "autor"),
    ("autoras", "autor"),
    ("autores", "autor"),
    ("primeira", "primeiro"),
    ("primeiros", "primeiro"),
    ("primeiras", "primeiro"),
    ("alta", "alta"),
    ("alto", "alto"),
    ("nasceu", "nascer"),
    ("nasceram", "nascer"),
    ("nascido", "nascer"),
    ("tem", "ter"),
    ("têm", "ter"),
    ("teve", "ter"),
    ("tinha", "ter"),
    ("tendo", "ter"),
    ("tido", "ter"),
    ("tornou", "tornar"),
    ("tornaram", "tornar"),
    ("tornam", "tornar"),
    ("tornando", "tornar"),
    ("tornado", "tornar"),
    ("torna", "torna"),
    ("torno", "torno"),
    ("principais", "principal"),
    ("principalmente", "principalmente"),
    ("regiões", "região"),
    ("brasileira", "brasileiro"),
    ("história", "história"),
    ("lápis", "lápis"),
    ("vírus", "vírus"),
    ("ônibus", "ônibus"),
    ("país", "país"),
    ("mais", "mais"),
    ("menos", "menos"),
    ("detém", "deter"),
]

NOT_REPRODUCED = [
    pytest.param(
        "mães",
        "mães",
        marks=pytest.mark.xfail(
            strict=True,
            reason="the game itself gives 'mãe' (black-box run); the tables follow the game",
        ),
    ),
    pytest.param(
        "historia",
        "historiar",
        marks=pytest.mark.xfail(
            strict=True,
            reason="the game itself leaves 'historia' (short present form of a regular verb)",
        ),
    ),
    pytest.param(
        "casinha",
        "casinho",
        marks=pytest.mark.xfail(
            strict=True,
            reason="MorphoBr lists 'casinha' as a feminine noun of its own, so the measured "
            "-inha protection keeps it (the game gives 'casinho')",
        ),
    ),
]


@pytest.mark.parametrize(("form", "lemma"), CASES + NOT_REPRODUCED)
def test_open_tables_case(form: str, lemma: str) -> None:
    assert get_lemmatizer("open-tables").lemma(form) == lemma
