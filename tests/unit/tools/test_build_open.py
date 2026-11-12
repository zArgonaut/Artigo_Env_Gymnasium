"""Unit tests for the open lemmatizer table builder (synthetic MorphoBr, no local data)."""

from __future__ import annotations

import dataclasses

import pytest

from artigo_env.text.lemmatizers.base import RuleLemmatizer
from artigo_env.text.lemmatizers.tables import tables_from_dict
from tools.lemma_tables import build_open

pytestmark = pytest.mark.fast

SYNTHETIC_MORPHOBR = """\
torno\ttornar+V+PRS+1+SG
torna\ttornar+V+PRS+3+SG
tornas\ttornar+V+PRS+2+SG
torne\ttornar+V+SBJR+3+SG
tornou\ttornar+V+PRF+3+SG
tornaram\ttornar+V+PRF+3+PL
tornam\ttornar+V+PRS+3+PL
tornando\ttornar+V+GRD
tornado\ttornar+V+PTPST+M+SG
tornar\ttornar+V+INF
cantou\tcantar+V+PRF+3+SG
cantaram\tcantar+V+PRF+3+PL
canta\tcantar+V+PRS+3+SG
cantar\tcantar+V+INF
tem\tter+V+PRS+3+SG
têm\tter+V+PRS+3+PL
teve\tter+V+PRF+3+SG
tinha\tter+V+IMPF+3+SG
tendo\tter+V+GRD
tido\tter+V+PTPST+M+SG
ter\tter+V+INF
detém\tdeter+V+PRS+3+SG
deteve\tdeter+V+PRF+3+SG
detinha\tdeter+V+IMPF+3+SG
detendo\tdeter+V+GRD
detido\tdeter+V+PTPST+M+SG
deter\tdeter+V+INF
ano\tano+N+M+SG
anos\tano+N+M+PL
autor\tautor+N+M+SG
autora\tautor+N+F+SG
autores\tautor+N+M+PL
autoras\tautor+N+F+PL
lápis\tlápis+N+M+SG
lápis\tlápis+N+M+PL
alto\talto+A+M+SG
alta\talto+A+F+SG
primeiro\tprimeiro+A+M+SG
primeira\tprimeiro+A+F+SG
casas\tcasa+N+F+PL
"""

FREQUENCIES = {
    "tornou": 50,
    "tornaram": 30,
    "cantou": 40,
    "teve": 100,
    "tinha": 100,
    "tendo": 50,
}

POLICY = dataclasses.replace(
    build_open.DEFAULT_POLICY,
    weak_max_deviating=0,
    # Too few pairs here for the suffix-pattern support the whole-word list relies on.
    gender_whole_words=False,
    gender_rules=(("ora", "or", True), ("ira", "iro", True)),
)


def _lexicon() -> build_open.Lexicon:
    lexicon = build_open.Lexicon()
    for line in SYNTHETIC_MORPHOBR.splitlines():
        lexicon.add(line)
    return lexicon


def _build() -> tuple[dict[str, object], dict[str, int]]:
    return build_open.build_tables(_lexicon(), FREQUENCIES, POLICY)


@pytest.mark.req("REQ-TXT-016")
def test_regular_rules_use_the_paradigm_without_short_endings():
    tables, stats = _build()
    stages = tables["stages"]
    assert isinstance(stages, dict)
    regular = stages["regular"]["members"]
    assert len(regular) == 1
    member = regular[0]
    assert member["rules"] == [["cant", "cantar", ""], ["torn", "tornar", ""]]
    endings = member["suffix"][len("(?:") : -1].split("|")
    assert {"ou", "aram", "am", "ando", "ado", "ar"} <= set(endings)
    assert not {"o", "a", "as", "e"} & set(endings)
    assert stats["verbs_regular"] == 2
    assert stats["verbs_irregular"] == 1  # ter; deter is too rare on its own


@pytest.mark.req("REQ-TXT-016")
def test_built_tables_lemmatize_as_expected():
    tables, _ = _build()
    lemmatizer = RuleLemmatizer("synthetic", tables_from_dict(tables))
    expected = {
        "tornou": "tornar",
        "tornaram": "tornar",
        "tornam": "tornar",
        "tornando": "tornar",
        "tornado": "tornar",
        "torna": "torna",
        "torno": "torno",
        "tem": "ter",
        "têm": "ter",
        "tinha": "ter",
        "tido": "ter",
        "detém": "deter",  # productive prefix validated by MorphoBr
        "anos": "ano",
        "autoras": "autor",
        "autores": "autor",
        "primeira": "primeiro",
        "lápis": "lápis",  # MorphoBr exception of the -s rule
        "alta": "alta",  # no generic -a -> -o rule
        "casas": "casa",
    }
    assert {form: lemmatizer.lemma(form) for form in expected} == expected


@pytest.mark.req("REQ-TXT-016")
def test_build_is_deterministic():
    first = build_open.dump_tables(_build()[0])
    second = build_open.dump_tables(_build()[0])
    assert first == second
    assert first.encode("utf-8") == second.encode("utf-8")
