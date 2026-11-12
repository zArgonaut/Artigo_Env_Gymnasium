import pytest

from artigo_env.core.errors import ArtigoEnvError
from artigo_env.corpus.stages import BuildContext, PlayableStage, TokenizeStage
from artigo_env.corpus.stages.playable import title_informative
from artigo_env.corpus.types import DOC_SCHEMA, REASONS, PlayabilityConfig, to_batch, to_rows

pytestmark = pytest.mark.fast

FILLER = "palavra"


def body_of(n: int, *words: str) -> str:
    """A body with exactly ``n`` tokens that contains ``words`` first."""
    assert len(words) <= n
    return " ".join([*words, *([FILLER] * (n - len(words)))])


def row(i: int, title: str, body: str) -> dict[str, object]:
    out: dict[str, object] = dict.fromkeys(DOC_SCHEMA.names)
    out.update(
        source_id="src", doc_id=f"d{i}", title=title, body=body, license_id="x", provenance="{}"
    )
    return out


def run(rows, cfg=None, lemmatizer="identity"):
    ctx = BuildContext(cfg or PlayabilityConfig(), lemmatizer=lemmatizer)
    tokenized = TokenizeStage().run(to_batch(rows), ctx).batch
    result = PlayableStage().run(tokenized, ctx)
    assert result.batch.schema.equals(DOC_SCHEMA)
    assert tuple(result.removed) == REASONS["playable"]
    assert sum(result.removed.values()) == len(rows) - result.batch.num_rows
    return result


def only(result, reason: str, n: int = 1) -> None:
    expected = dict.fromkeys(REASONS["playable"], 0)
    expected[reason] = n
    assert dict(result.removed) == expected


@pytest.mark.req("REQ-COR-006")
@pytest.mark.parametrize(
    "title",
    [
        "2020",
        "12/03/2020",
        "2020-03-12",
        "12 de março de 2020",
        "Segunda-feira, 12 de março de 2020",
        "março de 2020",
        "Sem título",
        "UNTITLED",
        "Post #12",
        "Parte 2",
        "Capítulo IV",
        "Episódio nº 10",
        "a3f9c2e17b",
        "— ? —",
    ],
)
def test_title_not_informative_patterns(title):
    assert not title_informative(title)
    result = run([row(0, title, body_of(200, "março", "parte"))])
    only(result, "title_not_informative")


@pytest.mark.req("REQ-COR-006")
@pytest.mark.parametrize(
    "title", ["Março", "Parte 2: a volta", "Post-rock", "Capítulo final", "Galáxia de 2020"]
)
def test_informative_titles_are_kept(title):
    assert title_informative(title)


@pytest.mark.req("REQ-COR-006")
def test_title_content_range_follows_the_configuration():
    title = "Guerra fria"  # two content words
    rows = [row(0, title, body_of(200, "guerra", "fria"))]
    assert run(rows).batch.num_rows == 1
    only(run(rows, PlayabilityConfig(max_title_content=1)), "title_content_out_of_range")
    only(run(rows, PlayabilityConfig(min_title_content=3)), "title_content_out_of_range")
    # Only stopwords: zero content words.
    only(run([row(0, "Ele e ela", body_of(200))]), "title_content_out_of_range")


@pytest.mark.req("REQ-COR-006")
def test_body_length_bounds_follow_the_configuration():
    rows = [row(0, "Guerra", body_of(149, "guerra")), row(1, "Guerra", body_of(150, "guerra"))]
    result = run(rows)
    only(result, "body_too_short")
    assert [r["doc_id"] for r in to_rows(result.batch)] == ["d1"]
    only(run(rows, PlayabilityConfig(min_body_tokens=100)), "body_too_short", 0)


@pytest.mark.req("REQ-COR-017")
def test_body_above_maximum_is_removed_and_never_truncated():
    cfg = PlayabilityConfig(min_body_tokens=10, max_body_tokens=50)
    at_max, above = body_of(50, "guerra"), body_of(51, "guerra")
    result = run([row(0, "Guerra", at_max), row(1, "Guerra", above)], cfg)
    only(result, "body_too_long")
    (kept,) = to_rows(result.batch)
    assert kept["body"] == at_max
    assert kept["n_body_tokens"] == 50


@pytest.mark.req("REQ-COR-006")
def test_title_lemmas_must_occur_in_the_body():
    rows = [
        row(0, "Guerra fria", body_of(200, "guerra", "fria")),
        row(1, "Guerra fria", body_of(200, "guerra")),
        # The title word only appears inside a stopword-free comparison: case is ignored.
        row(2, "GUERRA", body_of(200, "Guerra")),
    ]
    result = run(rows)
    only(result, "title_not_in_body")
    out = to_rows(result.batch)
    assert [r["doc_id"] for r in out] == ["d0", "d2"]
    assert all(r["title_in_body"] is True for r in out)


@pytest.mark.req("REQ-COR-006")
def test_title_in_body_uses_the_context_lemmatizer():
    rows = [row(0, "Nações", body_of(200, "nação"))]
    only(run(rows, lemmatizer="identity"), "title_not_in_body")
    assert run(rows, lemmatizer="test-protocol").batch.num_rows == 1


@pytest.mark.req("REQ-COR-006")
def test_none_mode_records_title_in_body_instead_of_removing():
    cfg = PlayabilityConfig(require_title_lemmas_in_body="none")
    rows = [
        row(0, "Guerra fria", body_of(200, "guerra", "fria")),
        row(1, "Guerra fria", body_of(200, "guerra")),
        row(2, "Guerra fria", body_of(10, "guerra")),
    ]
    result = run(rows, cfg)
    only(result, "body_too_short")
    assert [r["title_in_body"] for r in to_rows(result.batch)] == [True, False]


@pytest.mark.req("REQ-COR-006")
def test_reasons_are_checked_in_order():
    cfg = PlayabilityConfig(min_body_tokens=10, max_body_tokens=20)
    rows = [
        row(0, "2020", body_of(5)),  # not informative wins over short body
        row(1, "Ele", body_of(5)),  # content range wins over short body
        row(2, "Guerra", body_of(5)),  # short body wins over title not in body
        row(3, "Guerra", body_of(30)),  # long body wins over title not in body
        row(4, "Guerra", body_of(15)),
    ]
    result = run(rows, cfg)
    assert dict(result.removed) == {
        "title_not_informative": 1,
        "title_content_out_of_range": 1,
        "body_too_short": 1,
        "body_too_long": 1,
        "title_not_in_body": 1,
    }
    assert result.batch.num_rows == 0


@pytest.mark.req("REQ-COR-006")
def test_playable_requires_token_counts():
    ctx = BuildContext(PlayabilityConfig(), lemmatizer="identity")
    with pytest.raises(ArtigoEnvError, match="tokenize"):
        PlayableStage().run(to_batch([row(0, "Guerra", "guerra")]), ctx)
