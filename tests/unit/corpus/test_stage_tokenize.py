import pickle

import pytest

from artigo_env.corpus.stages import BuildContext, TokenizeStage
from artigo_env.corpus.types import DOC_SCHEMA, PlayabilityConfig, to_batch, to_rows
from artigo_env.text.stopwords import StopwordSpec

pytestmark = pytest.mark.fast


def row(i: int, title: str, body: str) -> dict[str, object]:
    out: dict[str, object] = dict.fromkeys(DOC_SCHEMA.names)
    out.update(
        source_id="src", doc_id=f"d{i}", title=title, body=body, license_id="x", provenance="{}"
    )
    return out


@pytest.mark.req("REQ-TXT-003")
def test_counts_tokens_and_title_content_words():
    ctx = BuildContext(PlayabilityConfig(), lemmatizer="identity")
    rows = [
        row(0, "A Guerra dos Cem Anos", "A guerra durou 116 anos (1337-1453)."),
        row(1, "Sul-rio-grandense", "== Seção ==\num dois"),
    ]
    result = TokenizeStage().run(to_batch(rows), ctx)
    assert result.removed == {}
    assert result.batch.schema.equals(DOC_SCHEMA)
    out = to_rows(result.batch)
    assert [(r["n_body_tokens"], r["n_title_tokens"], r["n_title_content"]) for r in out] == [
        (7, 5, 3),  # "a" and "dos" are game227 stopwords
        (3, 3, 3),
    ]
    assert [r["title"] for r in out] == [r["title"] for r in rows]


@pytest.mark.req("REQ-TXT-003")
def test_hyphen_mode_comes_from_the_context():
    ctx = BuildContext(PlayabilityConfig(), tokenizer_hyphen="join", lemmatizer="identity")
    (out,) = to_rows(TokenizeStage().run(to_batch([row(0, "Sul-rio-grandense", "x")]), ctx).batch)
    assert out["n_title_tokens"] == 1


@pytest.mark.req("REQ-COR-006")
def test_build_context_hash_tracks_every_component():
    base = BuildContext(PlayabilityConfig(), lemmatizer="identity")
    same = BuildContext(PlayabilityConfig(), lemmatizer="identity")
    assert base.content_hash() == same.content_hash()
    others = [
        BuildContext(PlayabilityConfig(min_body_tokens=10), lemmatizer="identity"),
        BuildContext(PlayabilityConfig(), tokenizer_hyphen="join", lemmatizer="identity"),
        BuildContext(PlayabilityConfig(), stopwords=StopwordSpec("none"), lemmatizer="identity"),
        BuildContext(PlayabilityConfig(), lemmatizer="test-protocol"),
    ]
    hashes = {base.content_hash(), *(ctx.content_hash() for ctx in others)}
    assert len(hashes) == 5
    with pytest.raises(ValueError):
        BuildContext(PlayabilityConfig(), tokenizer_hyphen="glue")


@pytest.mark.req("REQ-COR-006")
def test_build_context_stopwords_lemmas_and_pickling():
    ctx = BuildContext(PlayabilityConfig(), lemmatizer="test-protocol")
    assert ctx.is_stopword("Dos") and not ctx.is_stopword("guerra")
    assert ctx.lemma("nações") == "nação"
    assert ctx.lemma("nações") == "nação"  # cached
    clone = pickle.loads(pickle.dumps(ctx))
    assert clone.lemma("tornaram") == "tornar"
    assert clone.content_hash() == ctx.content_hash()
