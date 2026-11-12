import json

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from artigo_env.corpus.stages import (
    AnnotateStage,
    BuildContext,
    NormalizeStage,
    OriginStage,
    PartitionStage,
    PlayableStage,
    TokenizeStage,
    split_of,
)
from artigo_env.corpus.types import DOC_SCHEMA, REASONS, PlayabilityConfig, to_batch, to_rows

pytestmark = pytest.mark.fast

CTX = BuildContext(PlayabilityConfig(), lemmatizer="identity")


def row(source_id: str, doc_id: str) -> dict[str, object]:
    out: dict[str, object] = dict.fromkeys(DOC_SCHEMA.names)
    out.update(
        source_id=source_id, doc_id=doc_id, title="T", body="b", license_id="x", provenance="{}"
    )
    return out


def splits(rows) -> dict[tuple[str, str], str]:
    result = PartitionStage().run(to_batch(rows), CTX)
    assert result.removed == {}
    assert result.batch.schema.equals(DOC_SCHEMA)
    return {(r["source_id"], r["doc_id"]): r["split"] for r in to_rows(result.batch)}


@pytest.mark.req("REQ-COR-013")
def test_partition_is_stable_when_other_documents_are_added():
    common = [row("cc_news_pt", f"doc-{i}") for i in range(500)]
    extra = [row("recognasumm", f"doc-{i}") for i in range(300)]
    first = splits(common)
    second = splits(extra[:150] + common[::-1] + extra[150:])
    assert all(second[key] == value for key, value in first.items())
    assert set(first.values()) == {"train", "ext-val"}


@pytest.mark.req("REQ-COR-013")
def test_train_fraction_is_about_ninety_percent():
    values = [split_of("synthetic", str(i)) for i in range(20_000)]
    fraction = values.count("train") / len(values)
    assert 0.89 <= fraction <= 0.91
    assert set(values) == {"train", "ext-val"}


@pytest.mark.req("REQ-COR-013")
def test_split_depends_on_source_and_doc_id():
    assert split_of("a", "1") == split_of("a", "1")
    differing = sum(split_of("a", str(i)) != split_of("b", str(i)) for i in range(2_000))
    assert differing > 0


# --- property: the whole per-document chain keeps the contract ----------------------------------

_WORDS = st.sampled_from(
    ["guerra", "fria", "ação", "aÃ§Ã£o", "Rio", "de", "a", "2020", "Leia também", "***", "—"]
)
_TEXT = st.lists(_WORDS, max_size=40).map(" ".join)
_BODY = st.lists(_TEXT, max_size=6).map("\n".join)
_URL = st.sampled_from(
    [None, "https://pt.wikipedia.org/x", "https://g1.globo.com/x", "https://publico.pt/x"]
)
_DOC = st.fixed_dictionaries(
    {"title": st.one_of(st.none(), _TEXT), "body": _BODY, "url": _URL, "domain": st.text()}
)
STAGES = (
    OriginStage(),
    NormalizeStage(),
    TokenizeStage(),
    PlayableStage(),
    AnnotateStage(),
    PartitionStage(),
)


@pytest.mark.req("REQ-COR-006", "REQ-COR-013")
@settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(docs=st.lists(_DOC, max_size=12), require=st.sampled_from(["all", "none"]))
def test_every_stage_output_respects_the_schema_and_the_counts(docs, require):
    ctx = BuildContext(
        PlayabilityConfig(min_body_tokens=3, require_title_lemmas_in_body=require),
        lemmatizer="identity",
    )
    rows = []
    for i, doc in enumerate(docs):
        r = row("src", f"d{i}")
        r.update(
            title=doc["title"],
            body=doc["body"],
            url=doc["url"],
            provenance=json.dumps({"domain_class": doc["domain"]}),
        )
        rows.append(r)
    batch = to_batch(rows)
    for stage in STAGES:
        before = [r["doc_id"] for r in to_rows(batch)]
        result = stage.run(batch, ctx)
        after = [r["doc_id"] for r in to_rows(result.batch)]
        assert result.batch.schema.equals(DOC_SCHEMA)
        assert set(result.removed) <= set(REASONS.get(stage.name, ()))
        assert sum(result.removed.values()) == len(before) - len(after)
        assert after == [d for d in before if d in set(after)]  # order preserved
        batch = result.batch
    for r in to_rows(batch):
        assert r["split"] in ("train", "ext-val")
        assert r["title_in_body"] in (True, False)
