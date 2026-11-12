import json

import pytest

from artigo_env.core.errors import ArtigoEnvError
from artigo_env.corpus.stages import AnnotateStage, BuildContext
from artigo_env.corpus.stages.annotate import DOMAIN_CLASSES, length_band
from artigo_env.corpus.types import DOC_SCHEMA, PlayabilityConfig, to_batch, to_rows

pytestmark = pytest.mark.fast

CTX = BuildContext(PlayabilityConfig(), lemmatizer="identity")


def row(i: int, provenance: dict[str, object], n_body: int | None = 300) -> dict[str, object]:
    out: dict[str, object] = dict.fromkeys(DOC_SCHEMA.names)
    out.update(
        source_id="src",
        doc_id=f"d{i}",
        title="T",
        body="b",
        license_id="x",
        provenance=json.dumps(provenance),
        n_body_tokens=n_body,
    )
    return out


@pytest.mark.req("REQ-COR-012")
def test_domain_is_unknown_without_trusted_metadata():
    provenances = [
        {},
        {"domain_class": "politics"},
        {"domain_class": "History"},
        {"section": "history"},
        {"domain_class": "history"},
        {"domain_class": "arts-sports"},
    ]
    result = AnnotateStage().run(to_batch(row(i, p) for i, p in enumerate(provenances)), CTX)
    assert result.removed == {}
    assert result.batch.schema.equals(DOC_SCHEMA)
    assert [r["domain"] for r in to_rows(result.batch)] == [
        "unknown",
        "unknown",
        "unknown",
        "unknown",
        "history",
        "arts-sports",
    ]
    assert len(DOMAIN_CLASSES) == 8


@pytest.mark.req("REQ-COR-012")
def test_length_bands():
    assert [length_band(n) for n in (150, 499, 500, 1999, 2000, 16384)] == [
        "short",
        "short",
        "medium",
        "medium",
        "long",
        "long",
    ]
    rows = [row(0, {}, 499), row(1, {}, 500), row(2, {}, 2000)]
    out = to_rows(AnnotateStage().run(to_batch(rows), CTX).batch)
    assert [r["length_band"] for r in out] == ["short", "medium", "long"]


@pytest.mark.req("REQ-COR-012")
def test_annotate_requires_token_counts():
    with pytest.raises(ArtigoEnvError, match="tokenize"):
        AnnotateStage().run(to_batch([row(0, {}, None)]), CTX)
