"""S7 dedup: URL normalization, exact body duplicates and near duplicates (REQ-COR-011)."""

from __future__ import annotations

import random

import pytest

from artigo_env.corpus.dedup import DedupItem, dedup_batch, deduplicate, normalize_url
from artigo_env.corpus.types import DOC_SCHEMA, to_batch

pytestmark = pytest.mark.fast

WORDS = [f"pal{i:03d}" for i in range(400)]


def text(seed: int, n: int = 200) -> str:
    rng = random.Random(seed)
    return " ".join(rng.choice(WORDS) for _ in range(n)) + "."


@pytest.mark.req("REQ-COR-011")
@pytest.mark.parametrize(
    ("a", "b"),
    [
        ("https://www.exemplo.com.br/a/b/", "http://exemplo.com.br/a/b"),
        ("https://exemplo.com.br/a?utm_source=x&id=3#top", "exemplo.com.br/a?id=3"),
        ("https://EXEMPLO.com.br/a?fbclid=1&b=2&a=1", "https://exemplo.com.br/a?a=1&b=2"),
    ],
)
def test_normalize_url_equates_variants(a: str, b: str) -> None:
    assert normalize_url(a) == normalize_url(b)


@pytest.mark.req("REQ-COR-011")
def test_normalize_url_keeps_distinct_pages() -> None:
    assert normalize_url("https://a.com/x?id=1") != normalize_url("https://a.com/x?id=2")
    assert normalize_url("https://a.com/x") != normalize_url("https://b.com/x")
    assert normalize_url(None) is None
    assert normalize_url("   ") is None


@pytest.mark.req("REQ-COR-011")
def test_planted_duplicates_are_removed_and_the_preferred_one_kept() -> None:
    base = text(1)
    words = base.split()
    near = words[:]
    for i in range(0, len(near), 100):  # change 1% of the words
        near[i] = "trocada"
    items = [
        DedupItem("news", "n2", "https://www.site.com.br/x/", "2020-01-02", text(2)),
        DedupItem("news", "n1", "http://site.com.br/x?utm_medium=y", "2020-01-01", text(3)),
        DedupItem("news", "n3", "https://site.com.br/body", "2020-01-05", base),
        DedupItem("blog", "b1", "https://blog.pt/copia", "2019-01-01", base),  # lower priority
        DedupItem("news", "n4", "https://site.com.br/near", None, " ".join(near)),
        DedupItem("news", "n5", "https://site.com.br/clean", "2020-02-01", text(9)),
    ]
    reasons = deduplicate(items, {"news": 10, "blog": 20})
    assert reasons == [
        "dup_url",  # n2 loses to n1 (earlier date) on the same URL
        None,
        None,  # n3 is preferred over b1 because news has the lower priority number
        "dup_body",
        "near_duplicate",
        None,
    ]


@pytest.mark.req("REQ-COR-011")
def test_ties_are_broken_by_doc_id() -> None:
    body = text(5)
    items = [DedupItem("s", "b", None, None, body), DedupItem("s", "a", None, None, body)]
    assert deduplicate(items, {"s": 1}) == ["dup_body", None]


@pytest.mark.req("REQ-COR-011")
def test_dedup_batch_counts_every_reason() -> None:
    rows = []
    for i, body in enumerate([text(1), text(1), text(2)]):
        row = dict.fromkeys(DOC_SCHEMA.names)
        row.update(
            source_id="s", doc_id=f"d{i}", body=body, title="t", license_id="x", provenance="{}"
        )
        rows.append(row)
    result, reasons = dedup_batch(to_batch(rows), {"s": 1})
    assert result.batch.num_rows == 2
    assert result.removed == {"dup_url": 0, "dup_body": 1, "near_duplicate": 0}
    assert reasons == [None, "dup_body", None]
