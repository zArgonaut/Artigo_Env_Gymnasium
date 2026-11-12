"""S7 dedup: exact and near duplicates among the documents that survived S6 (SDD 06 §7.4).

Three tests, in order, over the whole build at once (REQ-COR-011):

* ``dup_url`` — same URL after normalization (scheme, ``www.``, trailing slash, fragment and
  tracking parameters removed, host lowercased);
* ``dup_body`` — same body text (BLAKE2b-128 of the UTF-8 body after S3);
* ``near_duplicate`` — exact Jaccard of the 5-gram shingles ``>= 0.8`` with a document already
  kept (candidates by MinHash/LSH, decision by exact Jaccard; ``NearDuplicateIndex``).

Documents are visited from the most to the least preferred one: lowest source ``priority``,
then the earliest ``date`` (documents without a date come last), then the smallest ``doc_id``
(and ``source_id`` to break the remaining ties). A document is removed when it duplicates one
already kept, so in every group of duplicates the preferred document survives.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from urllib.parse import parse_qsl, urlencode, urlsplit

import pyarrow as pa

from artigo_env.corpus.decontam.grams import gram_tokens
from artigo_env.corpus.decontam.minhash import NearDuplicateIndex, shingle_hashes
from artigo_env.corpus.stages.context import check_batch, finish
from artigo_env.corpus.types import REASONS, StageResult

__all__ = [
    "TRACKING_PARAMS",
    "DedupItem",
    "body_hash",
    "deduplicate",
    "dedup_batch",
    "normalize_url",
    "rank_key",
]

STAGE = "dedup"
TRACKING_PARAMS = frozenset(
    {
        "fbclid",
        "gclid",
        "dclid",
        "msclkid",
        "mc_cid",
        "mc_eid",
        "igshid",
        "yclid",
        "ref",
        "ref_src",
        "cmpid",
        "xtor",
        "amp",
        "_ga",
    }
)
_TRACKING_PREFIXES = ("utm_", "pk_", "hsa_")
_NO_DATE = "\U0010ffff"  # sorts after every ISO 8601 date


def normalize_url(url: str | None) -> str | None:
    """Comparison key of a URL, or ``None`` when it has no host.

    Scheme dropped, host lowercased without ``www.`` and default port, fragment removed,
    tracking parameters removed, remaining parameters sorted, trailing slash removed.
    """
    if not url or not url.strip():
        return None
    text = url.strip()
    if "://" not in text and not text.startswith("//"):
        text = "//" + text
    try:
        parts = urlsplit(text)
        host = parts.hostname
        port = parts.port
    except ValueError:
        return None
    if not host:
        return None
    host = host.lower().strip(".")
    if host.startswith("www."):
        host = host[4:]
    if port is not None and port not in (80, 443):
        host = f"{host}:{port}"
    query = [
        (key, value)
        for key, value in parse_qsl(parts.query, keep_blank_values=True)
        if key.lower() not in TRACKING_PARAMS and not key.lower().startswith(_TRACKING_PREFIXES)
    ]
    path = parts.path.rstrip("/")
    key = host + path
    if query:
        key += "?" + urlencode(sorted(query))
    return key


def body_hash(body: str) -> bytes:
    """BLAKE2b-128 digest of the body text."""
    return hashlib.blake2b(body.encode("utf-8"), digest_size=16).digest()


def rank_key(priority: int, date: str | None, doc_id: str, source_id: str) -> tuple[object, ...]:
    """Preference order: lower priority, earlier date (missing last), smaller doc_id."""
    return (priority, date if date else _NO_DATE, doc_id, source_id)


@dataclass(frozen=True, slots=True)
class DedupItem:
    """What the dedup stage needs of one document."""

    source_id: str
    doc_id: str
    url: str | None
    date: str | None
    body: str


def deduplicate(
    items: Sequence[DedupItem],
    priorities: Mapping[str, int],
    hyphen: str = "split",
) -> list[str | None]:
    """Removal reason per item (``None`` = kept), in the order of ``items``."""
    order = sorted(
        range(len(items)),
        key=lambda i: rank_key(
            priorities[items[i].source_id], items[i].date, items[i].doc_id, items[i].source_id
        ),
    )
    reasons: list[str | None] = [None] * len(items)
    urls: set[str] = set()
    bodies: set[bytes] = set()
    near = NearDuplicateIndex()
    for i in order:
        item = items[i]
        url = normalize_url(item.url)
        if url is not None and url in urls:
            reasons[i] = "dup_url"
            continue
        digest = body_hash(item.body)
        if digest in bodies:
            reasons[i] = "dup_body"
            continue
        shingles = shingle_hashes(gram_tokens(item.body, hyphen))
        if near.match(shingles) is not None:
            reasons[i] = "near_duplicate"
            continue
        if url is not None:
            urls.add(url)
        bodies.add(digest)
        near.add(i, shingles)
    allowed = set(REASONS[STAGE])
    assert all(reason is None or reason in allowed for reason in reasons)
    return reasons


def dedup_batch(
    batch: pa.RecordBatch, priorities: Mapping[str, int], hyphen: str = "split"
) -> tuple[StageResult, list[str | None]]:
    """S7 over one ``DOC_SCHEMA`` batch holding the whole build; also returns the reasons."""
    check_batch(batch, STAGE)
    columns = {name: batch.column(name).to_pylist() for name in ("source_id", "doc_id", "url")}
    dates = batch.column("date").to_pylist()
    bodies = batch.column("body").to_pylist()
    items = [
        DedupItem(source_id, doc_id, url, date, body or "")
        for source_id, doc_id, url, date, body in zip(
            columns["source_id"], columns["doc_id"], columns["url"], dates, bodies, strict=True
        )
    ]
    reasons = deduplicate(items, priorities, hyphen)
    return finish(STAGE, batch, reasons), reasons
