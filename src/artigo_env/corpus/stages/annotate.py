"""S8 annotate: domain class and length band (SDD 06 §8, DEC-052; REQ-COR-012)."""

from __future__ import annotations

import json

import pyarrow as pa

from artigo_env.core.errors import ArtigoEnvError
from artigo_env.corpus.stages.context import check_batch, finish
from artigo_env.corpus.types import StageContext, StageResult

__all__ = ["DOMAIN_CLASSES", "UNKNOWN_DOMAIN", "AnnotateStage", "domain_of", "length_band"]

DOMAIN_CLASSES: tuple[str, ...] = (
    "exact-tech",
    "bio-health",
    "humanities",
    "geography",
    "history",
    "biography",
    "arts-sports",
    "other",
)
UNKNOWN_DOMAIN = "unknown"
SHORT_MAX = 500  # exclusive
MEDIUM_MAX = 2_000  # exclusive


def domain_of(provenance: str | None) -> str:
    """``provenance["domain_class"]`` when it is one of the eight classes, else ``unknown``."""
    meta = json.loads(provenance) if provenance else {}
    value = meta.get("domain_class") if isinstance(meta, dict) else None
    return value if isinstance(value, str) and value in DOMAIN_CLASSES else UNKNOWN_DOMAIN


def length_band(n_body_tokens: int) -> str:
    """``short`` (< 500 tokens), ``medium`` (< 2,000) or ``long``."""
    if n_body_tokens < SHORT_MAX:
        return "short"
    if n_body_tokens < MEDIUM_MAX:
        return "medium"
    return "long"


class AnnotateStage:
    """Fills ``domain`` and ``length_band``; removes nothing."""

    name = "annotate"
    version = "1"

    def run(self, batch: pa.RecordBatch, ctx: StageContext) -> StageResult:
        check_batch(batch, self.name)
        domains: list[str] = []
        bands: list[str] = []
        provenances = batch.column("provenance").to_pylist()
        n_bodies = batch.column("n_body_tokens").to_pylist()
        for provenance, n_body in zip(provenances, n_bodies, strict=True):
            if n_body is None:
                raise ArtigoEnvError(
                    "annotate stage needs 'n_body_tokens'; run the tokenize stage first"
                )
            domains.append(domain_of(provenance))
            bands.append(length_band(n_body))
        return finish(
            self.name, batch, [None] * batch.num_rows, {"domain": domains, "length_band": bands}
        )
