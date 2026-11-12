"""S2 origin: Wikimedia and blocklist filter, URL domain and language variant (SDD 06 §4).

REQ-COR-003 (blocklist by domain and by metadata), REQ-COR-018 (``pt-PT`` marking).
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from types import MappingProxyType
from urllib.parse import urlsplit

import pyarrow as pa

from artigo_env.core.errors import ArtigoEnvError
from artigo_env.core.resources import resource_path
from artigo_env.corpus.stages.context import check_batch, finish
from artigo_env.corpus.types import StageContext, StageResult

__all__ = ["OriginStage", "domain_in", "load_blocklist", "url_domain", "variant_of"]

_NO_METADATA: Mapping[str, frozenset[str]] = MappingProxyType({})


def load_blocklist(name: str) -> frozenset[str]:
    """Domains of ``assets/blocklists/<name>.txt``: one per line, ``#`` starts a comment."""
    filename = name if name.endswith(".txt") else f"{name}.txt"
    node = resource_path("blocklists", filename)
    if not node.is_file():
        raise ArtigoEnvError(f"blocklist not found: assets/blocklists/{filename}")
    domains: set[str] = set()
    for line in node.read_text(encoding="utf-8").splitlines():
        entry = line.split("#", 1)[0].strip().lower().strip(".")
        if entry:
            domains.add(entry)
    return frozenset(domains)


def url_domain(url: str | None) -> str | None:
    """Lowercase host of ``url`` without a leading ``www.``; ``None`` if there is no host."""
    if not url or not url.strip():
        return None
    text = url.strip()
    if "://" not in text and not text.startswith("//"):
        text = "//" + text
    try:
        host = urlsplit(text).hostname
    except ValueError:
        return None
    if not host:
        return None
    host = host.lower().strip(".")
    if host.startswith("www."):
        host = host[4:]
    return host or None


def domain_in(host: str, domains: frozenset[str]) -> bool:
    """Whether ``host`` equals one of ``domains`` or is a subdomain of one."""
    labels = host.split(".")
    return any(".".join(labels[i:]) in domains for i in range(len(labels)))


def variant_of(host: str | None) -> str | None:
    """``pt-PT`` for ``.pt`` hosts, ``pt-BR`` for ``.br`` hosts, otherwise ``None``."""
    if host is None:
        return None
    if host == "pt" or host.endswith(".pt"):
        return "pt-PT"
    if host == "br" or host.endswith(".br"):
        return "pt-BR"
    return None


class OriginStage:
    """Removes Wikimedia, blocked-domain and blocked-metadata documents; fills ``url_domain``.

    Also marks ``variant`` from the host's country suffix, keeping a value already set by the
    source (REQ-COR-018).
    """

    name = "origin"
    version = "1"

    def __init__(
        self,
        blocked_domains: frozenset[str] | None = None,
        blocked_metadata: Mapping[str, frozenset[str]] = _NO_METADATA,
    ) -> None:
        self.wikimedia = load_blocklist("wikimedia")
        extra = frozenset(d.lower().strip(".") for d in (blocked_domains or ()))
        self.blocked = load_blocklist("mirrors") | extra
        self.blocked_metadata = {
            key: frozenset(values) for key, values in sorted(blocked_metadata.items())
        }

    def _metadata_blocked(self, provenance: str) -> bool:
        if not self.blocked_metadata:
            return False
        meta = json.loads(provenance) if provenance else {}
        for key, values in self.blocked_metadata.items():
            value = meta.get(key)
            if isinstance(value, str) and value in values:
                return True
        return False

    def run(self, batch: pa.RecordBatch, ctx: StageContext) -> StageResult:
        check_batch(batch, self.name)
        urls = batch.column("url").to_pylist()
        provenances = batch.column("provenance").to_pylist()
        variants = batch.column("variant").to_pylist()
        domains: list[str | None] = []
        new_variants: list[str | None] = []
        reasons: list[str | None] = []
        for url, provenance, variant in zip(urls, provenances, variants, strict=True):
            host = url_domain(url)
            domains.append(host)
            new_variants.append(variant if variant is not None else variant_of(host))
            if host is not None and domain_in(host, self.wikimedia):
                reasons.append("wikimedia_domain")
            elif host is not None and domain_in(host, self.blocked):
                reasons.append("blocked_domain")
            elif self._metadata_blocked(provenance):
                reasons.append("blocked_metadata")
            else:
                reasons.append(None)
        return finish(self.name, batch, reasons, {"url_domain": domains, "variant": new_variants})
