import json

import pytest

from artigo_env.core.resources import resource_path
from artigo_env.corpus.stages import BuildContext, OriginStage, load_blocklist
from artigo_env.corpus.stages.origin import domain_in, url_domain
from artigo_env.corpus.types import DOC_SCHEMA, REASONS, PlayabilityConfig, to_batch, to_rows

pytestmark = pytest.mark.fast

WIKIMEDIA = (
    "wikipedia.org",
    "wikinews.org",
    "wikibooks.org",
    "wikisource.org",
    "wikiquote.org",
    "wikiversity.org",
    "wikivoyage.org",
    "wiktionary.org",
    "wikimedia.org",
)


def row(i: int, url: str | None, **fields: object) -> dict[str, object]:
    out: dict[str, object] = dict.fromkeys(DOC_SCHEMA.names)
    out.update(
        source_id="src",
        doc_id=f"d{i}",
        title="Título",
        body="corpo",
        url=url,
        license_id="unknown",
        provenance="{}",
    )
    out.update(fields)
    return out


def run(rows, stage=None):
    ctx = BuildContext(PlayabilityConfig(), lemmatizer="identity")
    result = (stage or OriginStage()).run(to_batch(rows), ctx)
    assert result.batch.schema.equals(DOC_SCHEMA)
    assert set(result.removed) <= set(REASONS["origin"])
    assert sum(result.removed.values()) == len(rows) - result.batch.num_rows
    return result


@pytest.mark.req("REQ-COR-003")
def test_blocklist_files_are_package_data_and_list_every_wikimedia_project():
    assert resource_path("blocklists", "wikimedia.txt").is_file()
    assert resource_path("blocklists", "mirrors.txt").is_file()
    assert load_blocklist("wikimedia") == frozenset(WIKIMEDIA)
    assert load_blocklist("mirrors.txt")  # extension optional
    assert not any(line.startswith("#") for line in load_blocklist("mirrors"))


@pytest.mark.req("REQ-COR-003")
def test_every_wikimedia_domain_and_subdomain_is_removed():
    urls = [f"https://pt.{domain}/wiki/Artigo" for domain in WIKIMEDIA]
    urls += ["https://pt.m.wikipedia.org/wiki/Brasil", "http://WWW.Wikipedia.org/x"]
    kept_urls = ["https://wikifavelas.com.br/index.php/Rocinha", "https://g1.globo.com/a"]
    rows = [row(i, url) for i, url in enumerate(urls + kept_urls)]
    result = run(rows)
    assert result.removed["wikimedia_domain"] == len(urls)
    out = to_rows(result.batch)
    assert [r["url"] for r in out] == kept_urls
    assert [r["url_domain"] for r in out] == ["wikifavelas.com.br", "g1.globo.com"]


@pytest.mark.req("REQ-COR-003")
def test_mirrors_and_extra_domains_are_blocked_domains():
    rows = [
        row(0, "https://www.wikiwand.com/pt/Brasil"),
        row(1, "https://espelho.example.net/a"),
        row(2, "https://example.net.br/a"),  # suffix match must follow label boundaries
    ]
    result = run(rows, OriginStage(blocked_domains=frozenset({"Example.NET"})))
    assert result.removed == {"wikimedia_domain": 0, "blocked_domain": 2, "blocked_metadata": 0}
    assert to_rows(result.batch)[0]["doc_id"] == "d2"


@pytest.mark.req("REQ-COR-003")
def test_blocked_metadata_values_are_removed():
    stage = OriginStage(blocked_metadata={"corpus": frozenset({"wik", "dat"})})
    rows = [
        row(0, None, provenance=json.dumps({"corpus": "wik"})),
        row(1, None, provenance=json.dumps({"corpus": "jud"})),
        row(2, None, provenance=json.dumps({"other": "wik"})),
        row(3, "https://pt.wikipedia.org/x", provenance=json.dumps({"corpus": "dat"})),
    ]
    result = run(rows, stage)
    # The Wikimedia reason wins over the metadata reason; each row is counted once.
    assert result.removed == {"wikimedia_domain": 1, "blocked_domain": 0, "blocked_metadata": 1}
    assert [r["doc_id"] for r in to_rows(result.batch)] == ["d1", "d2"]


@pytest.mark.req("REQ-COR-018")
def test_variant_is_marked_from_the_host_and_kept_when_already_set():
    rows = [
        row(0, "https://www.publico.pt/2020/x"),
        row(1, "https://www1.folha.uol.com.br/x"),
        row(2, "https://example.com/x"),
        row(3, "https://expresso.pt/x", variant="pt-BR"),
        row(4, None, variant="pt-PT"),
        row(5, None),
    ]
    out = to_rows(run(rows).batch)
    assert [r["variant"] for r in out] == ["pt-PT", "pt-BR", None, "pt-BR", "pt-PT", None]
    assert out[0]["url_domain"] == "publico.pt"


@pytest.mark.req("REQ-COR-003")
def test_url_domain_rules():
    assert url_domain("https://WWW.G1.Globo.com:443/a?b=1") == "g1.globo.com"
    assert url_domain("g1.globo.com/economia") == "g1.globo.com"
    assert url_domain("//pt.m.wikipedia.org/wiki/X") == "pt.m.wikipedia.org"
    assert url_domain("") is None
    assert url_domain(None) is None
    assert url_domain("http://[::1") is None
    assert domain_in("pt.m.wikipedia.org", frozenset({"wikipedia.org"}))
    assert not domain_in("notwikipedia.org", frozenset({"wikipedia.org"}))


@pytest.mark.req("REQ-COR-003")
def test_row_order_and_other_columns_are_preserved():
    rows = [row(i, f"https://site{i}.com.br/x", title=f"T{i}") for i in range(5)]
    out = to_rows(run(rows).batch)
    for before, after in zip(rows, out, strict=True):
        for key in DOC_SCHEMA.names:
            if key not in ("url_domain", "variant"):
                assert after[key] == before[key]
