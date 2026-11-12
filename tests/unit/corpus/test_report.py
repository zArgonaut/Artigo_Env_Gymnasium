"""S11 report: schema of report.json, counts per stage, reason and source (REQ-COR-015)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from artigo_env.corpus.pipeline import build
from artigo_env.corpus.report import StageCounter, check_report, domain_suffix
from artigo_env.corpus.types import REASONS
from tests.unit.corpus.test_pipeline_config import (
    config,
    news_rows,
    synthetic_sources,
    write_shard,
)

pytestmark = pytest.mark.fast


@pytest.fixture
def report_build(tmp_path: Path):
    root = tmp_path / "data"
    rows = news_rows(1, 10)
    rows[0]["url"] = "https://pt.wikipedia.org/wiki/X"  # origin: wikimedia_domain
    rows[1]["text"] = "curto demais"  # playable: body_too_short
    rows[2].update(text=rows[3]["text"], title=rows[3]["title"])  # dedup: dup_body
    rows[4]["url"] = "https://www.jornal.pt/a"  # variant pt-PT
    write_shard(root, "news", "0001", rows)
    write_shard(root, "blog", "0001", news_rows(2, 5, host="blog.net"))
    return build(config(sources=("news", "blog")), data_root=root, sources=synthetic_sources())


@pytest.mark.req("REQ-COR-015")
def test_report_json_schema_and_counts(report_build) -> None:
    report = json.loads((report_build.out_dir / "report.json").read_text(encoding="utf-8"))
    assert check_report(report) == []
    assert report["stage_order"] == [
        "ingest",
        "origin",
        "normalize",
        "tokenize",
        "playable",
        "decontam",
        "dedup",
        "annotate",
        "partition",
        "compile",
    ]
    stages = report["stages"]
    assert stages["ingest"]["in"] == 15
    assert stages["origin"]["removed"]["wikimedia_domain"] == 1
    assert stages["playable"]["removed"]["body_too_short"] == 1
    assert stages["dedup"]["removed"]["dup_body"] == 1
    assert stages["decontam"]["skipped"] is True
    for stage, reasons in REASONS.items():
        assert set(stages[stage]["removed"]) == set(reasons)
    news = report["by_source"]["news"]
    blog = report["by_source"]["blog"]
    assert news["ingest"]["in"] == 10 and blog["ingest"]["in"] == 5
    assert news["compile"]["out"] + blog["compile"]["out"] == report["n_docs"] == 12
    dist = report["distributions"]
    assert dist["variant"]["pt-PT"] == 1
    assert dist["domain_suffix"] == {".com.br": 6, ".net": 5, ".pt": 1}
    assert sum(dist["split"].values()) == 12
    assert sum(dist["title_content_words"].values()) == 12
    assert report["decontam"] == {"status": "skipped"}


@pytest.mark.req("REQ-COR-015")
def test_only_report_md_carries_a_date(report_build) -> None:
    md = (report_build.out_dir / "report.md").read_text(encoding="utf-8")
    assert "Generated: 20" in md and "decontam: skipped" in md
    text = (report_build.out_dir / "report.json").read_text(encoding="utf-8")
    assert "Generated" not in text and "wall" not in text


@pytest.mark.req("REQ-COR-015")
def test_check_report_finds_problems(report_build) -> None:
    report = dict(report_build.report)
    broken = json.loads(json.dumps(report))
    broken["stages"]["origin"]["out"] += 1
    assert any("in - out" in p for p in check_report(broken))
    del broken["distributions"]
    assert "missing key 'distributions'" in check_report(broken)
    bad = json.loads(json.dumps(report))
    bad["decontam"] = {"status": "maybe"}
    assert check_report(bad)


@pytest.mark.req("REQ-COR-015")
def test_stage_counter_and_suffixes() -> None:
    counter = StageCounter()
    counter.add("b", 3, 2, {"x": 1})
    counter.merge(
        {"a": {"in": 2, "out": 2, "removed": {}}, "b": {"in": 1, "out": 0, "removed": {"x": 1}}}
    )
    counter.reorder(["a", "b"])
    assert counter.to_dict() == {
        "a": {"in": 2, "out": 2, "removed": {}},
        "b": {"in": 4, "out": 2, "removed": {"x": 2}},
    }
    assert domain_suffix("g1.globo.com") == ".com"
    assert domain_suffix("x.gov.br") == ".gov.br"
    assert domain_suffix("a.de") == ".de"
    assert domain_suffix(None) == "none"
