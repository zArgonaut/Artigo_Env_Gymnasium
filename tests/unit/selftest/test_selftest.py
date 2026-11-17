"""The product self-test: registry, runner, report and the quick run itself."""

from __future__ import annotations

import io
import json

import jsonschema
import pytest

from artigo_env.cli.main import main
from artigo_env.selftest import Registry, SkipCheck, check, run_checks
from artigo_env.selftest.registry import UnknownGroupError
from artigo_env.selftest.report import exit_code, table, to_json

pytestmark = pytest.mark.fast

REPORT_SCHEMA = {
    "type": "object",
    "required": [
        "format_version",
        "package_version",
        "python",
        "platform",
        "quick",
        "ok",
        "counts",
        "seconds",
        "results",
    ],
    "properties": {
        "format_version": {"const": 1},
        "quick": {"type": "boolean"},
        "ok": {"type": "boolean"},
        "seconds": {"type": "number"},
        "counts": {
            "type": "object",
            "required": ["pass", "fail", "skip"],
            "additionalProperties": {"type": "integer"},
        },
        "results": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["group", "name", "status", "detail", "seconds"],
                "additionalProperties": False,
                "properties": {
                    "group": {"type": "string"},
                    "name": {"type": "string"},
                    "status": {"enum": ["pass", "fail", "skip"]},
                    "detail": {"type": "string"},
                    "seconds": {"type": "number"},
                },
            },
        },
    },
}


def sample_registry() -> Registry:
    registry = Registry()

    @check("engine", "broken", registry=registry)
    def broken(ctx):
        raise RuntimeError("boom\nsecond line")

    @check("engine", "fine", registry=registry)
    def fine(ctx):
        return "all good"

    @check("install", "absent data", registry=registry)
    def absent(ctx):
        raise SkipCheck("no local data")

    @check("install", "slow", full_only=True, registry=registry)
    def slow(ctx):
        return "ran"

    @check("custom", "tmp", registry=registry)
    def tmp(ctx):
        path = ctx.workdir("x")
        assert path.is_dir() and not any(path.iterdir())
        return None

    return registry


@pytest.mark.req("REQ-PLT-001")
def test_registry_orders_groups_and_rejects_duplicates():
    registry = sample_registry()
    assert registry.groups() == ["install", "engine", "custom"]
    with pytest.raises(ValueError, match="duplicate"):
        check("engine", "fine", registry=registry)(lambda ctx: None)


@pytest.mark.req("REQ-PLT-001")
def test_a_failing_check_does_not_stop_the_run():
    results = run_checks(registry=sample_registry())
    status = {r.name: (r.status, r.detail) for r in results}
    assert status["broken"] == ("fail", "RuntimeError: boom")
    assert status["fine"] == ("pass", "all good")
    assert status["absent data"] == ("skip", "no local data")
    assert status["slow"] == ("pass", "ran")
    assert status["tmp"] == ("pass", "ok")
    assert exit_code(results) == 1
    assert "FAILED: 3 passed, 1 failed, 1 skipped" in table(results, quick=False, seconds=0.1)


@pytest.mark.req("REQ-PLT-001")
def test_quick_skips_full_only_checks_and_only_filters_groups():
    results = run_checks(registry=sample_registry(), quick=True, only=["install"])
    assert [(r.name, r.status) for r in results] == [("absent data", "skip"), ("slow", "skip")]
    assert exit_code(results) == 0
    with pytest.raises(UnknownGroupError):
        run_checks(registry=sample_registry(), only=["nope"])


@pytest.mark.req("REQ-PLT-001")
def test_json_report_schema():
    results = run_checks(registry=sample_registry())
    report = to_json(results, quick=False, seconds=1.0)
    jsonschema.validate(report, REPORT_SCHEMA)
    assert report["ok"] is False and report["counts"] == {"pass": 3, "fail": 1, "skip": 1}


@pytest.mark.req("REQ-PLT-001")
def test_cli_json_only_and_unknown_group():
    out = io.StringIO()
    code = main(["selftest", "--only", "text", "--json"], stdin=io.StringIO(), stdout=out)
    report = json.loads(out.getvalue())
    jsonschema.validate(report, REPORT_SCHEMA)
    assert code == 0 and report["ok"]
    assert {r["group"] for r in report["results"]} == {"text"}
    out = io.StringIO()
    assert main(["selftest", "--only", "nope"], stdin=io.StringIO(), stdout=out) == 2
    assert "unknown self-test group" in out.getvalue()


@pytest.mark.req("REQ-PLT-001", "REQ-REPO-005")
def test_quick_self_test_passes():
    results = run_checks(quick=True)
    failed = [(r.group, r.name, r.detail) for r in results if r.status == "fail"]
    assert failed == []
    groups = {r.group for r in results}
    expected = {"install", "text", "engine", "corpus", "env", "determinism", "leakage", "eval"}
    assert expected <= groups
