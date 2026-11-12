"""S11 report: counts per stage, reason and source, and distributions (SDD 06 §3; REQ-COR-015).

``report.json`` is deterministic (canonical JSON, no date, no timing) so that a rebuild gives the
same bytes. ``report.md`` is the human-readable view; it is the only file of a build that
carries a date, the wall time and the peak memory.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa

from artigo_env.core.config import canonical_json
from artigo_env.corpus.stages.normalize import PUBLIC_SUFFIXES

__all__ = [
    "PERCENTILES",
    "REPORT_VERSION",
    "StageCounter",
    "check_report",
    "distributions",
    "domain_suffix",
    "render_markdown",
    "write_report",
]

REPORT_VERSION = 1
PERCENTILES = (5, 25, 50, 75, 95)
MAX_TITLE_CONTENT_BIN = 6


class StageCounter:
    """``{stage: {"in": n, "out": n, "removed": {reason: n}}}`` in first-seen stage order."""

    def __init__(self) -> None:
        self.stages: dict[str, dict[str, Any]] = {}

    def add(
        self,
        stage: str,
        n_in: int,
        n_out: int,
        removed: Mapping[str, int] | None = None,
        **flags: Any,
    ) -> None:
        entry = self.stages.setdefault(stage, {"in": 0, "out": 0, "removed": {}})
        entry["in"] += int(n_in)
        entry["out"] += int(n_out)
        for reason, count in (removed or {}).items():
            entry["removed"][reason] = entry["removed"].get(reason, 0) + int(count)
        entry.update(flags)

    def merge(self, other: StageCounter | Mapping[str, Mapping[str, Any]]) -> None:
        stages = other.stages if isinstance(other, StageCounter) else other
        for stage, entry in stages.items():
            extra = {k: v for k, v in entry.items() if k not in ("in", "out", "removed")}
            self.add(stage, entry["in"], entry["out"], entry.get("removed", {}), **extra)

    def to_dict(self) -> dict[str, dict[str, Any]]:
        return json.loads(json.dumps(self.stages))  # type: ignore[no-any-return]

    def order(self) -> list[str]:
        return list(self.stages)

    def reorder(self, order: Sequence[str]) -> None:
        """Put the stages in ``order`` (stages not listed keep their place at the end)."""
        rank = {name: i for i, name in enumerate(order)}
        names = sorted(self.stages, key=lambda name: rank.get(name, len(rank)))
        self.stages = {name: self.stages[name] for name in names}


def domain_suffix(host: str | None) -> str:
    """Public suffix of a host (``.com.br``, ``.pt``, ...), its last label, or ``none``."""
    if not host:
        return "none"
    for suffix in PUBLIC_SUFFIXES:
        if host.endswith(suffix):
            return suffix
    return "." + host.rsplit(".", 1)[-1] if "." in host else "none"


def _counts(values: Iterable[Any]) -> dict[str, int]:
    out: dict[str, int] = {}
    for value in values:
        key = (
            "none"
            if value is None
            else str(value).lower()
            if isinstance(value, bool)
            else str(value)
        )
        out[key] = out.get(key, 0) + 1
    return dict(sorted(out.items()))


def distributions(docs: pa.Table) -> dict[str, Any]:
    """Distributions over the compiled documents (a table with ``DOC_SCHEMA`` columns)."""
    n_body = np.asarray(docs.column("n_body_tokens").to_pylist() or [], dtype=np.int64)
    if n_body.size:
        body: dict[str, int] = {
            f"p{p}": int(np.percentile(n_body, p, method="nearest")) for p in PERCENTILES
        }
        body["min"] = int(n_body.min())
        body["max"] = int(n_body.max())
        body["total"] = int(n_body.sum())
    else:
        body = {f"p{p}": 0 for p in PERCENTILES} | {"min": 0, "max": 0, "total": 0}
    content_bins: dict[str, int] = {str(i): 0 for i in range(0, MAX_TITLE_CONTENT_BIN + 1)}
    content_bins[f">{MAX_TITLE_CONTENT_BIN}"] = 0
    for value in docs.column("n_title_content").to_pylist():
        n = int(value or 0)
        key = str(n) if n <= MAX_TITLE_CONTENT_BIN else f">{MAX_TITLE_CONTENT_BIN}"
        content_bins[key] += 1
    return {
        "body_tokens": body,
        "title_content_words": content_bins,
        "variant": _counts(docs.column("variant").to_pylist()),
        "domain_suffix": _counts(domain_suffix(h) for h in docs.column("url_domain").to_pylist()),
        "split": _counts(docs.column("split").to_pylist()),
        "length_band": _counts(docs.column("length_band").to_pylist()),
        "domain": _counts(docs.column("domain").to_pylist()),
        "title_in_body": _counts(docs.column("title_in_body").to_pylist()),
    }


# --- schema --------------------------------------------------------------------------------

_TOP_KEYS = (
    "report_version",
    "build_id",
    "name",
    "kind",
    "trainable",
    "config",
    "sources",
    "decontam",
    "stage_order",
    "stages",
    "by_source",
    "n_docs",
    "distributions",
    "notes",
)
_DIST_KEYS = (
    "body_tokens",
    "title_content_words",
    "variant",
    "domain_suffix",
    "split",
    "length_band",
    "domain",
    "title_in_body",
)


def _check_stages(stages: Any, where: str, problems: list[str]) -> None:
    if not isinstance(stages, Mapping):
        problems.append(f"{where}: expected a mapping of stages")
        return
    for name, entry in stages.items():
        label = f"{where}.{name}"
        if not isinstance(entry, Mapping):
            problems.append(f"{label}: expected a mapping")
            continue
        for key in ("in", "out"):
            if not isinstance(entry.get(key), int) or entry[key] < 0:
                problems.append(f"{label}.{key}: expected a non-negative integer")
        removed = entry.get("removed")
        if not isinstance(removed, Mapping) or not all(
            isinstance(k, str) and isinstance(v, int) and v >= 0 for k, v in removed.items()
        ):
            problems.append(f"{label}.removed: expected {{reason: count}}")
            continue
        counts_ok = isinstance(entry.get("in"), int) and isinstance(entry.get("out"), int)
        if counts_ok and entry["in"] - entry["out"] != sum(removed.values()):
            problems.append(f"{label}: in - out != sum(removed)")


def check_report(report: Mapping[str, Any]) -> list[str]:
    """Problems with the structure of a ``report.json`` document (empty list = valid)."""
    problems: list[str] = []
    for key in _TOP_KEYS:
        if key not in report:
            problems.append(f"missing key {key!r}")
    if problems:
        return problems
    if report["report_version"] != REPORT_VERSION:
        problems.append("unsupported report_version")
    if not isinstance(report["build_id"], str) or len(report["build_id"]) != 16:
        problems.append("build_id must be 16 hex characters")
    _check_stages(report["stages"], "stages", problems)
    order = list(report["stage_order"])
    if len(set(order)) != len(order) or set(order) != set(report["stages"]):
        problems.append("stage_order must list each stage once")
    by_source = report["by_source"]
    if not isinstance(by_source, Mapping):
        problems.append("by_source: expected a mapping")
    else:
        for source, stages in by_source.items():
            _check_stages(stages, f"by_source.{source}", problems)
    decontam = report["decontam"]
    if not isinstance(decontam, Mapping) or decontam.get("status") not in (
        "applied",
        "skipped",
        "not-applicable",
    ):
        problems.append("decontam.status must be applied, skipped or not-applicable")
    dist = report["distributions"]
    if not isinstance(dist, Mapping):
        problems.append("distributions: expected a mapping")
    else:
        for key in _DIST_KEYS:
            if key not in dist:
                problems.append(f"distributions: missing {key!r}")
    stages = report["stages"]
    if isinstance(stages, Mapping) and order and order[-1] in stages:
        last = stages[order[-1]]
        if isinstance(last, Mapping) and last.get("out") != report["n_docs"]:
            problems.append("n_docs differs from the output of the last stage")
    return problems


# --- markdown ------------------------------------------------------------------------------


def _stage_rows(stages: Mapping[str, Any], order: Sequence[str]) -> list[str]:
    rows = ["| Stage | In | Out | Removed |", "|---|---:|---:|---|"]
    for name in order:
        entry = stages[name]
        removed = ", ".join(f"{r} {n:,}" for r, n in entry["removed"].items() if n) or "-"
        if entry.get("skipped"):
            removed = "skipped"
        rows.append(f"| {name} | {entry['in']:,} | {entry['out']:,} | {removed} |")
    return rows


def render_markdown(report: Mapping[str, Any], run: Mapping[str, Any]) -> str:
    """Human-readable report; ``run`` holds the date, wall time, memory and reuse counts."""
    lines = [
        f"# Corpus build {report['build_id']} ({report['name']})",
        "",
        f"- Generated: {run.get('generated', '?')}",
        f"- Kind: {report['kind']} (trainable: {str(report['trainable']).lower()})",
        f"- Documents compiled: {report['n_docs']:,}",
    ]
    if "wall_seconds" in run:
        lines.append(f"- Wall time: {run['wall_seconds']:.1f} s")
    if run.get("peak_memory_bytes"):
        lines.append(
            f"- Peak memory (largest process): {run['peak_memory_bytes'] / 2**20:,.0f} MiB"
        )
    if "shards_processed" in run:
        lines.append(
            f"- Shards processed: {run['shards_processed']}, reused: {run.get('shards_reused', 0)}"
        )
    if "size_bytes" in run:
        lines.append(f"- Size on disk: {run['size_bytes'] / 2**20:,.1f} MiB")
    decontam = report["decontam"]
    note = f"decontam: {decontam['status']}"
    if decontam.get("w_index"):
        note += f" (W index {decontam['w_index']})"
    elif decontam["status"] == "skipped":
        note += " (no W index configured)"
    lines += [
        f"- {note}",
        "",
        "## Stages",
        "",
        *_stage_rows(report["stages"], report["stage_order"]),
    ]
    for source, stages in report["by_source"].items():
        lines += ["", f"### Source {source}", "", *_stage_rows(stages, list(stages))]
    dist = report["distributions"]
    lines += ["", "## Distributions", ""]
    for key in _DIST_KEYS:
        value = dist[key]
        rendered = ", ".join(f"{k} {v:,}" for k, v in value.items())
        lines.append(f"- {key}: {rendered}")
    if report.get("w"):
        lines += ["", "## Evaluation set W", ""]
        lines += [f"- {k}: {v}" for k, v in report["w"].items()]
    if report["notes"]:
        lines += ["", "## Notes", ""] + [f"- {note}" for note in report["notes"]]
    return "\n".join(lines) + "\n"


def write_report(directory: Path, report: Mapping[str, Any], run: Mapping[str, Any]) -> None:
    """Write ``report.md`` then ``report.json`` (the latter marks a finished build)."""
    problems = check_report(report)
    if problems:
        raise ValueError("invalid report: " + "; ".join(problems))
    (directory / "report.md").write_text(render_markdown(report, run), encoding="utf-8")
    tmp = directory / "report.json.tmp"
    tmp.write_bytes((canonical_json(report) + "\n").encode("utf-8"))
    tmp.replace(directory / "report.json")
