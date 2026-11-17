"""Rendering of self-test results: a readable table or a machine-readable JSON report."""

from __future__ import annotations

import json
import platform
import sys
from collections.abc import Sequence
from dataclasses import asdict
from typing import Any

import artigo_env
from artigo_env.selftest.registry import CheckResult

REPORT_FORMAT_VERSION = 1
_DETAIL_WIDTH = 72


def counts(results: Sequence[CheckResult]) -> dict[str, int]:
    out = {"pass": 0, "fail": 0, "skip": 0}
    for r in results:
        out[r.status] += 1
    return out


def exit_code(results: Sequence[CheckResult]) -> int:
    """0 when nothing failed, 1 otherwise."""
    return 1 if any(r.status == "fail" for r in results) else 0


def to_json(results: Sequence[CheckResult], *, quick: bool, seconds: float) -> dict[str, Any]:
    """The JSON report (schema version ``REPORT_FORMAT_VERSION``)."""
    return {
        "format_version": REPORT_FORMAT_VERSION,
        "package_version": artigo_env.__version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "executable": sys.executable,
        "quick": quick,
        "ok": exit_code(results) == 0,
        "counts": counts(results),
        "seconds": round(seconds, 3),
        "results": [{**asdict(r), "seconds": round(r.seconds, 3)} for r in results],
    }


def to_json_text(results: Sequence[CheckResult], *, quick: bool, seconds: float) -> str:
    return json.dumps(to_json(results, quick=quick, seconds=seconds), indent=2, ensure_ascii=False)


def _clip(text: str, width: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= width else text[: width - 3] + "..."


def table(results: Sequence[CheckResult], *, quick: bool, seconds: float) -> str:
    """Readable table (group, check, status, time, detail) and a final line with counts."""
    gw = max([5, *(len(r.group) for r in results)])
    nw = max([5, *(len(r.name) for r in results)])
    lines = [
        f"artigo-env {artigo_env.__version__} self-test ({'quick' if quick else 'full'}), "
        f"Python {platform.python_version()} on {platform.system()}",
        "",
        f"{'group':<{gw}}  {'check':<{nw}}  {'status':<6}  {'time':>7}  detail",
        f"{'-' * gw}  {'-' * nw}  {'-' * 6}  {'-' * 7}  {'-' * 6}",
    ]
    for r in results:
        lines.append(
            f"{r.group:<{gw}}  {r.name:<{nw}}  {r.status.upper():<6}  {r.seconds:>6.2f}s  "
            f"{_clip(r.detail, _DETAIL_WIDTH)}"
        )
    c = counts(results)
    verdict = "OK" if exit_code(results) == 0 else "FAILED"
    lines += [
        "",
        f"{verdict}: {c['pass']} passed, {c['fail']} failed, {c['skip']} skipped in {seconds:.1f}s",
    ]
    return "\n".join(lines)
