"""Registry and runner of the product self-test (``artigo-env selftest``).

A check is a function ``(ctx: Context) -> str | None`` registered with ``@check(group, name)``:
returning means *pass* (the string is the one-line detail), raising ``SkipCheck`` means *skip*
(the message is the reason) and raising anything else means *fail*. One failing check never stops
the others. ``full_only=True`` checks are skipped by ``--quick``.

Adding a check for a new subsystem is one decorated function in any module listed in
``BUILTIN_MODULES`` (or a new module appended to it).
"""

from __future__ import annotations

import importlib
import shutil
import tempfile
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

Status = Literal["pass", "fail", "skip"]

# Display order of the groups; groups not listed here come after, in registration order.
GROUP_ORDER = ("install", "text", "engine", "corpus", "env", "determinism", "leakage", "eval")
BUILTIN_MODULES = ("artigo_env.selftest.checks", "artigo_env.selftest.checks_env")


class SkipCheck(Exception):  # noqa: N818 - a signal, not an error
    """Raised by a check that cannot run here (the message says why)."""


@dataclass
class Context:
    """What a check may use: the mode and a private temporary directory."""

    quick: bool
    tmp_dir: Path
    _counter: int = 0

    def workdir(self, name: str) -> Path:
        """A fresh empty directory under the run's temporary folder."""
        self._counter += 1
        path = self.tmp_dir / f"{self._counter:03d}-{name}"
        path.mkdir(parents=True)
        return path


CheckFunc = Callable[[Context], "str | None"]


@dataclass(frozen=True)
class Check:
    group: str
    name: str
    func: CheckFunc
    full_only: bool = False


@dataclass(frozen=True)
class CheckResult:
    group: str
    name: str
    status: Status
    detail: str
    seconds: float


@dataclass
class Registry:
    _checks: list[Check] = field(default_factory=list)

    def add(self, item: Check) -> None:
        if any(c.group == item.group and c.name == item.name for c in self._checks):
            raise ValueError(f"duplicate self-test check {item.group}/{item.name}")
        self._checks.append(item)

    def checks(self) -> list[Check]:
        order = {g: i for i, g in enumerate(GROUP_ORDER)}
        indexed = list(enumerate(self._checks))
        indexed.sort(key=lambda p: (order.get(p[1].group, len(order)), p[0]))
        return [c for _, c in indexed]

    def groups(self) -> list[str]:
        seen: list[str] = []
        for c in self.checks():
            if c.group not in seen:
                seen.append(c.group)
        return seen


REGISTRY = Registry()
_LOAD_ERRORS: list[CheckResult] = []
_loaded = False


def check(
    group: str, name: str, *, full_only: bool = False, registry: Registry | None = None
) -> Callable[[CheckFunc], CheckFunc]:
    """Register the decorated function as the self-test check ``group/name``."""

    def decorate(func: CheckFunc) -> CheckFunc:
        (REGISTRY if registry is None else registry).add(Check(group, name, func, full_only))
        return func

    return decorate


def load_builtin_checks() -> list[CheckResult]:
    """Import the built-in check modules once; an import failure becomes a failed result."""
    global _loaded
    if not _loaded:
        _loaded = True
        for module in BUILTIN_MODULES:
            start = time.perf_counter()
            try:
                importlib.import_module(module)
            except Exception as exc:  # noqa: BLE001 - reported, never raised
                _LOAD_ERRORS.append(
                    CheckResult(
                        "install",
                        f"import {module.rsplit('.', 1)[-1]}",
                        "fail",
                        _describe(exc),
                        time.perf_counter() - start,
                    )
                )
    return list(_LOAD_ERRORS)


def _describe(exc: BaseException) -> str:
    text = str(exc).strip().splitlines()[0] if str(exc).strip() else ""
    return f"{type(exc).__name__}: {text}" if text else type(exc).__name__


def run_check(item: Check, ctx: Context) -> CheckResult:
    if ctx.quick and item.full_only:
        return CheckResult(item.group, item.name, "skip", "skipped by --quick", 0.0)
    start = time.perf_counter()
    status: Status
    try:
        detail = item.func(ctx)
        status, text = "pass", (detail or "ok")
    except SkipCheck as exc:
        status, text = "skip", str(exc) or "skipped"
    except Exception as exc:  # noqa: BLE001 - a failing check never stops the run
        status, text = "fail", _describe(exc)
    return CheckResult(item.group, item.name, status, text, time.perf_counter() - start)


class UnknownGroupError(ValueError):
    pass


def run_checks(
    *,
    quick: bool = False,
    only: Sequence[str] | None = None,
    registry: Registry | None = None,
) -> list[CheckResult]:
    """Run the selected checks in group order; never raises for a failing check."""
    results: list[CheckResult] = []
    if registry is None:
        results.extend(load_builtin_checks())
        registry = REGISTRY
    checks: Iterable[Check] = registry.checks()
    if only:
        unknown = sorted(set(only) - set(registry.groups()))
        if unknown:
            raise UnknownGroupError(
                f"unknown self-test group(s) {unknown}; known: {registry.groups()}"
            )
        checks = [c for c in checks if c.group in set(only)]
        results = [r for r in results if r.group in set(only)]
    tmp = Path(tempfile.mkdtemp(prefix="artigo-selftest-"))
    try:
        ctx = Context(quick=quick, tmp_dir=tmp)
        for item in checks:
            results.append(run_check(item, ctx))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return results
