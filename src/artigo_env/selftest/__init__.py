"""Product self-test (``artigo-env selftest``): checks that every subsystem works on this machine.

Plain Python (no pytest at run time), runs from an installed wheel against the packaged assets,
needs neither network nor ``ARTIGO_DATA_ROOT``. See ``registry`` for how checks are added.
"""

from __future__ import annotations

from artigo_env.selftest.registry import (
    REGISTRY,
    Check,
    CheckResult,
    Context,
    Registry,
    SkipCheck,
    check,
    run_checks,
)

__all__ = [
    "REGISTRY",
    "Check",
    "CheckResult",
    "Context",
    "Registry",
    "SkipCheck",
    "check",
    "run_checks",
]
