"""Single entry point for data locations (REQ-PLT-003).

Every data path derives from the ``ARTIGO_DATA_ROOT`` environment variable. The package never
hard-codes absolute paths and never falls back to the current directory.
"""

from __future__ import annotations

import os
from pathlib import Path, PurePosixPath, PureWindowsPath

from artigo_env.core.errors import DataRootNotConfiguredError

ENV_VAR = "ARTIGO_DATA_ROOT"

DATA_SUBDIRS = (
    "raw",
    "corpus",
    "knowledge",
    "recordings",
    "transcripts",
    "runs",
    "calibration",
    "reference",
    "locks",
    "vocab",
    "human",
)


def data_root() -> Path:
    """Return the data root configured in the environment.

    The variable is read on every call. The directory is neither created nor required to exist.
    """
    raw = os.environ.get(ENV_VAR, "").strip()
    if not raw:
        raise DataRootNotConfiguredError(
            f"{ENV_VAR} is not set. Copy .env.example to .env, set an absolute path and load it "
            "into the environment."
        )
    root = Path(raw)
    if not root.is_absolute():
        raise DataRootNotConfiguredError(f"{ENV_VAR} must be an absolute path, got {raw!r}.")
    return root


def data_path(*parts: str) -> Path:
    """Return a path below the data root; parts must be relative and must not escape it."""
    for part in parts:
        windows, posix = PureWindowsPath(part), PurePosixPath(part)
        if windows.drive or windows.root or posix.is_absolute():
            raise ValueError(f"data path parts must be relative, got {part!r}")
        if ".." in windows.parts or ".." in posix.parts:
            raise ValueError(f"data path parts must not contain '..', got {part!r}")
    return data_root().joinpath(*parts)
