"""pre-commit check: recordings must never enter Git (REQ-REC-007)."""

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import PurePosixPath

#: Versioned fidelity fixtures live in a test folder that happens to be called ``recordings``.
ALLOWED_PREFIX = ("tests", "fidelity", "recordings")


def check(paths: Sequence[str]) -> str | None:
    """Return an error message when a path is under ``recordings/`` or named ``events.jsonl``.

    ``tests/fidelity/recordings/`` (calibration fixtures, no raw sessions) is exempt from the
    folder rule but not from the ``events.jsonl`` rule.
    """
    for path in paths:
        parts = PurePosixPath(path.replace("\\", "/")).parts
        folders = parts[:-1]
        if folders[: len(ALLOWED_PREFIX)] == ALLOWED_PREFIX:
            folders = folders[len(ALLOWED_PREFIX) :]
        if "recordings" in folders or (parts and parts[-1] == "events.jsonl"):
            return f"{path}: recordings stay in ARTIGO_DATA_ROOT and must not be committed"
    return None


def main(argv: Sequence[str]) -> int:
    error = check(argv)
    if error:
        print(error)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
