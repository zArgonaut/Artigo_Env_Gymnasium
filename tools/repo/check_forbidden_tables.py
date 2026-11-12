"""pre-commit check: game lemmatizer tables must never enter Git (REQ-TXT-012).

Two rules, neither of which contains game data:

* generic: a text line that contains the word ``connosco`` and more than 40 ``|``-separated
  alternatives looks like a pasted exception list;
* local signatures: when ``<ARTIGO_DATA_ROOT>/reference/lemmatizer/signatures.txt`` exists, any
  file containing one of its lines as a substring is forbidden.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Sequence
from pathlib import Path

from tools.repo._env import load_dotenv_min

MAX_ALTERNATIVES = 40
GENERIC_MARKER = "connosco"
REPO_ROOT = Path(__file__).resolve().parents[2]
SIGNATURES_RELATIVE = ("reference", "lemmatizer", "signatures.txt")


def _read_text(path: Path) -> str | None:
    """Return the file text, or ``None`` for unreadable, binary or undecodable files."""
    try:
        data = path.read_bytes()
    except OSError:
        return None
    if b"\x00" in data:
        return None
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return None


def check(paths: Sequence[Path], signatures: Sequence[str]) -> list[str]:
    """Return ``path: reason`` strings for every forbidden file in ``paths``."""
    wanted = [s for s in signatures if s.strip()]
    offenders: list[str] = []
    for path in paths:
        text = _read_text(Path(path))
        if text is None:
            continue
        reason: str | None = None
        for number, line in enumerate(text.splitlines(), start=1):
            if GENERIC_MARKER in line and line.count("|") + 1 > MAX_ALTERNATIVES:
                reason = f"line {number} looks like a pasted game exception list"
                break
        if reason is None:
            for index, signature in enumerate(wanted, start=1):
                if signature in text:
                    reason = f"contains local reference signature #{index}"
                    break
        if reason is not None:
            offenders.append(f"{path}: {reason}")
    return offenders


def load_signatures() -> list[str]:
    """Load signatures from the data root (environment first, then the repo ``.env``)."""
    root = os.environ.get("ARTIGO_DATA_ROOT", "").strip()
    if not root:
        root = load_dotenv_min(REPO_ROOT / ".env").get("ARTIGO_DATA_ROOT", "").strip()
    if not root:
        return []
    file = Path(root).joinpath(*SIGNATURES_RELATIVE)
    try:
        text = file.read_text(encoding="utf-8")
    except OSError:
        return []
    return [line for line in text.splitlines() if line.strip()]


def main(argv: Sequence[str]) -> int:
    offenders = check([Path(a) for a in argv], load_signatures())
    for offender in offenders:
        print(offender)
    if offenders:
        print("Game lemmatizer tables must stay under ARTIGO_DATA_ROOT/reference/.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
