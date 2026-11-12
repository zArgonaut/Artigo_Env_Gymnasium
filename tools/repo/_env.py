"""Minimal ``.env`` reader for repository tooling (the package itself only reads os.environ)."""

from __future__ import annotations

from pathlib import Path


def load_dotenv_min(path: Path) -> dict[str, str]:
    """Parse ``KEY=value`` lines; ignores comments and blanks; a missing file yields ``{}``."""
    values: dict[str, str] = {}
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return values
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        values[key.strip()] = value
    return values
