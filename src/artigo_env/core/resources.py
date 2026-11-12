"""Access to the data bundled inside the installed package (``artigo_env/assets``)."""

from __future__ import annotations

from importlib.resources import files
from importlib.resources.abc import Traversable


def resource_path(*parts: str) -> Traversable:
    """Return a handle to a file or directory below ``artigo_env/assets``.

    Works from a source checkout and from an installed wheel. Parts must be plain names.
    """
    for part in parts:
        if not part or "/" in part or "\\" in part or part in {".", ".."}:
            raise ValueError(f"resource path parts must be plain names, got {part!r}")
    node: Traversable = files("artigo_env") / "assets"
    for part in parts:
        node = node / part
    return node
