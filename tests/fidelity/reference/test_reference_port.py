"""The Python rule interpreter must reproduce the game's lemmatizer exactly (REQ-TXT-012)."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest

pytestmark = [pytest.mark.local, pytest.mark.req("REQ-TXT-012", "DEC-041")]


def _reference_dir() -> Path | None:
    root = os.environ.get("ARTIGO_DATA_ROOT", "").strip()
    return Path(root) / "reference" / "lemmatizer" if root else None


def _lemmatizers_importable() -> bool:
    try:
        return importlib.util.find_spec("artigo_env.text.lemmatizers.base") is not None
    except ModuleNotFoundError:
        return False


def test_reference_local_port_matches_original_bundle_output() -> None:
    directory = _reference_dir()
    if directory is None:
        pytest.skip("ARTIGO_DATA_ROOT is not set")
    tables_path = directory / "tables.json"
    forms_path = directory / "reference_forms.tsv"
    if not tables_path.is_file() or not forms_path.is_file():
        pytest.skip(f"reference artifacts missing under {directory} (run tools/lemma_tables)")
    if not _lemmatizers_importable():
        pytest.skip("artigo_env.text.lemmatizers is not importable yet")

    from artigo_env.text.lemmatizers.base import RuleLemmatizer
    from artigo_env.text.lemmatizers.tables import load_tables

    lemmatizer = RuleLemmatizer("reference-local", load_tables(tables_path))
    mismatches: list[str] = []
    total = 0
    for line in forms_path.read_text(encoding="utf-8").split("\n"):
        if not line:
            continue
        form, _, expected = line.partition("\t")
        total += 1
        got = lemmatizer.lemma(form)
        if got != expected:
            mismatches.append(f"{form!r}: port={got!r} original={expected!r}")
    assert total > 0, "reference_forms.tsv is empty"
    assert not mismatches, f"{len(mismatches)} of {total} forms disagree; first 20:\n" + "\n".join(
        mismatches[:20]
    )
