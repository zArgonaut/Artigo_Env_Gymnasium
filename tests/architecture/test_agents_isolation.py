"""Agents see only obs, the allow-listed info and the mask (REQ-AGT-002): static search."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = [pytest.mark.fast, pytest.mark.req("REQ-AGT-002")]

FORBIDDEN_TEXT = ("unwrapped", "episode_record", "artigo_env.corpus", "artigo_env.envs")
FORBIDDEN_PACKAGES = ("corpus", "envs", "scenarios", "evaluation", "cli", "analysis")


def agent_sources(repo_root: Path) -> list[Path]:
    files = sorted((repo_root / "src" / "artigo_env" / "agents").rglob("*.py"))
    assert files, "no agent sources found"
    return files


def test_no_forbidden_text_in_agents(repo_root):
    offences = [
        f"{path.relative_to(repo_root).as_posix()}: {word}"
        for path in agent_sources(repo_root)
        for word in FORBIDDEN_TEXT
        if word in path.read_text(encoding="utf-8")
    ]
    assert not offences, offences


def test_agents_import_no_upper_or_corpus_layer(repo_root):
    offences = []
    for path in agent_sources(repo_root):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            for name in names:
                parts = name.split(".")
                if parts[0] == "artigo_env" and len(parts) > 1 and parts[1] in FORBIDDEN_PACKAGES:
                    offences.append(f"{path.name}: {name}")
    assert not offences, offences
