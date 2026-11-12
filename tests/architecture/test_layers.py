import configparser
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.fast, pytest.mark.req("REQ-REPO-001")]

EXPECTED = [
    "cli",
    "analysis",
    "evaluation",
    "agents",
    "wrappers",
    "envs",
    "services",
    "corpus | scenarios | synthetic",
    "engine",
    "text",
    "core",
]


def test_layer_order_matches_sdd(repo_root):
    parser = configparser.ConfigParser()
    parser.read(repo_root / ".importlinter", encoding="utf-8")
    layers = [
        line.strip()
        for line in parser["importlinter:contract:layers"]["layers"].splitlines()
        if line.strip()
    ]
    assert layers == EXPECTED


def test_every_layer_is_a_package(repo_root):
    for entry in EXPECTED:
        for layer in entry.split("|"):
            assert (repo_root / "src" / "artigo_env" / layer.strip() / "__init__.py").is_file()


def test_lint_imports_passes(repo_root):
    scripts = Path(sys.executable).parent
    executable = shutil.which("lint-imports", path=str(scripts)) or shutil.which("lint-imports")
    assert executable, "lint-imports not installed"
    result = subprocess.run([executable], cwd=repo_root, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
