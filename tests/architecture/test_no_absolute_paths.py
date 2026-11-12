import ast
import re

import pytest

pytestmark = [pytest.mark.fast, pytest.mark.req("REQ-PLT-003")]

ABSOLUTE = re.compile(r"(?<![A-Za-z])[A-Za-z]:[\\/]|^/(home|mnt|Users|d|c)/")


def _string_literals(tree: ast.AST):
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            yield node


def test_no_absolute_paths_in_package(repo_root):
    offenders = []
    for path in sorted((repo_root / "src").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in _string_literals(tree):
            if ABSOLUTE.search(node.value):
                offenders.append(f"{path.relative_to(repo_root)}:{node.lineno}: {node.value!r}")
    assert offenders == []


@pytest.mark.parametrize("text", ["C:\\data", "D:/artigo", "/home/user/x", "/mnt/d/x"])
def test_pattern_detects_absolute(text):
    assert ABSOLUTE.search(text)


@pytest.mark.parametrize("text", ["https://artigo.app/", "runs/x", "a: b", "n:2100"])
def test_pattern_ignores_relative_and_urls(text):
    assert not ABSOLUTE.search(text)
