import ast
import re
import tomllib

import pytest

pytestmark = pytest.mark.fast

FORBIDDEN = {"hydra-core", "hydra", "playwright", "optuna", "wandb"}
OUTSIDE_PACKAGE = {"tools", "runner", "sandbox", "tests"}


@pytest.fixture(scope="module")
def pyproject(request):
    root = request.config.rootpath
    return tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))


def _names(requirements):
    return {re.split(r"[\[<>=!~; ]", req, maxsplit=1)[0].lower() for req in requirements}


@pytest.mark.req("REQ-REPO-003")
def test_runtime_dependencies_exclude_forbidden(pyproject):
    assert _names(pyproject["project"]["dependencies"]) & FORBIDDEN == set()
    for extra in pyproject["project"].get("optional-dependencies", {}).values():
        assert _names(extra) & FORBIDDEN == set()


@pytest.mark.req("REQ-REPO-008")
def test_wheel_only_packages_src(pyproject):
    wheel = pyproject["tool"]["hatch"]["build"]["targets"]["wheel"]
    assert wheel["packages"] == ["src/artigo_env"]


@pytest.mark.req("REQ-REPO-005")
def test_no_custom_build_hooks(pyproject, repo_root):
    assert pyproject["build-system"]["build-backend"] == "hatchling.build"
    assert "hooks" not in pyproject["tool"]["hatch"]["build"]
    assert not (repo_root / "setup.py").exists()
    assert not (repo_root / "hatch_build.py").exists()


@pytest.mark.req("REQ-REPO-008", "REQ-REPO-003")
def test_package_never_imports_outside_itself(repo_root):
    offenders = []
    for path in sorted((repo_root / "src").rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                roots = {alias.name.split(".")[0] for alias in node.names}
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                roots = {node.module.split(".")[0]}
            else:
                continue
            if roots & (OUTSIDE_PACKAGE | FORBIDDEN):
                offenders.append(f"{path.relative_to(repo_root)}:{node.lineno}")
    assert offenders == []
