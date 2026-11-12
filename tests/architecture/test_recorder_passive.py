"""The recorder only observes the page (REQ-REC-001): no Playwright action methods anywhere."""

import ast

import pytest

pytestmark = [pytest.mark.fast, pytest.mark.req("REQ-REC-001")]

FORBIDDEN_CALLS = frozenset(
    {
        "click",
        "dblclick",
        "fill",
        "type",
        "press",
        "check",
        "uncheck",
        "set_checked",
        "select_option",
        "hover",
        "tap",
        "focus",
        "drag_and_drop",
        "dispatch_event",
        "route",
        "unroute",
        "route_from_har",
        "set_extra_http_headers",
        "reload",
        "go_back",
        "go_forward",
        "fetch",
        "post",
        "put",
        "patch",
        "delete",
        "head",
    }
)
# Attribute access alone is already an action surface (page.keyboard.press, context.request.get).
FORBIDDEN_ATTRIBUTES = frozenset({"keyboard", "mouse", "touchscreen", "request"})
FORBIDDEN_JS = (
    "preventDefault",
    "dispatchEvent",
    ".click(",
    ".submit(",
    "innerHTML =",
    "fetch(",
    "XMLHttpRequest",
)


def violations(source: str) -> list[str]:
    """Forbidden calls/attributes in ``source`` plus the number of ``goto`` calls as a marker."""
    found: list[str] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Attribute) and node.attr in FORBIDDEN_ATTRIBUTES:
            found.append(f"line {node.lineno}: attribute .{node.attr}")
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in FORBIDDEN_CALLS
        ):
            found.append(f"line {node.lineno}: call .{node.func.attr}()")
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "get"
            and ast.unparse(node.func.value).endswith("request")
        ):
            found.append(f"line {node.lineno}: call request.get()")
    return found


def count_goto(source: str) -> int:
    return sum(
        1
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "goto"
    )


def test_no_action_methods_in_recorder(repo_root):
    offenders = []
    for path in sorted((repo_root / "tools" / "recorder").rglob("*.py")):
        for item in violations(path.read_text(encoding="utf-8")):
            offenders.append(f"{path.relative_to(repo_root)}: {item}")
    assert offenders == []


def test_goto_is_called_exactly_once(repo_root):
    total = sum(
        count_goto(path.read_text(encoding="utf-8"))
        for path in (repo_root / "tools" / "recorder").rglob("*.py")
    )
    assert total == 1


@pytest.mark.parametrize(
    "snippet",
    [
        "page.click('#x')",
        "page.fill('#x', 'a')",
        "page.keyboard.press('Enter')",
        "page.mouse.move(1, 2)",
        "context.route('**', handler)",
        "context.request.get('http://x')",
        "page.reload()",
        "request.get('http://x')",
    ],
)
def test_checker_detects_actions(snippet):
    assert violations(snippet)


def test_checker_accepts_observation():
    assert violations("page.evaluate('1')\npage.screenshot()\ncontext.on('page', f)") == []
    assert violations("location.get('url')") == []


def test_inject_script_is_passive(repo_root):
    source = (repo_root / "tools" / "recorder" / "capture" / "inject.js").read_text(
        encoding="utf-8"
    )
    assert [token for token in FORBIDDEN_JS if token in source] == []
    assert "__artigoRecorder(" in source
