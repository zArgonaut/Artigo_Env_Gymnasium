import pytest

from tools.repo.check_no_recordings import check, main

pytestmark = [pytest.mark.fast, pytest.mark.req("REQ-REC-007")]


@pytest.mark.parametrize(
    "path",
    [
        "recordings/2026-11-12_210301_random/meta.json",
        "data/recordings/x/events.jsonl",
        "events.jsonl",
        "some/dir/events.jsonl",
        "tests\\fidelity\\recordings\\s\\events.jsonl",
        "a/recordings/b.txt",
    ],
)
def test_rejects_recordings(path):
    assert check(["README.md", path]) is not None


@pytest.mark.parametrize(
    "path",
    [
        "README.md",
        "tools/recorder/session.py",
        "tests/fidelity/recordings/test_replay.py",
        "tests/fidelity/recordings/fixtures/.gitkeep",
        "docs/recordings.md",
    ],
)
def test_accepts_other_paths(path):
    assert check([path]) is None


def test_main_exit_codes(capsys):
    assert main(["README.md"]) == 0
    assert main(["recordings/x/meta.json"]) == 1
    assert "recordings" in capsys.readouterr().out
