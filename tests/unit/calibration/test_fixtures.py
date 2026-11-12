from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from synthetic import TEXT, TITLE, Guess, build_session

from tools.calibration.cli import main
from tools.calibration.fixtures import build_fixture, dump_fixture, load_fixtures, store_article
from tools.calibration.session import load_session

GUESSES = [
    Guess("rio", True, (0, 3, 22)),
    Guess("de", False, (), "não conta"),
    Guess("rio", False, (), "já foi"),
    Guess("vale", True, (7, 19)),
    Guess("zzz", True, ()),
]


def run_cli(session: Path, tmp_path: Path, version: str = "cal-v1") -> None:
    code = main(
        [
            "process",
            str(session),
            "--version",
            version,
            "--fixtures-dir",
            str(tmp_path / "fixtures"),
            "--data-root-dir",
            str(tmp_path / "calibration"),
        ]
    )
    assert code == 0


def snapshot_outputs(tmp_path: Path) -> dict[str, bytes]:
    return {
        str(path.relative_to(tmp_path)): path.read_bytes()
        for path in sorted(tmp_path.rglob("*"))
        if path.is_file() and path.suffix in {".json", ".jsonl", ".md"} and "rec" not in path.parts
    }


@pytest.mark.fast
@pytest.mark.req("REQ-CAL-001")
def test_reprocessing_is_byte_identical(tmp_path: Path) -> None:
    session = build_session(tmp_path / "rec", GUESSES)
    run_cli(session, tmp_path)
    first = {k: v for k, v in snapshot_outputs(tmp_path).items() if not k.startswith("rec")}
    run_cli(session, tmp_path)
    second = {k: v for k, v in snapshot_outputs(tmp_path).items() if not k.startswith("rec")}
    assert first == second
    names = sorted(first)
    assert any(n.startswith("fixtures") and n.endswith(".json") for n in names)
    assert any(n.endswith("divergences.jsonl") for n in names)
    assert any(n.endswith("report.md") for n in names)
    assert any("articles" in n for n in names)
    assert b"\r\n" not in b"".join(first.values())


@pytest.mark.fast
@pytest.mark.req("REQ-CAL-003")
def test_fixture_and_report_do_not_leak_the_article(tmp_path: Path) -> None:
    session = build_session(tmp_path / "rec", GUESSES)
    run_cli(session, tmp_path)
    fixture_text = next((tmp_path / "fixtures").glob("*.json")).read_text(encoding="utf-8")
    report_text = next((tmp_path / "calibration").rglob("report.md")).read_text(encoding="utf-8")
    for produced in (fixture_text, report_text):
        assert TITLE not in produced
        for start in range(len(TEXT) - 19):
            assert TEXT[start : start + 20] not in produced
    fixture = json.loads(fixture_text)
    assert fixture["article_ref"]["sha256"] == hashlib.sha256(TEXT.encode("utf-8")).hexdigest()
    assert fixture["article_ref"]["title_sha256"] == hashlib.sha256(TITLE.encode()).hexdigest()
    assert str(tmp_path) not in fixture_text + report_text
    assert "Nenhuma" in report_text


@pytest.mark.fast
@pytest.mark.req("REQ-CAL-004")
def test_load_fixtures_does_not_mix_versions(tmp_path: Path) -> None:
    fixtures = tmp_path / "fixtures"
    fixtures.mkdir()
    for name, version in (("b", "cal-v1"), ("a", "cal-v1"), ("c", "cal-v2")):
        item = {"calibration_version": version, "source_session": name, "guesses": []}
        (fixtures / f"{name}.json").write_text(json.dumps(item), encoding="utf-8")
    assert [f["source_session"] for f in load_fixtures("cal-v1", fixtures)] == ["a", "b"]
    assert [f["source_session"] for f in load_fixtures("cal-v2", fixtures)] == ["c"]
    assert load_fixtures("cal-v3", fixtures) == []
    assert load_fixtures("cal-v1", tmp_path / "missing") == []


@pytest.mark.fast
@pytest.mark.req("REQ-CAL-001")
def test_fixture_values_for_known_session(tmp_path: Path) -> None:
    session = load_session(build_session(tmp_path, GUESSES))
    fixture = build_fixture(session, "cal-v1")
    assert fixture["fixture_version"] == 1
    assert fixture["calibration_version"] == "cal-v1"
    assert fixture["bundle_hash"] == "0a1b2c3d"
    assert fixture["source_session"] == session.path.name
    assert [g["raw"] for g in fixture["guesses"]] == ["rio", "de", "rio", "vale", "zzz"]
    expected = [g["expected"] for g in fixture["guesses"]]
    assert expected == [
        {"counted": True, "count": 3, "revealed_token_indices": [0, 3, 22], "message": None},
        {"counted": False, "count": 0, "revealed_token_indices": [], "message": "não conta"},
        {"counted": False, "count": 0, "revealed_token_indices": [], "message": "já foi"},
        {"counted": True, "count": 2, "revealed_token_indices": [7, 19], "message": None},
        {"counted": True, "count": 0, "revealed_token_indices": [], "message": None},
    ]
    dumped = dump_fixture(fixture)
    assert dumped.endswith("}\n")
    assert json.loads(dumped) == fixture


@pytest.mark.fast
@pytest.mark.req("REQ-CAL-001")
def test_truncated_session_is_marked_unreliable(tmp_path: Path) -> None:
    path = build_session(tmp_path, GUESSES[:2], complete=False)
    events = path / "events.jsonl"
    lines = events.read_text(encoding="utf-8").splitlines()
    # Drop the last guess's quiescence and snapshot, as if the process died mid-guess.
    events.write_text("\n".join(lines[:-2]) + "\n", encoding="utf-8")
    fixture = build_fixture(load_session(path), "cal-v1")
    assert fixture["guesses"][0]["expected"].get("unreliable") is None
    assert fixture["guesses"][1]["expected"]["unreliable"] is True


@pytest.mark.fast
@pytest.mark.req("REQ-CAL-003")
def test_store_article_is_atomic_and_hash_addressed(tmp_path: Path) -> None:
    digest = store_article(TITLE, TEXT, tmp_path / "cal-v1")
    assert digest == hashlib.sha256(TEXT.encode("utf-8")).hexdigest()
    stored = tmp_path / "cal-v1" / "articles" / f"{digest}.json"
    assert json.loads(stored.read_text(encoding="utf-8")) == {"title": TITLE, "text": TEXT}
    assert not list(tmp_path.rglob("*.tmp"))
