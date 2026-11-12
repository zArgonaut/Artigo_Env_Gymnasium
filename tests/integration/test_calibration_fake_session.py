"""End to end: record a session on the fake page, then turn it into a fidelity fixture."""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from urllib.parse import urlsplit

import pytest

from tests.fixtures.fake_game import server
from tools.calibration.cli import main as calibration_main
from tools.recorder.session import RecorderConfig, run_session
from tools.recorder.writer import read_events

pytestmark = pytest.mark.slow

# Unified token positions of the fake article (title first): "Guarda-chuva" is 0, 1; the
# body repeats "guarda" at 3 and 15 and "chuva" at 4 and 16.
GUESSES = ("chuva", "de", "chuva", "guarda")
EXPECTED = [
    {"counted": True, "count": 3, "revealed_token_indices": [1, 4, 16], "message": None},
    {"counted": False, "count": 0, "revealed_token_indices": [], "message": "não conta"},
    {"counted": False, "count": 0, "revealed_token_indices": [], "message": "já foi"},
    {"counted": True, "count": 3, "revealed_token_indices": [0, 3, 15], "message": None},
]


def _guess_screens(out_root: Path) -> int:
    sessions = list(out_root.iterdir()) if out_root.exists() else []
    if not sessions:
        return 0
    try:
        events = read_events(sessions[0])
    except (OSError, ValueError):
        return 0
    return sum(1 for e in events if e["type"] == "screen" and e["payload"]["reason"] == "guess")


@pytest.fixture(scope="module")
def processed(tmp_path_factory):
    sync_api = pytest.importorskip("playwright.sync_api", reason="playwright is not installed")
    try:
        with sync_api.sync_playwright() as playwright:
            playwright.chromium.launch(headless=True).close()
    except Exception as error:
        pytest.skip(f"Playwright Chromium cannot be launched: {error}")

    base = tmp_path_factory.mktemp("calibration-e2e")
    out_root = base / "recordings"
    game = server.start_with_third_party()
    try:
        config = RecorderConfig(
            mode="free",
            start_url=game.base_url + "/",
            out_root=out_root,
            profile_dir=base / "profile",
            allowed_hosts=(urlsplit(game.base_url).netloc,),
            quiescence_ms=150,
            headless=True,
        )

        async def play(page) -> None:
            for number, guess in enumerate(GUESSES, start=1):
                await page.fill("#guess-input", guess)
                await page.press("#guess-input", "Enter")
                deadline = time.monotonic() + 15
                while _guess_screens(out_root) < number and time.monotonic() < deadline:
                    await asyncio.sleep(0.1)

        session_dir = run_session(config, on_ready=play)
    finally:
        game.stop()

    fixtures_dir = base / "fixtures"
    calibration_root = base / "calibration"
    args = [
        "process",
        str(session_dir),
        "--version",
        "cal-test",
        "--fixtures-dir",
        str(fixtures_dir),
        "--data-root-dir",
        str(calibration_root),
    ]
    assert calibration_main(args) == 0
    return session_dir, fixtures_dir, calibration_root, args


@pytest.mark.req("REQ-CAL-001", "REQ-REC-004")
def test_recorded_session_becomes_the_expected_fixture(processed):
    session_dir, fixtures_dir, _, _ = processed
    fixture = json.loads((fixtures_dir / f"{session_dir.name}.json").read_text(encoding="utf-8"))
    assert fixture["fixture_version"] == 1
    assert fixture["calibration_version"] == "cal-test"
    assert fixture["bundle_hash"] == "0a1b2c3d"
    assert fixture["source_session"] == session_dir.name
    assert [guess["raw"] for guess in fixture["guesses"]] == list(GUESSES)
    assert [guess["expected"] for guess in fixture["guesses"]] == EXPECTED


@pytest.mark.req("REQ-REC-001")
def test_recorder_itself_causes_no_dom_mutation(processed):
    # Regression: screenshots used to hide the caret by restyling the input, which polluted
    # the mutation log and fired the quiescence timer before the page answered the next guess.
    session_dir, _, _, _ = processed
    events = read_events(session_dir)
    assert not [e for e in events if "caret-color" in json.dumps(e["payload"])]
    for index, event in enumerate(events):
        if event["type"] == "input.submit":
            following = [e["type"] for e in events[index + 1 :]]
            window = following[: following.index("dom.quiescent")]
            assert set(window) <= {"dom.mutation", "input.key", "input.click"}


@pytest.mark.req("REQ-CAL-002")
def test_real_dom_aligns_with_our_tokenizer(processed):
    session_dir, _, calibration_root, _ = processed
    out_dir = calibration_root / "cal-test" / session_dir.name
    assert (out_dir / "divergences.jsonl").read_text(encoding="utf-8") == ""
    assert "Nenhuma" in (out_dir / "report.md").read_text(encoding="utf-8")


@pytest.mark.req("REQ-CAL-003")
def test_article_text_stays_out_of_the_fixture(processed):
    session_dir, fixtures_dir, calibration_root, _ = processed
    fixture_text = (fixtures_dir / f"{session_dir.name}.json").read_text(encoding="utf-8")
    assert "surgiu na China" not in fixture_text and "Guarda-chuva" not in fixture_text
    sha = json.loads(fixture_text)["article_ref"]["sha256"]
    article = json.loads(
        (calibration_root / "cal-test" / "articles" / f"{sha}.json").read_text(encoding="utf-8")
    )
    assert article["title"] == server.ARTICLE["title"]
    assert article["text"] == server.ARTICLE["text"]


@pytest.mark.req("REQ-CAL-001")
def test_reprocessing_is_byte_identical(processed):
    session_dir, fixtures_dir, calibration_root, args = processed
    out_dir = calibration_root / "cal-test" / session_dir.name
    paths = [
        fixtures_dir / f"{session_dir.name}.json",
        out_dir / "divergences.jsonl",
        out_dir / "report.md",
    ]
    before = [path.read_bytes() for path in paths]
    assert calibration_main(args) == 0
    assert [path.read_bytes() for path in paths] == before
