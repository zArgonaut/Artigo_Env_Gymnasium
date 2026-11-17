"""End-to-end recorder tests against the local fake game page (never the real site)."""

from __future__ import annotations

import asyncio
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import urlsplit

import jsonschema
import pytest

from tests.fixtures.fake_game import server
from tools.recorder.session import RecorderConfig, run_session
from tools.recorder.writer import read_events, read_meta

pytestmark = pytest.mark.slow

SCHEMA_PATH = Path("tools/recorder/schema/recorder-events.v1.json")
GUESSES = ("chuva", "de", "chuva")
LAYERS = {
    "net": ("net.request", "net.response"),
    "input": ("input.key", "input.submit"),
    "dom mutation": ("dom.mutation", "dom.quiescent"),
    "snapshot": ("snapshot",),
    "console": ("console",),
    "storage": ("storage",),
    "screen": ("screen",),
}


@pytest.fixture(scope="module")
def chromium_available() -> None:
    sync_api = pytest.importorskip("playwright.sync_api", reason="playwright is not installed")
    try:
        with sync_api.sync_playwright() as playwright:
            if not Path(playwright.chromium.executable_path).exists():
                pytest.skip("Playwright Chromium is not installed (playwright install chromium)")
            playwright.chromium.launch(headless=True).close()
    except Exception as error:
        pytest.skip(f"Playwright Chromium cannot be launched: {error}")


@pytest.fixture(scope="module")
def fake_game(chromium_available):
    game = server.start_with_third_party()
    yield game
    game.stop()


def _committed(out_root: Path, guesses: int) -> bool:
    """True once the last guess has been fully captured (its screen event exists)."""
    sessions = list(out_root.iterdir())
    if not sessions:
        return False
    try:
        events = read_events(sessions[0])
    except (OSError, ValueError):
        return False
    screens = [e for e in events if e["type"] == "screen" and e["payload"]["reason"] == "guess"]
    return len(screens) >= guesses


@pytest.fixture(scope="module")
def session_dir(fake_game, tmp_path_factory) -> Path:
    base = tmp_path_factory.mktemp("recording")
    out_root = base / "recordings"
    config = RecorderConfig(
        mode="free",
        start_url=fake_game.base_url + "/",
        out_root=out_root,
        profile_dir=base / "profile",
        allowed_hosts=(urlsplit(fake_game.base_url).netloc,),
        quiescence_ms=150,
        headless=True,
    )

    async def play(page) -> None:
        # The test, not the recorder, drives the page.
        for number, guess in enumerate(GUESSES, start=1):
            await page.fill("#guess-input", guess)
            if number == len(GUESSES):  # the last guess goes through the send button
                await page.click("button[title='Enviar']")
            else:
                await page.press("#guess-input", "Enter")
            deadline = time.monotonic() + 15
            while not _committed(out_root, number) and time.monotonic() < deadline:
                await asyncio.sleep(0.1)

    return run_session(config, on_ready=play)


@pytest.fixture(scope="module")
def events(session_dir) -> list[dict]:
    return read_events(session_dir)


def _types(events: list[dict]) -> list[str]:
    return [event["type"] for event in events]


@pytest.mark.req("REQ-REC-004")
@pytest.mark.parametrize("layer", list(LAYERS))
def test_layer_is_captured(events, layer):
    present = set(_types(events))
    assert present & set(LAYERS[layer]), f"layer {layer!r} missing; got {sorted(present)}"


@pytest.mark.req("REQ-REC-004")
def test_all_event_types_of_the_contract_are_emitted(events):
    expected = {
        "session.start",
        "session.end",
        "net.request",
        "net.response",
        "input.key",
        "input.submit",
        "dom.mutation",
        "dom.quiescent",
        "snapshot",
        "console",
        "storage",
        "screen",
        "third_party.count",
    }
    assert expected <= set(_types(events))


@pytest.mark.req("REQ-REC-004")
def test_order_after_a_guess_follows_the_contract(events):
    submits = [e for e in events if e["type"] == "input.submit"]
    assert [e["payload"]["value"] for e in submits] == list(GUESSES)
    for submit in submits:
        later = [e for e in events if e["seq"] > submit["seq"]]
        quiet = next(e for e in later if e["type"] == "dom.quiescent")
        after = [e for e in later if e["seq"] > quiet["seq"]]
        order = [e["type"] for e in after if e["type"] in ("snapshot", "storage", "screen")][:3]
        assert order == ["snapshot", "storage", "screen"]
        snapshot = next(e for e in after if e["type"] == "snapshot")
        assert snapshot["payload"]["reason"] == "guess"
    first_snapshot = next(e for e in events if e["type"] == "snapshot")
    assert first_snapshot["payload"]["reason"] == "load"
    assert first_snapshot["seq"] < submits[0]["seq"]
    assert events[-1]["type"] == "session.end"


@pytest.mark.req("REQ-REC-004")
def test_snapshots_hold_the_game_state(session_dir, events):
    snapshots = [e for e in events if e["type"] == "snapshot"]
    load = (session_dir / snapshots[0]["payload"]["file"]).read_text(encoding="utf-8")
    assert 'id="root"' in load and 'id="token-text-0"' in load
    assert 'data-token-length="5"' in load
    last_guess = [e for e in snapshots if e["payload"]["reason"] == "guess"][-1]
    html = (session_dir / last_guess["payload"]["file"]).read_text(encoding="utf-8")
    assert ">chuva<" in html
    counts = [e for e in events if e["type"] == "dom.mutation"]
    assert any(e["payload"].get("target") == "#guess-list" for e in counts)


@pytest.mark.req("REQ-REC-004")
def test_every_event_validates_against_the_schema(repo_root, events):
    schema = json.loads((repo_root / SCHEMA_PATH).read_text(encoding="utf-8"))
    validator = jsonschema.Draft202012Validator(schema)
    errors = [str(error.message) for e in events for error in validator.iter_errors(e)]
    assert errors == []
    assert [e["seq"] for e in events] == list(range(len(events)))


@pytest.mark.req("REQ-REC-002")
@pytest.mark.parametrize(
    "secret",
    [server.PLANTED_COOKIE, server.PLANTED_JWT, server.PLANTED_STORAGE_SECRET],
)
def test_planted_secret_is_in_no_file(session_dir, secret):
    needle = secret.encode("utf-8")
    leaks = [
        str(path.relative_to(session_dir))
        for path in session_dir.rglob("*")
        if path.is_file() and needle in path.read_bytes()
    ]
    assert leaks == []


@pytest.mark.req("REQ-REC-002")
def test_secrets_were_present_and_redacted(session_dir, events):
    storage = [e for e in events if e["type"] == "storage"][-1]["payload"]["items"]
    assert storage["authToken"] == "[REDACTED]"
    assert "history" in storage
    article = next(
        e for e in events if e["type"] == "net.response" and "/articles/" in e["payload"]["url"]
    )
    body = (session_dir / article["payload"]["body_file"]).read_text(encoding="utf-8")
    assert "[REDACTED]" in body
    assert article["payload"]["headers"].get("set-cookie", "[REDACTED]") == "[REDACTED]"


@pytest.mark.req("REQ-REC-003")
def test_third_party_host_only_in_the_aggregate_event(session_dir, events, fake_game):
    host = urlsplit(fake_game.third_party_url).netloc
    pattern = re.compile(re.escape(host) + r"(?!\d)")
    for path in session_dir.rglob("*"):
        if path.is_file() and path.name != "events.jsonl":
            assert not pattern.search(path.read_bytes().decode("latin-1")), path.name
    for event in events:
        text = json.dumps(event)
        if event["type"] == "third_party.count":
            assert event["payload"]["hosts"] == {host: 1}
        else:
            assert not pattern.search(text), event["type"]


@pytest.mark.req("REQ-REC-005")
def test_meta_records_bundle_hash_and_completion(session_dir):
    meta = read_meta(session_dir)
    assert meta["bundle_hash"] == "0a1b2c3d"
    assert meta["complete"] is True
    assert meta["n_guesses"] == len(GUESSES)
    assert meta["playwright_version"]
    assert meta["browser_version"]


@pytest.mark.req("REQ-REC-006")
def test_killed_recorder_leaves_a_readable_incomplete_session(fake_game, tmp_path):
    out_root = tmp_path / "recordings"
    command = [
        sys.executable,
        "-m",
        "tools.recorder",
        "record",
        "--mode",
        "free",
        "--url",
        fake_game.base_url + "/",
        "--headless",
        "--no-screens",
        "--allow-host",
        urlsplit(fake_game.base_url).netloc,
        "--out-root",
        str(out_root),
        "--profile-dir",
        str(tmp_path / "profile"),
    ]
    repo = Path(__file__).resolve().parents[2]
    process = subprocess.Popen(command, cwd=repo, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    try:
        deadline = time.monotonic() + 60
        seen = False
        while time.monotonic() < deadline and not seen:
            time.sleep(0.2)
            sessions = list(out_root.iterdir()) if out_root.exists() else []
            if sessions and (sessions[0] / "events.jsonl").exists():
                seen = any(e["type"] == "snapshot" for e in read_events(sessions[0]))
            assert process.poll() is None or seen, "recorder exited early"
        assert seen, "no snapshot within 60 s"
    finally:
        process.kill()
        process.wait(timeout=30)
    session = next(out_root.iterdir())
    assert read_meta(session)["complete"] is False
    assert any(e["type"] == "snapshot" for e in read_events(session))
    assert not list(session.rglob("*.tmp"))
