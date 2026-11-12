import ast
import json
import re
from datetime import datetime
from pathlib import Path

import jsonschema
import pytest

from tools.recorder.events import EVENT_TYPES, RedactedEvent, make_event
from tools.recorder.redact import REDACTED
from tools.recorder.writer import MODES, SessionWriter, read_events, read_meta

pytestmark = pytest.mark.fast

JWT = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0In0.c2lnbmF0dXJlLXZhbHVl"
SCHEMA_PATH = Path("tools/recorder/schema/recorder-events.v1.json")


@pytest.fixture
def schema(repo_root):
    return json.loads((repo_root / SCHEMA_PATH).read_text(encoding="utf-8"))


@pytest.fixture
def writer(tmp_path):
    session = SessionWriter(tmp_path, "random", now=lambda: datetime(2026, 11, 12, 21, 3, 1))
    yield session
    session.close(0)


SAMPLES = {
    "session.start": {"url": "https://artigo.app/aleatorio"},
    "session.end": {"reason": "window-closed"},
    "net.request": {"method": "GET", "url": "https://artigo.app/x", "headers": {}},
    "net.response": {"url": "https://artigo.app/x", "status": 200, "headers": {}},
    "input.click": {"target": "#guess", "text": "Enviar"},
    "input.key": {"code": "KeyA", "target": "#guess"},
    "input.submit": {"value": "anos", "target": "form"},
    "dom.mutation": {"kind": "attributes", "target": "span:nth-of-type(3)"},
    "dom.quiescent": {"after_ms": 300},
    "snapshot": {"file": "snapshots/9.html", "reason": "load"},
    "console": {"level": "log", "text": "ready"},
    "storage": {"items": {"history": "[]"}},
    "screen": {"file": "screens/12.png", "reason": "guess"},
    "third_party.count": {"hosts": {"ads.example.com": 4}},
}


@pytest.mark.req("REQ-REC-002")
def test_writer_rejects_unredacted_events(writer):
    with pytest.raises(TypeError):
        writer.emit({"type": "console", "payload": {}})  # type: ignore[arg-type]


@pytest.mark.req("REQ-REC-002")
def test_make_event_redacts_payload():
    event = make_event(
        "net.response",
        {
            "url": "https://artigo.app/a?id=3&session_id=abc",
            "status": 200,
            "headers": {"Set-Cookie": "sid=1", "Content-Type": "text/html"},
            "note": f"Bearer abc123.def and {JWT}",
        },
    )
    payload = event.payload
    assert payload["url"] == f"https://artigo.app/a?id=3&session_id={REDACTED}"
    assert payload["headers"] == {"Set-Cookie": REDACTED, "Content-Type": "text/html"}
    assert "abc123.def" not in payload["note"] and JWT not in payload["note"]
    assert payload["status"] == 200


@pytest.mark.req("REQ-REC-002")
def test_storage_items_redact_sensitive_keys_only_there():
    storage = make_event("storage", {"items": {"authToken": "secret-value", "history": "[]"}})
    assert storage.payload["items"] == {"authToken": REDACTED, "history": "[]"}
    key_event = make_event("input.key", {"code": "KeyA", "key": "a"})
    assert key_event.payload == {"code": "KeyA", "key": "a"}


@pytest.mark.req("REQ-REC-002")
def test_unknown_event_type_rejected():
    with pytest.raises(ValueError):
        make_event("made.up", {})


@pytest.mark.req("REQ-REC-002")
def test_redacted_event_is_only_built_in_events_module(repo_root):
    offenders = []
    for path in (repo_root / "tools").rglob("*.py"):
        if path.name == "events.py":
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "RedactedEvent"
            ):
                offenders.append(f"{path.name}:{node.lineno}")
    assert offenders == []


@pytest.mark.req("REQ-REC-006")
def test_folder_layout_and_name(writer):
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}_\d{6}_random", writer.path.name)
    assert writer.path.name == "2026-11-12_210301_random"
    assert sorted(p.name for p in writer.path.iterdir()) == [
        "bodies",
        "events.jsonl",
        "meta.json",
        "screens",
        "snapshots",
    ]


@pytest.mark.req("REQ-REC-006")
def test_invalid_mode_rejected(tmp_path):
    assert MODES == ("daily", "random", "history", "free")
    with pytest.raises(ValueError):
        SessionWriter(tmp_path, "bogus")


@pytest.mark.req("REQ-REC-004")
def test_every_event_type_validates_against_schema(writer, schema):
    assert set(SAMPLES) == set(EVENT_TYPES)
    for event_type in EVENT_TYPES:
        writer.emit(make_event(event_type, SAMPLES[event_type]))
    events = read_events(writer.path)
    assert [event["seq"] for event in events] == list(range(len(EVENT_TYPES)))
    assert [event["type"] for event in events] == list(EVENT_TYPES)
    validator = jsonschema.Draft202012Validator(schema)
    for event in events:
        validator.validate(event)
    times = [event["t_mono_ms"] for event in events]
    assert times == sorted(times)


@pytest.mark.req("REQ-REC-004")
def test_schema_rejects_malformed_events(schema):
    validator = jsonschema.Draft202012Validator(schema)
    base = {"seq": 0, "t_mono_ms": 1.0, "t_wall": "2026-11-12T21:03:01.000-03:00"}
    bad = [
        {**base, "type": "net.request", "payload": {"url": "x"}},
        {**base, "type": "unknown", "payload": {}},
        {**base, "type": "console", "payload": {"level": "log", "text": "x"}, "extra": 1},
        {"seq": -1, "t_mono_ms": 1.0, "t_wall": base["t_wall"], "type": "console", "payload": {}},
    ]
    for event in bad:
        assert not validator.is_valid(event)


@pytest.mark.req("REQ-REC-006")
def test_interrupted_session_is_readable_and_incomplete(tmp_path):
    session = SessionWriter(tmp_path, "daily")
    session.emit(make_event("console", {"level": "log", "text": "one"}))
    session.emit(make_event("console", {"level": "log", "text": "two"}))
    # No close(): simulate a killed process, then a partially written last line.
    with (session.path / "events.jsonl").open("a", encoding="utf-8") as handle:
        handle.write('{"seq": 2, "t_mono_ms": 3.0, "type": "cons')
    meta = read_meta(session.path)
    assert meta["complete"] is False and meta["ended_at"] is None
    assert [e["payload"]["text"] for e in read_events(session.path)] == ["one", "two"]
    session.close(2)


@pytest.mark.req("REQ-REC-006")
def test_corruption_in_the_middle_is_not_swallowed(writer):
    writer.emit(make_event("console", {"level": "log", "text": "one"}))
    with (writer.path / "events.jsonl").open("a", encoding="utf-8") as handle:
        handle.write("garbage\n")
    writer.emit(make_event("console", {"level": "log", "text": "three"}))
    with pytest.raises(ValueError):
        read_events(writer.path)


@pytest.mark.req("REQ-REC-006")
def test_close_marks_complete_and_leaves_no_temporary_files(tmp_path):
    session = SessionWriter(tmp_path, "free")
    session.write_meta(bundle_hash="be54b82b", browser_version="chromium 1")
    session.save_screen(1, b"\x89PNG")
    session.close(7)
    meta = read_meta(session.path)
    assert meta["complete"] is True and meta["n_guesses"] == 7
    assert meta["bundle_hash"] == "be54b82b" and meta["schema_version"] == 1
    assert meta["ended_at"] is not None
    assert not list(session.path.rglob("*.tmp"))
    with pytest.raises(ValueError):
        session.write_meta(not_a_field=1)


@pytest.mark.req("REQ-REC-002")
def test_bodies_and_snapshots_are_redacted_on_disk(writer):
    body = json.dumps({"title": "x", "debug": f"Authorization: Bearer abc123.def {JWT}"})
    name = writer.save_body(3, body.encode("utf-8"), "json")
    snapshot = writer.save_snapshot(4, f'<div data-t="{JWT}">ok</div>')
    binary = writer.save_body(5, b"\x00\x01", "bin", textual=False)
    assert (name, snapshot, binary) == ("bodies/3.json", "snapshots/4.html", "bodies/5.bin")
    stored = (writer.path / name).read_text(encoding="utf-8")
    assert "abc123.def" not in stored and JWT not in stored
    assert json.loads(stored)["title"] == "x"
    assert JWT not in (writer.path / snapshot).read_text(encoding="utf-8")
    assert (writer.path / binary).read_bytes() == b"\x00\x01"


@pytest.mark.req("REQ-REC-002")
def test_redacted_event_type_is_exported():
    assert isinstance(make_event("console", {"level": "log", "text": "x"}), RedactedEvent)
