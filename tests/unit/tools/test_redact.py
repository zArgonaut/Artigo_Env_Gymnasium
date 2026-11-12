import json

import pytest

from tools.recorder.redact import (
    REDACTED,
    redact_headers,
    redact_mapping,
    redact_text,
    redact_url,
)

pytestmark = [pytest.mark.fast, pytest.mark.req("REQ-REC-002", "REQ-DOC-003")]

JWT = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0In0.c2lnbmF0dXJlLXZhbHVl"
SK = "sk-ant-api03-AbCdEfGhIjKlMnOpQrStUvWx"
COMMIT = "3f2a9c41d7e8b6a5f0c1d2e3f4a5b6c7d8e9f0a1"
SHA256 = "9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08"


def test_headers_named_secrets():
    out = redact_headers(
        {
            "Authorization": "Bearer abc",
            "COOKIE": "sid=1",
            "Set-Cookie": "a=b",
            "X-Csrf-Token": "zzz",
            "x-api-key": "k",
            "Proxy-Authorization": "Basic x",
            "Content-Type": "application/json",
        }
    )
    assert out["Content-Type"] == "application/json"
    assert {v for k, v in out.items() if k != "Content-Type"} == {REDACTED}
    assert list(out) == [
        "Authorization",
        "COOKIE",
        "Set-Cookie",
        "X-Csrf-Token",
        "x-api-key",
        "Proxy-Authorization",
        "Content-Type",
    ]


def test_header_long_opaque_value():
    assert redact_headers({"X-Trace": "a" * 40}) == {"X-Trace": REDACTED}
    assert redact_headers({"ETag": "abc123"}) == {"ETag": "abc123"}


def test_url_params():
    out = redact_url("https://artigo.app/a?id=3&session_id=abc&TOKEN=x&flag#frag")
    assert out == f"https://artigo.app/a?id=3&session_id={REDACTED}&TOKEN={REDACTED}&flag#frag"
    assert redact_url("https://artigo.app/a") == "https://artigo.app/a"


def test_mapping_recursive_keeps_keys():
    out = redact_mapping({"user": {"authToken": "x", "name": "n"}, "l": [{"jwt": "y"}, "plain"]})
    assert out == {"user": {"authToken": REDACTED, "name": "n"}, "l": [{"jwt": REDACTED}, "plain"]}


def test_text_bearer_jwt_sk():
    text = f"curl -H 'Authorization: Bearer abc123.def' and {JWT} and key {SK} ghp_{'a' * 36}"
    out = redact_text(text)
    for secret in ("abc123.def", JWT, SK, "ghp_" + "a" * 36):
        assert secret not in out
    assert out.startswith("curl -H 'Authorization: ")


@pytest.mark.parametrize(
    "text, secret",
    [
        ('password="hunter2"', "hunter2"),
        ("API_KEY=abcd1234", "abcd1234"),
        ("secret: s3cr3t", "s3cr3t"),
        ('{"token": "0123456789abcdefXYZ"}', "0123456789abcdefXYZ"),
    ],
)
def test_text_assignments(text, secret):
    out = redact_text(text)
    assert secret not in out
    assert REDACTED in out


def test_text_keeps_project_vocabulary():
    text = "Token: sequência máxima de letras; n_tokens=4096; max_tokens: 200000"
    assert redact_text(text) == text


def test_text_keeps_commit_and_sha256():
    text = f"commit {COMMIT} sha256 {SHA256}"
    assert redact_text(text) == text


def test_text_secret_inside_json_line():
    line = json.dumps({"content": 'curl -H "Authorization: Bearer abc.def" x', "n": 1})
    out = redact_text(line)
    assert "abc.def" not in out
    assert json.loads(out)["n"] == 1


def test_extra_patterns():
    assert redact_text("id ACME-123456 ok", extra_patterns=[r"ACME-\d{6}"]) == f"id {REDACTED} ok"


def test_idempotent():
    text = f'Bearer abc.def {JWT} password="hunter2" {SK}'
    once = redact_text(text)
    assert redact_text(once) == once
