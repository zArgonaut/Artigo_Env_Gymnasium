"""In-memory removal of sensitive data (SDD 03 §5).

Shared by the recorder, the session-archiving hook and the external documentation tool: nothing
is written to disk before passing through these functions. Redacted values become
``"[REDACTED]"``; key and header names are kept so the structure stays analysable.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

REDACTED = "[REDACTED]"

_SENSITIVE_NAME = re.compile(r"(?i)(token|auth|session|secret|key|bearer|jwt|sid)")
_SENSITIVE_HEADERS = frozenset(
    {"authorization", "cookie", "set-cookie", "proxy-authorization", "x-api-key"}
)
_TOKEN_HEADER = re.compile(r"(?i)^x-.*-token$")
_LONG_OPAQUE = re.compile(r"[A-Za-z0-9+/=_-]{32,}")

# Values stop at quotes and backslashes so that redacting inside a JSON string keeps it valid.
_VALUE = r"[^\s\"'\\,}]"
_CREDENTIALS = (
    re.compile(rf"(?i)\bBearer\s+{_VALUE}+"),
    re.compile(r"\beyJ[\w-]+\.[\w-]+\.[\w-]+"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{20,}"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36,}"),
)
_SEPARATOR = r"\\?[\"']?\s*[=:]\s*\\?[\"']?"
_ASSIGNMENTS = (
    re.compile(rf"(?i)(\b(?:api[_-]?key|secret|password|passwd)\b{_SEPARATOR})({_VALUE}+)"),
    # A bare "token" is also a common word in this project, so require an opaque-looking value.
    re.compile(rf"(?i)(\btoken\b{_SEPARATOR})([A-Za-z0-9_.\-+/=]{{16,}})"),
)


def redact_text(text: str, extra_patterns: Sequence[str] = ()) -> str:
    """Remove credentials from free text. Long hex strings (hashes) are left untouched."""
    for pattern in _CREDENTIALS:
        text = pattern.sub(REDACTED, text)
    for pattern in _ASSIGNMENTS:
        text = pattern.sub(lambda match: match.group(1) + REDACTED, text)
    for extra in extra_patterns:
        text = re.sub(extra, REDACTED, text)
    return text


def redact_headers(headers: Mapping[str, str]) -> dict[str, str]:
    """Redact HTTP headers by name and by credential-looking value."""
    out: dict[str, str] = {}
    for name, value in headers.items():
        if name.lower() in _SENSITIVE_HEADERS or _TOKEN_HEADER.match(name):
            out[name] = REDACTED
        else:
            out[name] = _LONG_OPAQUE.sub(REDACTED, redact_text(value))
    return out


def redact_url(url: str) -> str:
    """Redact query parameters whose name looks sensitive; everything else is preserved."""
    head, question, rest = url.partition("?")
    if not question:
        return url
    query, hash_mark, fragment = rest.partition("#")
    pairs = []
    for pair in query.split("&"):
        name, equals, _value = pair.partition("=")
        pairs.append(f"{name}={REDACTED}" if equals and _SENSITIVE_NAME.search(name) else pair)
    return head + question + "&".join(pairs) + hash_mark + fragment


def _redact_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return redact_mapping(value)
    if isinstance(value, list | tuple):
        return [_redact_value(item) for item in value]
    if isinstance(value, str):
        return redact_text(value)
    return value


def redact_mapping(data: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively redact a JSON-like mapping (localStorage, payloads)."""
    out: dict[str, Any] = {}
    for key, value in data.items():
        if _SENSITIVE_NAME.search(str(key)):
            out[str(key)] = REDACTED
        else:
            out[str(key)] = _redact_value(value)
    return out
