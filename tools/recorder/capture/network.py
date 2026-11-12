"""Network capture: detailed only for allowed hosts, aggregate counts for everyone else."""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Sequence
from typing import Any
from urllib.parse import urlsplit

from tools.recorder.events import make_event
from tools.recorder.writer import SessionWriter

_BUNDLE = re.compile(r"main\.([0-9a-f]{8})\.js")
_BODY_KINDS = (
    ("json", "json"),
    ("javascript", "js"),
    ("ecmascript", "js"),
    ("css", "css"),
    ("html", "html"),
    ("manifest", "json"),
)


def is_allowed(url: str, allowed_hosts: Sequence[str]) -> bool:
    """True when the URL host equals an allowed host or is a subdomain of it.

    An entry without a port matches any port; an entry with a port (``"127.0.0.1:8123"``, used
    by the tests) also requires that port.
    """
    parts = urlsplit(url)
    host = (parts.hostname or "").lower()
    if not host:
        return False
    try:
        port = parts.port
    except ValueError:
        return False
    for entry in allowed_hosts:
        name, _, entry_port = entry.lower().partition(":")
        if entry_port and port != int(entry_port):
            continue
        if host == name or host.endswith("." + name):
            return True
    return False


def host_key(url: str) -> str:
    """Host (with port when explicit) used to aggregate third-party requests."""
    parts = urlsplit(url)
    host = (parts.hostname or "unknown").lower()
    try:
        port = parts.port
    except ValueError:
        port = None
    return f"{host}:{port}" if port else host


def bundle_hash_from_url(url: str) -> str | None:
    """Extract the 8-hex-digit bundle hash from a ``main.<hash>.js`` URL."""
    match = _BUNDLE.search(url)
    return match.group(1) if match else None


def body_extension(content_type: str, url: str) -> str | None:
    """File extension for response bodies worth keeping, or None to skip the body."""
    lowered = content_type.lower()
    for marker, extension in _BODY_KINDS:
        if marker in lowered:
            return extension
    if urlsplit(url).path.endswith(".webmanifest"):
        return "json"
    return None


def hash_from_manifest(body: bytes) -> str | None:
    """Look for a ``main.<hash>.js`` reference inside an asset manifest body."""
    try:
        data: Any = json.loads(body)
    except ValueError:
        return None
    return bundle_hash_from_url(json.dumps(data))


class NetworkCapture:
    """Turns Playwright request/response objects into events (read-only on the objects)."""

    def __init__(self, writer: SessionWriter, allowed_hosts: Sequence[str]) -> None:
        self._writer = writer
        self._allowed = tuple(allowed_hosts)
        self.third_party: Counter[str] = Counter()
        self._hash_from_url: str | None = None
        self._hash_from_manifest: str | None = None

    def _note_hash(self, value: str | None, *, manifest: bool) -> None:
        if value is None:
            return
        if manifest:
            self._hash_from_manifest = self._hash_from_manifest or value
        else:
            self._hash_from_url = self._hash_from_url or value
        # The URL wins; the manifest only fills in when no URL carried the hash.
        chosen = self._hash_from_url or self._hash_from_manifest
        self._writer.write_meta(bundle_hash=chosen)

    def on_request(self, request: Any) -> None:
        url = request.url
        if not is_allowed(url, self._allowed):
            self.third_party[host_key(url)] += 1
            return
        self._note_hash(bundle_hash_from_url(url), manifest=False)
        self._writer.emit(
            make_event(
                "net.request",
                {
                    "method": request.method,
                    "url": url,
                    "headers": dict(request.headers),
                    "resource_type": request.resource_type,
                },
            )
        )

    async def on_response(self, response: Any) -> None:
        url = response.url
        if not is_allowed(url, self._allowed):
            return
        self._note_hash(bundle_hash_from_url(url), manifest=False)
        try:
            headers = dict(await response.all_headers())
        except Exception:
            headers = dict(response.headers)
        payload: dict[str, Any] = {"url": url, "status": response.status, "headers": headers}
        extension = body_extension(headers.get("content-type", ""), url)
        if extension is not None:
            try:
                body = await response.body()
            except Exception:
                body = None  # redirects, aborted requests and the like have no body
            if body is not None:
                # No await between reading next_seq and emitting: the name matches the seq.
                payload["body_file"] = self._writer.save_body(
                    self._writer.next_seq, body, extension
                )
                if url.endswith(".json") and "manifest" in url:
                    self._note_hash(hash_from_manifest(body), manifest=True)
        self._writer.emit(make_event("net.response", payload))

    def third_party_counts(self) -> dict[str, int]:
        return dict(sorted(self.third_party.items()))
