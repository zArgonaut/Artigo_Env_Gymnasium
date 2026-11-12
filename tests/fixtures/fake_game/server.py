"""Local fake of the game site used by the recorder tests (never contacts the real one).

Serves the fake page on one port and acts as a "third-party host" on a second port. The article
endpoint carries planted secrets (cookie, JWT) that the recorder must never write to disk.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Callable
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).parent
PLANTED_COOKIE = "PLANTED-COOKIE-123"
PLANTED_JWT = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJwbGFudGVkIn0.cGxhbnRlZC1zaWduYXR1cmU"
PLANTED_STORAGE_SECRET = "PLANTED-STORAGE-SECRET-456"
ARTICLE = {
    "title": "Guarda-chuva",
    "text": (
        "O guarda-chuva é um objeto de uso comum e antigo.\n"
        "== História ==\n"
        "O guarda-chuva surgiu na China e o uso se espalhou pelo mundo."
    ),
    "debug": PLANTED_JWT,
}

_STATIC = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/index.html": ("index.html", "text/html; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
    "/asset-manifest.json": ("asset-manifest.json", "application/json"),
    "/static/js/main.0a1b2c3d.js": ("static/js/main.0a1b2c3d.js", "text/javascript"),
}


@dataclass(frozen=True)
class FakeGame:
    base_url: str
    third_party_url: str
    stop: Callable[[], None]


def _serve(handler: type[BaseHTTPRequestHandler], port: int) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer(("127.0.0.1", port), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def _url(server: ThreadingHTTPServer) -> str:
    return f"http://127.0.0.1:{server.server_address[1]}"


class _ThirdPartyHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        body = b"{}"
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        pass


def start_with_third_party(port: int = 0) -> FakeGame:
    """Start the fake game and its third-party host on 127.0.0.1."""
    third = _serve(_ThirdPartyHandler, 0)
    third_url = _url(third)

    class Handler(BaseHTTPRequestHandler):
        def _send(
            self, status: int, body: bytes, content_type: str, extra: dict[str, str] | None = None
        ) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            for name, value in (extra or {}).items():
                self.send_header(name, value)
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            path = self.path.split("?", 1)[0]
            if path in _STATIC:
                name, content_type = _STATIC[path]
                self._send(200, (ROOT / name).read_bytes(), content_type)
            elif path == "/articles/fake.json":
                body = json.dumps(ARTICLE, ensure_ascii=False).encode("utf-8")
                self._send(
                    200,
                    body,
                    "application/json; charset=utf-8",
                    {"Set-Cookie": f"sid={PLANTED_COOKIE}; Path=/"},
                )
            elif path == "/third-party-url":
                # text/plain on purpose: the recorder does not save such bodies, so the
                # third-party address never reaches the session folder.
                self._send(200, third_url.encode("utf-8"), "text/plain")
            else:
                self._send(404, b"not found", "text/plain")

        def log_message(self, format: str, *args: object) -> None:
            pass

    main = _serve(Handler, port)

    def stop() -> None:
        for server in (main, third):
            server.shutdown()
            server.server_close()

    return FakeGame(_url(main), third_url, stop)


def start(port: int = 0) -> tuple[str, Callable[[], None]]:
    """Start the fake game (and a third-party host); return its base URL and a stop callable."""
    game = start_with_third_party(port)
    return game.base_url, game.stop
