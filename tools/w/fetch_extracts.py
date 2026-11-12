"""Snapshot the plain-text extract of every article of the evaluation list (set W).

Reads the title list (TSV, first column), asks the public TextExtracts API of pt.wikipedia for
each title (one title per request, serial, with a pause) and stores the raw answer under
``ARTIGO_DATA_ROOT/raw/wikipedia_extracts/<snapshot>/``. The run is resumable: titles already in
``index.jsonl`` are skipped. Nothing is written inside the repository.

Usage:
    uv run --env-file .env python -m tools.w.fetch_extracts \
        --titles <list.tsv> --snapshot 2026-11-12
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterable, Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from artigo_env.core.paths import data_path

API_URL = "https://pt.wikipedia.org/w/api.php"
USER_AGENT = "wiki-extract-snapshot/0.1 (academic coursework; serial low-rate requests)"
MAX_ATTEMPTS = 6

Fetch = Callable[[str], dict[str, Any]]


def read_titles(path: Path) -> list[str]:
    """Titles from the first TSV column, skipping ``#`` comments and the header row."""
    titles: list[str] = []
    seen: set[str] = set()
    header_skipped = False
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        if not header_skipped:
            header_skipped = True
            continue
        title = line.split("\t")[0].strip()
        if title and title not in seen:
            seen.add(title)
            titles.append(title)
    return titles


def file_name(title: str) -> str:
    return hashlib.sha1(title.encode("utf-8")).hexdigest()[:16] + ".json"


def api_url(title: str) -> str:
    query = {
        "action": "query",
        "prop": "extracts|info",
        "explaintext": 1,
        "exsectionformat": "wiki",
        "redirects": 1,
        "titles": title,
        "format": "json",
        "formatversion": 2,
        "maxlag": 5,
    }
    return API_URL + "?" + urllib.parse.urlencode(query)


def http_fetch(title: str) -> dict[str, Any]:
    """One API call with polite retries (HTTP 429/503 and the ``maxlag`` error)."""
    request = urllib.request.Request(api_url(title), headers={"User-Agent": USER_AGENT})
    wait = 5.0
    for _ in range(MAX_ATTEMPTS):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                data: dict[str, Any] = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            if error.code not in (429, 500, 502, 503, 504):
                raise
            retry_after = error.headers.get("Retry-After") if error.headers else None
            time.sleep(float(retry_after) if retry_after and retry_after.isdigit() else wait)
            wait *= 2
            continue
        except (urllib.error.URLError, TimeoutError):
            time.sleep(wait)
            wait *= 2
            continue
        if data.get("error", {}).get("code") == "maxlag":
            time.sleep(wait)
            wait *= 2
            continue
        return data
    raise RuntimeError(f"gave up after {MAX_ATTEMPTS} attempts: {title!r}")


def summarize(title: str, data: dict[str, Any]) -> dict[str, Any]:
    """Index row for one answer."""
    query = data.get("query", {})
    pages = query.get("pages") or [{}]
    page = pages[0]
    extract = page.get("extract") or ""
    if page.get("missing") or page.get("invalid") or "pageid" not in page:
        status = "missing"
    elif not extract.strip():
        status = "empty"
    else:
        status = "ok"
    return {
        "requested": title,
        "resolved": page.get("title"),
        "redirected": bool(query.get("redirects")),
        "pageid": page.get("pageid"),
        "lastrevid": page.get("lastrevid"),
        "touched": page.get("touched"),
        "n_chars": len(extract),
        "status": status,
        "file": file_name(title),
    }


def load_index(index_path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    if index_path.is_file():
        for line in index_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                rows[row["requested"]] = row
    return rows


def snapshot(
    titles: Iterable[str],
    out_dir: Path,
    fetch: Fetch = http_fetch,
    interval: float = 1.5,
    sleep: Callable[[float], None] = time.sleep,
    now: Callable[[], str] = lambda: datetime.now(UTC).isoformat(timespec="seconds"),
) -> Iterator[dict[str, Any]]:
    """Fetch every title not yet in the index; yields one index row per new title."""
    pages_dir = out_dir / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)
    index_path = out_dir / "index.jsonl"
    done = load_index(index_path)
    first = True
    with index_path.open("a", encoding="utf-8", newline="\n") as index:
        for title in titles:
            if title in done:
                continue
            if not first:
                sleep(interval)
            first = False
            data = fetch(title)
            row = summarize(title, data)
            row["fetched_at"] = now()
            payload = {"requested": title, "fetched_at": row["fetched_at"], "response": data}
            target = pages_dir / row["file"]
            target.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            index.write(json.dumps(row, ensure_ascii=False) + "\n")
            index.flush()
            done[title] = row
            yield row


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--titles", required=True, type=Path, help="TSV with titles in column 1")
    parser.add_argument("--snapshot", required=True, help="snapshot label, e.g. 2026-11-12")
    parser.add_argument("--interval", type=float, default=1.5, help="seconds between requests")
    parser.add_argument("--limit", type=int, default=None, help="stop after this many new titles")
    args = parser.parse_args(argv)

    out_dir = data_path("raw", "wikipedia_extracts", args.snapshot)
    titles = read_titles(args.titles)
    counts: dict[str, int] = {}
    for n, row in enumerate(snapshot(titles, out_dir, interval=args.interval), start=1):
        counts[row["status"]] = counts.get(row["status"], 0) + 1
        if n % 100 == 0:
            print(f"{n} new titles fetched: {counts}", flush=True)
        if args.limit is not None and n >= args.limit:
            break
    total = len(load_index(out_dir / "index.jsonl"))
    print(f"done: {total}/{len(titles)} titles in the index; new this run: {counts}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
