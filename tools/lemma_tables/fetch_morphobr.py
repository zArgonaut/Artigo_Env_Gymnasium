"""Download a pinned commit of MorphoBr (Apache-2.0) into ``ARTIGO_DATA_ROOT/raw/morphobr``.

Only the word-class dictionaries used by ``build_open`` are fetched (verbs, nouns, adjectives,
adverbs) plus the licence and documentation files. The clitic dictionaries are skipped: their
forms contain a hyphen, which the game's tokenizer always splits.

The commit and the SHA-256 of every file are written to ``MANIFEST.json`` next to the data.

Usage::

    uv run --env-file .env python -m tools.lemma_tables.fetch_morphobr [--out DIR] [--force]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from artigo_env.core.paths import data_path

REPOSITORY = "LR-POR/MorphoBr"
UPSTREAM_URL = f"https://github.com/{REPOSITORY}"
COMMIT = "b49015cd308f128338f1aae4b49f46c887ce9608"
LICENSE_NAME = "Apache-2.0"
CLASS_DIRECTORIES = ("verbs", "nouns", "adjectives", "adverbs")
EXTRA_FILES = ("LICENSE", "README.org", "TAGSET.org")
_TREE_URL = f"https://api.github.com/repos/{REPOSITORY}/git/trees/{COMMIT}?recursive=1"
_RAW_URL = f"https://raw.githubusercontent.com/{REPOSITORY}/{COMMIT}/"
_USER_AGENT = "artigo-env-fetch/1"


def _get(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(request, timeout=120) as response:
        data: bytes = response.read()
    return data


def list_files() -> list[str]:
    """Paths of the files to fetch at the pinned commit, sorted."""
    tree = json.loads(_get(_TREE_URL))
    if tree.get("truncated"):
        raise RuntimeError("GitHub returned a truncated tree listing")
    wanted: list[str] = []
    for entry in tree["tree"]:
        path = entry["path"]
        if entry["type"] != "blob":
            continue
        top = path.split("/", 1)[0]
        if path in EXTRA_FILES or (top in CLASS_DIRECTORIES and path.endswith(".dict")):
            wanted.append(path)
    return sorted(wanted)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(out: Path, force: bool = False) -> dict[str, object]:
    files = list_files()
    out.mkdir(parents=True, exist_ok=True)

    def one(relative: str) -> None:
        target = out.joinpath(*relative.split("/"))
        if target.is_file() and not force:
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        data = _get(_RAW_URL + relative)
        partial = target.with_name(target.name + ".part")
        partial.write_bytes(data)
        partial.replace(target)

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(one, files))
    manifest: dict[str, object] = {
        "source": UPSTREAM_URL,
        "license": LICENSE_NAME,
        "commit": COMMIT,
        "files": {relative: sha256_file(out.joinpath(*relative.split("/"))) for relative in files},
    }
    (out / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=None, help="default: <data root>/raw/morphobr")
    parser.add_argument("--force", action="store_true", help="download files already present")
    args = parser.parse_args(argv)
    out = args.out if args.out is not None else data_path("raw", "morphobr")
    manifest = fetch(out, force=args.force)
    files = manifest["files"]
    assert isinstance(files, dict)
    print(f"MorphoBr {COMMIT[:12]}: {len(files)} files in {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
