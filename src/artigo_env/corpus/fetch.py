"""S0 fetch: put the pinned raw files of a source under ``raw/<source>/`` (SDD 02 §B.8).

Every file is checked against the SHA-256 and size pinned in the manifest. A file that does not
match is deleted and the fetch fails (REQ-REPO-006). Only open ``hf_parquet`` sources are
downloaded automatically; other sources explain how to obtain the files, and files placed by
hand are verified the same way.
"""

from __future__ import annotations

import hashlib
import shutil
from collections.abc import Callable, Sequence
from pathlib import Path, PurePosixPath

from artigo_env.core.errors import ArtigoEnvError
from artigo_env.corpus.manifest import SourceFile, SourceManifestEntry

CHUNK = 1 << 20  # 1 MiB

Download = Callable[[SourceManifestEntry, SourceFile, Path], None]
"""Writes ``file`` of ``entry`` at the given target path (injected in tests)."""

Log = Callable[[str], None]


class FetchError(ArtigoEnvError):
    """A source could not be fetched."""


class ChecksumMismatchError(FetchError):
    """A raw file does not match the SHA-256 or size pinned in the manifest (REQ-REPO-006)."""


class TermsNotAcceptedError(FetchError):
    """The source has terms to accept and ``accept_terms`` was not given."""


class ManualFetchRequired(FetchError):
    """The files of this source cannot be downloaded automatically; the message says how."""


def sha256_file(path: Path) -> str:
    """SHA-256 of a file, read in 1 MiB blocks."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(CHUNK):
            digest.update(block)
    return digest.hexdigest()


def file_problem(path: Path, expected: SourceFile) -> str | None:
    """``None`` if ``path`` matches the pinned size and SHA-256, else a description."""
    size = path.stat().st_size
    if size != expected.size:
        return f"size {size} bytes, expected {expected.size}"
    actual = sha256_file(path)
    if actual != expected.sha256:
        return f"SHA-256 {actual}, expected {expected.sha256}"
    return None


def downloadable(entry: SourceManifestEntry) -> bool:
    """Whether ``fetch`` can download this source by itself."""
    return entry.access == "open" and entry.kind == "hf_parquet"


def select_files(entry: SourceManifestEntry, names: Sequence[str] | None) -> list[SourceFile]:
    """Pinned files to fetch: all, or those whose path, file name or stem is in ``names``."""
    if names is None:
        return list(entry.files)
    chosen: list[SourceFile] = []
    for name in names:
        matches = [
            f
            for f in entry.files
            if name in (f.path, PurePosixPath(f.path).name, PurePosixPath(f.path).stem)
        ]
        if not matches:
            pinned = ", ".join(f.path for f in entry.files) or "<none>"
            raise FetchError(f"{name!r} is not a pinned file of {entry.id!r}; pinned: {pinned}")
        chosen.extend(f for f in matches if f not in chosen)
    return chosen


def instructions(entry: SourceManifestEntry, raw_root: Path) -> str:
    """How to obtain the files of a source that ``fetch`` does not download."""
    target = entry.source_dir(raw_root)
    if entry.kind == "wikipedia_extracts":
        subdir = entry.options.get("subdir", "<snapshot>")
        return (
            f"{entry.id} is a snapshot of the Wikipedia TextExtracts API, written by "
            f"'python -m tools.w.fetch_extracts --titles <list.tsv> --snapshot {subdir}' "
            f"into {target / subdir}; data fetch does not download it."
        )
    if entry.kind == "demo":
        return f"{entry.id} is bundled with the package; there is nothing to fetch."
    where = f" from {entry.url}" if entry.url else ""
    how = {
        "registration": "register with the provider",
        "request": "request access from the provider",
        "open": "download the files",
    }[entry.access]
    files = "".join(f"\n  {f.path} ({f.size} bytes)" for f in entry.files)
    return (
        f"{entry.id} cannot be downloaded automatically: {how}{where} and place the files "
        f"under {target}, then run this command again to verify them.{files}"
    )


def hf_download(entry: SourceManifestEntry, file: SourceFile, target: Path) -> None:
    """Download one file of a Hugging Face dataset at the pinned revision to ``target``."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise FetchError(
            "downloading needs the optional 'data' dependencies: pip install 'artigo-env[data]' "
            "(or uv sync --extra data)"
        ) from None
    repo = entry.options.get("repo")
    if not repo:
        raise FetchError(f"source {entry.id!r} needs a 'repo' option to download from")
    if not entry.revision:
        raise FetchError(f"source {entry.id!r} has no pinned revision")
    depth = len(PurePosixPath(file.path).parts)
    local_dir = target.parents[depth - 1]
    path = Path(
        hf_hub_download(
            repo_id=repo,
            filename=file.path,
            repo_type="dataset",
            revision=entry.revision,
            local_dir=local_dir,
        )
    )
    if path.resolve() != target.resolve():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(path, target)


def fetch(
    entry: SourceManifestEntry,
    raw_root: Path,
    *,
    accept_terms: bool = False,
    files: Sequence[str] | None = None,
    download: Download | None = None,
    log: Log | None = None,
) -> list[Path]:
    """Fetch and verify the pinned files of ``entry`` under ``raw_root/<source dir>/``.

    Returns the verified paths. Files already present and matching are not downloaded again; a
    file that does not match is deleted, then downloaded again when the source is downloadable,
    and a download that does not match is deleted and fails (REQ-REPO-006).
    """
    say = log or (lambda _message: None)
    get = hf_download if download is None else download
    if entry.terms and not accept_terms:
        raise TermsNotAcceptedError(
            f"{entry.id} has terms of use that must be accepted with --accept-terms:\n{entry.terms}"
        )
    if entry.kind == "demo":
        say(instructions(entry, raw_root))
        return []
    if not entry.files:
        if downloadable(entry):
            raise FetchError(
                f"no files are pinned for {entry.id!r}; pin the revision, paths, SHA-256 and "
                "sizes in sources.yaml first"
            )
        raise ManualFetchRequired(instructions(entry, raw_root))

    source_dir = entry.source_dir(raw_root)
    done: list[Path] = []
    missing: list[SourceFile] = []
    for file in select_files(entry, files):
        target = source_dir.joinpath(*PurePosixPath(file.path).parts)
        if target.is_file():
            problem = file_problem(target, file)
            if problem is None:
                say(f"ok {file.path} (already present, {file.size} bytes)")
                done.append(target)
                continue
            target.unlink()
            if not downloadable(entry):
                raise ChecksumMismatchError(
                    f"{target} does not match the manifest ({problem}); the file was deleted"
                )
            say(f"deleted {file.path}: {problem}; downloading it again")
        if not downloadable(entry):
            missing.append(file)
            continue
        say(f"downloading {file.path} ({file.size} bytes)")
        target.parent.mkdir(parents=True, exist_ok=True)
        get(entry, file, target)
        if not target.is_file():
            raise FetchError(f"download of {file.path} did not produce {target}")
        problem = file_problem(target, file)
        if problem is not None:
            target.unlink()
            raise ChecksumMismatchError(
                f"downloaded {file.path} does not match the manifest ({problem}); "
                "the file was deleted"
            )
        say(f"ok {file.path} ({file.size} bytes)")
        done.append(target)
    if missing:
        names = ", ".join(f.path for f in missing)
        raise ManualFetchRequired(f"missing: {names}\n{instructions(entry, raw_root)}")
    return done
