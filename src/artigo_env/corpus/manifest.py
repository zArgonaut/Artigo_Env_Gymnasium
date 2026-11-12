"""Versioned manifest of raw data sources (SDD 06 §4, SDD 02 §B.8).

Every raw file the corpus pipeline reads is declared in ``assets/sources.yaml``: where it comes
from, which revision and file hashes are pinned, its licence and whether the author accepted it for
training. Paths in the manifest are relative to ``ARTIGO_DATA_ROOT/raw/<source dir>/``, where the
source dir is the source id unless the ``raw_dir`` option says otherwise.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from importlib.resources.abc import Traversable
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

import yaml

from artigo_env.core.config import ConfigBase
from artigo_env.core.errors import ArtigoEnvError
from artigo_env.core.resources import resource_path

KINDS = ("hf_parquet", "jsonl_gz", "wikipedia_extracts", "demo")
STATES = ("accepted", "pending", "rejected", "stats-only", "evaluation")
ACCESS = ("open", "registration", "request")
OPTION_KEYS = ("title", "body", "url", "date", "id", "repo", "subdir", "raw_dir")
MANIFEST_VERSION = 1

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SOURCE_ID = re.compile(r"^[a-z0-9][a-z0-9_]*$")


class SourceManifestError(ArtigoEnvError):
    """The source manifest is malformed (duplicate id, invalid state, bad file entry, ...)."""


class SourceNotAcceptedError(ArtigoEnvError):
    """A source that is not ``accepted`` was used where training data is required."""


def check_relative(path: str, what: str) -> str:
    """Return ``path`` if it is a non-empty relative POSIX path that stays below its root."""
    windows, posix = PureWindowsPath(path), PurePosixPath(path)
    if (
        not path
        or "\\" in path
        or windows.drive
        or windows.root
        or posix.is_absolute()
        or ".." in posix.parts
    ):
        raise ValueError(f"{what} must be a relative POSIX path without '..', got {path!r}")
    return path


@dataclass(frozen=True)
class SourceFile(ConfigBase):
    """One pinned raw file: path relative to the source dir, SHA-256 and size in bytes."""

    path: str
    sha256: str
    size: int

    def __post_init__(self) -> None:
        check_relative(self.path, "file path")
        if not isinstance(self.sha256, str) or not _SHA256.match(self.sha256):
            raise ValueError(f"sha256 of {self.path!r} must be 64 lowercase hex characters")
        if isinstance(self.size, bool) or not isinstance(self.size, int) or self.size < 0:
            raise ValueError(f"size of {self.path!r} must be a non-negative integer")


@dataclass(frozen=True)
class SourceManifestEntry(ConfigBase):
    """One source of raw documents and the rules for using it."""

    id: str
    kind: str  # one of KINDS
    state: str  # one of STATES
    priority: int  # lower = preferred when deduplicating across sources
    license_id: str
    url: str | None = None
    revision: str | None = None
    files: tuple[SourceFile, ...] = ()
    license_note: str = ""
    access: str = "open"  # one of ACCESS
    redistributable: bool = False
    terms: str | None = None  # text the user must accept with --accept-terms before fetching
    variant: str | None = None  # "pt-BR", "pt-PT" or None
    options: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not _SOURCE_ID.match(self.id):
            raise ValueError(f"invalid source id {self.id!r} (lowercase letters, digits, '_')")
        if self.kind not in KINDS:
            raise ValueError(f"unknown kind {self.kind!r}; expected one of {KINDS}")
        if self.state not in STATES:
            raise ValueError(f"invalid state {self.state!r}; expected one of {STATES}")
        if self.access not in ACCESS:
            raise ValueError(f"invalid access {self.access!r}; expected one of {ACCESS}")
        if isinstance(self.priority, bool) or not isinstance(self.priority, int):
            raise ValueError("priority must be an integer")
        if not isinstance(self.redistributable, bool):
            raise ValueError("redistributable must be true or false")
        if self.variant not in (None, "pt-BR", "pt-PT"):
            raise ValueError(f"variant must be 'pt-BR', 'pt-PT' or null, got {self.variant!r}")
        for name in ("license_id", "license_note"):
            if not isinstance(getattr(self, name), str):
                raise ValueError(f"{name} must be a string")
        for name in ("url", "revision", "terms"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, str):
                raise ValueError(f"{name} must be a string or null (quote it in YAML)")
        if not isinstance(self.options, Mapping):
            raise ValueError("options must be a mapping")
        for key, value in self.options.items():
            if key not in OPTION_KEYS:
                raise ValueError(f"unknown option {key!r}; expected one of {OPTION_KEYS}")
            if not isinstance(value, str):
                raise ValueError(f"option {key!r} must be a string (quote it in YAML)")
        for key in ("subdir", "raw_dir"):
            if key in self.options:
                check_relative(self.options[key], f"option {key!r}")
        paths = [f.path for f in self.files]
        if len(set(paths)) != len(paths):
            raise ValueError("file paths must be unique")

    @property
    def trainable(self) -> bool:
        return self.state == "accepted"

    def source_dir(self, raw_root: Path) -> Path:
        """Directory of this source's raw files: ``raw_root/<raw_dir or id>``."""
        return raw_root / self.options.get("raw_dir", self.id)

    def column(self, name: str) -> str | None:
        """Raw column mapped to a field: ``title``, ``body``, ``url``, ``date`` or ``id``."""
        return self.options.get(name)


def default_manifest() -> Traversable:
    """The manifest bundled with the package."""
    return resource_path("sources.yaml")


def parse_sources(data: Any) -> dict[str, SourceManifestEntry]:
    """Validate a parsed manifest document; returns entries keyed by id, in file order."""
    if not isinstance(data, Mapping) or not isinstance(data.get("sources"), list):
        raise SourceManifestError("manifest must be a mapping with a 'sources' list")
    version = data.get("version", MANIFEST_VERSION)
    if version != MANIFEST_VERSION:
        raise SourceManifestError(f"unsupported manifest version {version!r}")
    entries: dict[str, SourceManifestEntry] = {}
    for position, raw in enumerate(data["sources"]):
        label = raw.get("id", f"#{position}") if isinstance(raw, Mapping) else f"#{position}"
        if not isinstance(raw, Mapping):
            raise SourceManifestError(f"source {label}: entry must be a mapping")
        try:
            entry = SourceManifestEntry.from_dict(raw)
        except (ValueError, TypeError) as error:
            raise SourceManifestError(f"source {label!r}: {error}") from None
        if entry.id in entries:
            raise SourceManifestError(f"duplicate source id {entry.id!r}")
        entries[entry.id] = entry
    return entries


def load_sources(path: Traversable | Path | None = None) -> dict[str, SourceManifestEntry]:
    """Load and validate a source manifest (default: the package's ``assets/sources.yaml``)."""
    node = default_manifest() if path is None else path
    try:
        text = node.read_text(encoding="utf-8")
    except OSError as error:
        raise SourceManifestError(f"cannot read source manifest {node}: {error}") from None
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as error:
        raise SourceManifestError(f"source manifest {node} is not valid YAML: {error}") from None
    return parse_sources(data)


def get_source(
    source_id: str, sources: Mapping[str, SourceManifestEntry] | None = None
) -> SourceManifestEntry:
    """One entry by id, with the list of known ids in the error message."""
    table = load_sources() if sources is None else sources
    try:
        return table[source_id]
    except KeyError:
        known = ", ".join(sorted(table)) or "<none>"
        raise SourceManifestError(f"unknown source {source_id!r}; known: {known}") from None


def require_trainable(entry: SourceManifestEntry) -> None:
    """Refuse a source that the author has not accepted for training (REQ-COR-002)."""
    if entry.state != "accepted":
        raise SourceNotAcceptedError(
            f"source {entry.id!r} is {entry.state!r}, not 'accepted': it cannot be used for "
            "training (accept it in sources.yaml after the checks of SDD 06 §4.1)"
        )
