"""Reader of the minimal knowledge store of phase F4 (SDD 10 §4, DEC-074; WP4.4).

A knowledge store holds per-action statistics of the training partition of a build, indexed by
the action ids of one vocabulary V. The F4 artifact has three files:

- ``lemma_stats.npy``: structured ``float32`` array of length ``|V|`` with the fields ``freq``
  (token frequency in titles and bodies), ``doc_freq`` (documents whose title or body contain
  the key) and ``mean_count_present`` (``freq / doc_freq``, 0 when the key never occurs);
- ``title_stats.npy``: ``int64[|V|]``, number of training documents whose title contains the key;
- ``manifest.json``: identity (``knowledge_id``), build, ruleset, vocabulary, source
  configuration, regime and the SHA-256 of both arrays.

``KnowledgeStore.open`` memory-maps the arrays; the properties return contiguous in-memory copies
made once, so callers can index them freely. The builder lives in ``knowledge_build``.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from artigo_env.core.errors import ArtigoEnvError
from artigo_env.core.paths import data_path
from artigo_env.core.resources import resource_path
from artigo_env.core.vocab import Vocabulary

KNOWLEDGE_FORMAT_VERSION = 1
MANIFEST_FILE = "manifest.json"
LEMMA_STATS_FILE = "lemma_stats.npy"
TITLE_STATS_FILE = "title_stats.npy"
LEMMA_STATS_DTYPE = np.dtype([("freq", "<f4"), ("doc_freq", "<f4"), ("mean_count_present", "<f4")])
SOURCE_CONFIGS = ("X->W", "W->W", "X+W->W", "S->W", "X+S->W")
REGIMES = ("open", "closed", "oracle")


class KnowledgeMismatchError(ArtigoEnvError, ValueError):
    """The knowledge store does not match the active configuration (REQ-SVC-005)."""


def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class KnowledgeStore:
    """Read access to a knowledge directory written by ``build_knowledge``."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        manifest_path = self.path / MANIFEST_FILE
        if not manifest_path.is_file():
            raise ArtigoEnvError(f"not a knowledge directory (no {MANIFEST_FILE}): {self.path}")
        self.manifest: dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))
        version = self.manifest.get("format_version")
        if version != KNOWLEDGE_FORMAT_VERSION:
            raise ArtigoEnvError(
                f"unsupported knowledge format version {version!r} "
                f"(expected {KNOWLEDGE_FORMAT_VERSION})"
            )
        self.knowledge_id: str = str(self.manifest["knowledge_id"])
        self.n_docs: int = int(self.manifest["n_docs"])
        self.vocab_id: str = str(self.manifest["vocab_id"])
        self.vocab_size: int = int(self.manifest["vocab_size"])
        self.build_id: str = str(self.manifest["build_id"])
        self.ruleset_hash: str = str(self.manifest["ruleset_hash"])
        self.source_config: str = str(self.manifest["source_config"])
        self.regime: str = str(self.manifest["regime"])
        lemma = np.load(self.path / LEMMA_STATS_FILE, mmap_mode="r")
        title = np.load(self.path / TITLE_STATS_FILE, mmap_mode="r")
        if lemma.dtype != LEMMA_STATS_DTYPE or lemma.shape != (self.vocab_size,):
            raise ArtigoEnvError(f"inconsistent {LEMMA_STATS_FILE} in {self.path}")
        if title.dtype != np.dtype("<i8") or title.shape != (self.vocab_size,):
            raise ArtigoEnvError(f"inconsistent {TITLE_STATS_FILE} in {self.path}")
        # Copy once into contiguous arrays so that no file stays mapped (Windows locks them).
        self._lemma: dict[str, npt.NDArray[np.float32]] = {
            name: np.ascontiguousarray(lemma[name], dtype=np.float32)
            for name in LEMMA_STATS_DTYPE.names or ()
        }
        self._title_doc_freq = np.ascontiguousarray(title, dtype=np.int64)
        del lemma, title

    @classmethod
    def open(cls, path: Path | str, *, vocab: Vocabulary | None = None) -> KnowledgeStore:
        """Open ``path``; with ``vocab``, fail unless the store was built for that vocabulary."""
        store = cls(Path(path))
        if vocab is not None:
            store.check(vocab=vocab)
        return store

    def check(
        self,
        *,
        vocab: Vocabulary | None = None,
        ruleset_hash: str | None = None,
        source_config: str | None = None,
        regime: str | None = None,
    ) -> None:
        """Raise ``KnowledgeMismatchError`` when any given item differs from the manifest."""
        problems: list[str] = []
        if vocab is not None and (vocab.vocab_id != self.vocab_id or vocab.size != self.vocab_size):
            problems.append(
                f"vocabulary {vocab.vocab_id!r} (|V|={vocab.size}) but the store was built for "
                f"{self.vocab_id!r} (|V|={self.vocab_size})"
            )
        if ruleset_hash is not None and ruleset_hash != self.ruleset_hash:
            problems.append(
                f"ruleset hash {ruleset_hash[:16]} but the store has {self.ruleset_hash[:16]}"
            )
        if source_config is not None and source_config != self.source_config:
            problems.append(
                f"source configuration {source_config!r} but the store has {self.source_config!r}"
            )
        if regime is not None and regime != self.regime:
            problems.append(f"regime {regime!r} but the store has {self.regime!r}")
        if problems:
            raise KnowledgeMismatchError(
                f"knowledge store {self.knowledge_id} does not match the active configuration: "
                + "; ".join(problems)
            )

    # -- statistics -------------------------------------------------------------------------

    @property
    def lemma_stats(self) -> dict[str, npt.NDArray[np.float32]]:
        """``freq``, ``doc_freq`` and ``mean_count_present`` as ``float32[|V|]``."""
        return dict(self._lemma)

    @property
    def title_stats(self) -> dict[str, npt.NDArray[np.int64]]:
        """``title_doc_freq`` as ``int64[|V|]``."""
        return {"title_doc_freq": self._title_doc_freq}

    def __repr__(self) -> str:
        return (
            f"KnowledgeStore(knowledge_id={self.knowledge_id!r}, vocab_id={self.vocab_id!r}, "
            f"n_docs={self.n_docs}, source_config={self.source_config!r}, regime={self.regime!r})"
        )


def _candidates(root: Path) -> Iterator[Path]:
    if root.is_dir():
        for manifest in sorted(root.glob(f"*/{MANIFEST_FILE}")):
            yield manifest.parent


def find_knowledge(
    vocab_id: str,
    *,
    source_config: str = "X->W",
    regime: str = "open",
    root: Path | None = None,
) -> Path:
    """Directory of the knowledge store for ``vocab_id`` (packaged demo, then the data root).

    With ``root`` only that directory is searched. Exactly one match is required.
    """
    if root is not None:
        roots = [Path(root)]
    else:
        roots = [Path(str(resource_path("demo", "knowledge")))]
        with contextlib.suppress(ArtigoEnvError):  # no data root configured: demo only
            roots.append(data_path("knowledge"))
    matches: list[Path] = []
    for directory in roots:
        for candidate in _candidates(directory):
            data = json.loads((candidate / MANIFEST_FILE).read_text(encoding="utf-8"))
            if (
                data.get("vocab_id") == vocab_id
                and data.get("source_config") == source_config
                and data.get("regime") == regime
            ):
                matches.append(candidate)
        if matches:
            break
    if len(matches) != 1:
        where = ", ".join(str(r) for r in roots)
        found = "no" if not matches else f"{len(matches)}"
        raise ArtigoEnvError(
            f"{found} knowledge store(s) for vocabulary {vocab_id!r} ({source_config}, {regime}) "
            f"under {where}; build one with 'artigo-env knowledge build'"
        )
    return matches[0]
