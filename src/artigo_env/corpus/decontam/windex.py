"""The W index: the three barriers' data for the evaluation set, saved as plain files.

Layout of a saved index (every file deterministic for the same input):

* ``meta.json`` — format version and the barrier constants it was built with;
* ``titles.txt`` — normalized titles, sorted, one per line (UTF-8);
* ``grams.npy`` — sorted unique ``uint64`` 13-gram hashes;
* ``near_keys.npy``, ``near_offsets.npy``, ``near_shingles.npy``, ``near_signatures.npy`` —
  the near-duplicate index (keys = position of the article in the build input).
"""

from __future__ import annotations

import hashlib
import io
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from artigo_env.corpus.decontam import coverage, minhash
from artigo_env.corpus.decontam.coverage import TrainDoc, Verdict, WGramIndex, judge
from artigo_env.corpus.decontam.minhash import NearDuplicateIndex, shingle_hashes
from artigo_env.corpus.decontam.titles import TitleBarrier

FORMAT_VERSION = 1

_META = {
    "format_version": FORMAT_VERSION,
    "n_gram": coverage.N_GRAM,
    "shingle": minhash.SHINGLE,
    "n_perm": minhash.N_PERM,
    "bands": minhash.BANDS,
    "rows": minhash.ROWS,
    "seed": minhash.SEED,
    "gram_hash": "xxh64(seed=0, utf-8, sep=0x1f)",
}
_NEAR_FILES = {
    "keys": "near_keys.npy",
    "offsets": "near_offsets.npy",
    "shingles": "near_shingles.npy",
    "signatures": "near_signatures.npy",
}


def _npy_bytes(array: NDArray[np.generic]) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, np.ascontiguousarray(array), allow_pickle=False)
    return buffer.getvalue()


def files_hash(files: Mapping[str, bytes]) -> str:
    """SHA-256 over the files in name order (name, size and content of each)."""
    digest = hashlib.sha256()
    for name in sorted(files):
        content = files[name]
        digest.update(f"{name}\0{len(content)}\0".encode())
        digest.update(content)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class WIndex:
    """Title barrier, 13-gram index and near-duplicate index of W."""

    titles: TitleBarrier
    grams: WGramIndex
    near: NearDuplicateIndex

    @classmethod
    def build(
        cls, docs: Iterable[tuple[str, Sequence[str]]], extra_titles: Iterable[str] = ()
    ) -> WIndex:
        """``docs`` are ``(title, body tokens)``; ``extra_titles`` are W titles without text."""
        titles: list[str] = []
        gram_parts: list[NDArray[np.uint64]] = []
        near = NearDuplicateIndex()
        for key, (title, tokens) in enumerate(docs):
            titles.append(title)
            gram_parts.append(WGramIndex.build([tokens]).hashes)
            near.add(key, shingle_hashes(tokens))
        titles.extend(extra_titles)
        grams = WGramIndex(np.concatenate(gram_parts) if gram_parts else None)
        return cls(TitleBarrier(titles), grams, near)

    def judge(self, doc: TrainDoc, frequent: NDArray[np.uint64]) -> Verdict:
        """Pass C of the three-pass protocol for one document (all three barriers)."""
        return judge(doc, titles=self.titles, grams=self.grams, near=self.near, frequent=frequent)

    # --- persistence -------------------------------------------------------------------------

    def _files(self) -> dict[str, bytes]:
        files = {
            "meta.json": (json.dumps(_META, sort_keys=True, indent=2) + "\n").encode("utf-8"),
            "titles.txt": self.titles.to_text().encode("utf-8"),
            "grams.npy": _npy_bytes(self.grams.hashes),
        }
        for field, array in self.near.to_arrays().items():
            files[_NEAR_FILES[field]] = _npy_bytes(array)
        return files

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        for name, content in sorted(self._files().items()):
            (directory / name).write_bytes(content)

    @classmethod
    def load(cls, directory: Path) -> WIndex:
        meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
        if meta != _META:
            raise ValueError(
                f"W index at {directory.name!r} was built with different settings: {meta}"
            )
        arrays = {
            field: np.load(directory / name, allow_pickle=False)
            for field, name in _NEAR_FILES.items()
        }
        near = NearDuplicateIndex.from_arrays(
            arrays["keys"], arrays["offsets"], arrays["shingles"], arrays["signatures"]
        )
        titles = TitleBarrier.load(directory / "titles.txt")
        return cls(titles, WGramIndex.load(directory / "grams.npy"), near)

    def content_hash(self) -> str:
        """SHA-256 of the files :meth:`save` writes, in name order (see :func:`files_hash`)."""
        return files_hash(self._files())


def directory_hash(directory: Path) -> str:
    """:func:`files_hash` of a saved index directory; equals ``WIndex.content_hash()``."""
    names = (*_NEAR_FILES.values(), "meta.json", "titles.txt", "grams.npy")
    return files_hash({name: (directory / name).read_bytes() for name in names})
