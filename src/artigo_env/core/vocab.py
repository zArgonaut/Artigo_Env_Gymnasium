"""Reader of the action vocabulary artifact (SDD 09 §2, DEC-064).

The vocabulary V is the fixed list of lemma keys an agent can guess, built once per (training
build, ruleset) by ``artigo_env.corpus.vocab.build_vocab``. Action ``i`` guesses ``key(i)``: a
lemma, or ``n:<digits>`` for a numeric lexeme. The same artifact gives observation ids to the
ruleset's stopwords (``|V| + i``) and to visible forms (the form vocabulary F).

``Vocabulary.open`` is cheap: the numeric arrays are memory-mapped and the text tables are read
only on first use. Call ``close`` (or use the object as a context manager) before the directory
has to be deleted, which matters on Windows where mapped files are locked; arrays obtained from
the properties keep their file mapped while they are referenced.

This module lives in ``core`` so services and agents can use V without importing ``corpus``; it
depends only on NumPy, pyarrow and the standard library.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pyarrow.parquet as pq

from artigo_env.core.errors import ArtigoEnvError

VOCAB_FORMAT_VERSION = 1
NUMERIC_PREFIX = "n:"
ALL_LENGTHS = np.uint64(0xFFFF_FFFF_FFFF_FFFF)

LEMMAS_FILE = "lemmas.parquet"
STOP_FILE = "stop.parquet"
FORMS_FILE = "forms.parquet"
MANIFEST_FILE = "manifest.json"
ARRAY_FILES = {
    "freq": "freq.npy",
    "doc_freq": "doc_freq.npy",
    "is_numeric": "is_numeric.npy",
    "length_mask": "length_mask.npy",
}
MAPS_DIR = "maps"


def length_bit(n_chars: int) -> int:
    """Bit index of a length in ``length_mask``: ``min(n_chars, 64) - 1`` (``n_chars >= 1``)."""
    return min(int(n_chars), 64) - 1


class Vocabulary:
    """Read access to a vocabulary directory written by ``build_vocab``."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        manifest_path = self.path / MANIFEST_FILE
        if not manifest_path.is_file():
            raise ArtigoEnvError(f"not a vocabulary directory (no {MANIFEST_FILE}): {self.path}")
        self.manifest: dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))
        version = self.manifest.get("format_version")
        if version != VOCAB_FORMAT_VERSION:
            raise ArtigoEnvError(
                f"unsupported vocabulary format version {version!r} "
                f"(expected {VOCAB_FORMAT_VERSION})"
            )
        self.vocab_id: str = str(self.manifest["vocab_id"])
        self.size: int = int(self.manifest["size"])
        self.n_stop: int = int(self.manifest["n_stop"])
        self.n_forms: int = int(self.manifest["n_forms"])
        self._arrays: dict[str, np.memmap[Any, Any]] = {
            name: np.load(self.path / filename, mmap_mode="r")
            for name, filename in ARRAY_FILES.items()
        }
        for name, array in self._arrays.items():
            if array.shape != (self.size,):
                raise ArtigoEnvError(f"inconsistent vocabulary array {name!r} in {self.path}")
        self._keys: list[str] | None = None
        self._key_ids: dict[str, int] | None = None
        self._stops: list[str] | None = None
        self._stop_ids: dict[str, int] | None = None
        self._forms: list[str] | None = None
        self._form_ids: dict[str, int] | None = None
        self._closed = False

    @classmethod
    def open(cls, path: Path | str) -> Vocabulary:
        return cls(Path(path))

    # -- lifecycle --------------------------------------------------------------------------

    def close(self) -> None:
        """Drop every memory map; the vocabulary cannot be used afterwards."""
        if self._closed:
            return
        self._closed = True
        self._arrays.clear()

    def __enter__(self) -> Vocabulary:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()

    def _check_open(self) -> None:
        if self._closed:
            raise ArtigoEnvError("the vocabulary is closed")

    def _array(self, name: str) -> np.memmap[Any, Any]:
        self._check_open()
        return self._arrays[name]

    # -- arrays -----------------------------------------------------------------------------

    @property
    def freq(self) -> npt.NDArray[np.int64]:
        """Token frequency of each action's key in the training partition."""
        return self._array("freq")

    @property
    def doc_freq(self) -> npt.NDArray[np.int64]:
        """Number of training documents (title or body) containing each action's key."""
        return self._array("doc_freq")

    @property
    def is_numeric(self) -> npt.NDArray[np.bool_]:
        return self._array("is_numeric")

    @property
    def length_mask(self) -> npt.NDArray[np.uint64]:
        """Bit ``min(len, 64) - 1`` set when a known form of the lemma has that length.

        Numeric lexemes have every bit set (a number matches inside tokens of any length).
        """
        return self._array("length_mask")

    # -- text tables (loaded on first use) --------------------------------------------------

    def _read_column(self, filename: str, column: str) -> list[str]:
        self._check_open()
        table = pq.read_table(self.path / filename, columns=[column], memory_map=False)
        values: list[str] = table.column(column).to_pylist()
        return values

    def keys(self) -> list[str]:
        """Every key in action-id order (a new list)."""
        if self._keys is None:
            self._keys = self._read_column(LEMMAS_FILE, "key")
            if len(self._keys) != self.size:
                raise ArtigoEnvError(f"inconsistent {LEMMAS_FILE} in {self.path}")
        return list(self._keys)

    def key(self, action_id: int) -> str:
        """The key guessed by ``action_id``: a lemma, or ``n:<digits>``."""
        if self._keys is None:
            self.keys()
        assert self._keys is not None
        if not 0 <= action_id < self.size:
            raise KeyError(f"unknown action id: {action_id}")
        return self._keys[action_id]

    def action_id(self, key: str) -> int:
        """Action id of ``key``, or ``-1`` when it is not in V."""
        if self._key_ids is None:
            if self._keys is None:
                self.keys()
            assert self._keys is not None
            self._key_ids = {text: i for i, text in enumerate(self._keys)}
        return self._key_ids.get(key, -1)

    def stopwords(self) -> list[str]:
        """The ruleset's stopwords in stop-index order (observation id = ``size + index``)."""
        if self._stops is None:
            self._stops = self._read_column(STOP_FILE, "form")
            if len(self._stops) != self.n_stop:
                raise ArtigoEnvError(f"inconsistent {STOP_FILE} in {self.path}")
        return list(self._stops)

    def stopword(self, i: int) -> str:
        """The stopword with stop index ``i`` (its observation id is ``size + i``)."""
        if self._stops is None:
            self.stopwords()
        assert self._stops is not None
        if not 0 <= i < self.n_stop:
            raise KeyError(f"unknown stop index: {i}")
        return self._stops[i]

    def stop_id(self, form: str) -> int:
        """Stop index of a lowercase stopword, or ``-1``."""
        if self._stop_ids is None:
            if self._stops is None:
                self.stopwords()
            assert self._stops is not None
            self._stop_ids = {text: i for i, text in enumerate(self._stops)}
        return self._stop_ids.get(form, -1)

    def forms(self) -> list[str]:
        """The form vocabulary F in form-id order (a new list)."""
        if self._forms is None:
            self._forms = self._read_column(FORMS_FILE, "form")
            if len(self._forms) != self.n_forms:
                raise ArtigoEnvError(f"inconsistent {FORMS_FILE} in {self.path}")
        return list(self._forms)

    def form(self, i: int) -> str:
        if self._forms is None:
            self.forms()
        assert self._forms is not None
        if not 0 <= i < self.n_forms:
            raise KeyError(f"unknown form id: {i}")
        return self._forms[i]

    def form_id(self, form: str) -> int:
        """Id of a lowercase form in F, or ``-1``."""
        if self._form_ids is None:
            if self._forms is None:
                self.forms()
            assert self._forms is not None
            self._form_ids = {text: i for i, text in enumerate(self._forms)}
        return self._form_ids.get(form, -1)

    def __len__(self) -> int:
        return self.size

    def __repr__(self) -> str:
        return (
            f"Vocabulary(vocab_id={self.vocab_id!r}, size={self.size}, n_stop={self.n_stop}, "
            f"n_forms={self.n_forms})"
        )
