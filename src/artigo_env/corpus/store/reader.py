"""Memory-mapped reader of the compiled corpus and its bridge to the engine (SDD 06 §10).

Token arrays are opened with ``np.load(mmap_mode="r")`` and the texts with ``pa.memory_map``;
nothing is loaded whole except the small tables (document metadata, forms, lemmas). Building an
``Article`` from a stored document is vectorized: no tokenizer and no lemmatizer run.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from artigo_env.core.config import canonical_json, sha256_hex
from artigo_env.core.errors import ArtigoEnvError
from artigo_env.corpus.store.writer import FORMAT_VERSION, TEXT_COLUMNS
from artigo_env.corpus.types import DOC_SCHEMA, Row
from artigo_env.engine.article import assemble_article
from artigo_env.engine.rulesets.ruleset import Ruleset
from artigo_env.engine.types import Article, TokenSeq
from artigo_env.text.lemmatizers.base import Lemmatizer
from artigo_env.text.lemmatizers.registry import get_lemmatizer
from artigo_env.text.stopwords import StopwordSpec, load_stopwords

_ARRAYS = (
    "body_forms",
    "body_offsets",
    "body_spans",
    "body_flags",
    "body_case",
    "title_forms",
    "title_offsets",
    "title_spans",
    "title_flags",
)


class MissingLemmaViewError(ArtigoEnvError, KeyError):
    """The build has no form-to-lemma view for the requested lemmatizer."""

    def __str__(self) -> str:
        return str(self.args[0]) if self.args else ""


class LemmaView:
    """Form-to-lemma mapping of one lemmatizer inside a build.

    ``form2lemma[form_id]`` is a lemma id; lemma ids are global to the build and follow the
    order of first appearance over the form ids.
    """

    def __init__(self, directory: Path, key: str, name: str, content_hash: str) -> None:
        self.key = key
        self.name = name
        self.content_hash = content_hash
        self.form2lemma: npt.NDArray[np.uint32] = np.load(
            directory / "form2lemma.npy", mmap_mode="r"
        )
        table = pq.read_table(directory / "lemmas.parquet", columns=["lemma"])
        self._lemmas: pa.StringArray = table.column("lemma").combine_chunks()
        self._ids: dict[str, int] | None = None

    def __len__(self) -> int:
        return len(self._lemmas)

    def lemma_text(self, lemma_id: int) -> str:
        if not 0 <= lemma_id < len(self._lemmas):
            raise KeyError(f"unknown lemma id: {lemma_id}")
        text: str = self._lemmas[lemma_id].as_py()
        return text

    def lemma_id(self, lemma: str) -> int:
        """Id of ``lemma``, or ``-1`` when no form of the build has it (built on first use)."""
        if self._ids is None:
            self._ids = {text: i for i, text in enumerate(self._lemmas.to_pylist())}
        return self._ids.get(lemma, -1)

    def _release(self) -> None:
        del self.form2lemma


class CorpusStore:
    """Read access to a compiled corpus directory written by ``CorpusWriter``.

    Use ``CorpusStore.open(path)``; call ``close`` (or use it as a context manager) before the
    directory has to be deleted, which matters on Windows where mapped files are locked.
    Per-document accessors return copies, so they never keep a file mapped after ``close``.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.manifest: dict[str, Any] = json.loads(
            (self.path / "manifest.json").read_text(encoding="utf-8")
        )
        version = self.manifest.get("format_version")
        if version != FORMAT_VERSION:
            raise ArtigoEnvError(
                f"unsupported corpus format version {version!r} (expected {FORMAT_VERSION})"
            )
        self.tokenizer_hyphen: str = self.manifest["tokenizer_hyphen"]
        self.build_id: str = str(self.manifest["build_id"])
        self._arrays: dict[str, np.memmap[Any, Any]] = {
            name: np.load(self.path / f"{name}.npy", mmap_mode="r") for name in _ARRAYS
        }
        self._texts_file = pa.memory_map(str(self.path / "texts.arrow"), "r")
        texts = pa.ipc.open_file(self._texts_file).read_all()
        self._text_chunks = {name: texts.column(name).chunks for name in TEXT_COLUMNS}
        lengths = [len(chunk) for chunk in self._text_chunks["body"]]
        self._chunk_starts = np.concatenate([[0], np.cumsum(lengths, dtype=np.int64)])
        self.docs: pa.Table = pq.read_table(self.path / "docs.parquet", memory_map=True)
        self._doc_ids: pa.StringArray = self.docs.column("doc_id").combine_chunks()
        forms = pq.read_table(self.path / "forms.parquet", columns=["form"])
        self._forms: pa.StringArray = forms.column("form").combine_chunks()
        self._form_ids: dict[str, int] | None = None
        self._stop_masks: dict[str, npt.NDArray[np.bool_]] = {}
        self._view_keys: dict[str, str] = {}  # lemmatizer name -> view key
        self._views: dict[str, LemmaView] = {}
        self._closed = False
        if len(self._doc_ids) != self.n_docs or int(self._chunk_starts[-1]) != self.n_docs:
            raise ArtigoEnvError(f"inconsistent corpus directory: {self.path}")

    @classmethod
    def open(cls, path: Path | str) -> CorpusStore:
        return cls(Path(path))

    # -- lifecycle --------------------------------------------------------------------------

    def close(self) -> None:
        """Drop every memory map; the store cannot be used afterwards."""
        if self._closed:
            return
        self._closed = True
        self._arrays.clear()
        for view in self._views.values():
            view._release()
        self._views.clear()
        self._text_chunks.clear()
        self._texts_file.close()

    def __enter__(self) -> CorpusStore:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()

    def _array(self, name: str) -> np.memmap[Any, Any]:
        if self._closed:
            raise ArtigoEnvError("the corpus store is closed")
        return self._arrays[name]

    # -- documents --------------------------------------------------------------------------

    @property
    def n_docs(self) -> int:
        return int(self.manifest["n_docs"])

    @property
    def n_forms(self) -> int:
        return len(self._forms)

    def _check_doc(self, doc_idx: int) -> int:
        if not 0 <= doc_idx < self.n_docs:
            raise IndexError(f"doc_idx {doc_idx} out of range (0..{self.n_docs - 1})")
        return int(doc_idx)

    def _text(self, column: str, doc_idx: int) -> str | None:
        if self._closed:
            raise ArtigoEnvError("the corpus store is closed")
        i = self._check_doc(doc_idx)
        k = int(np.searchsorted(self._chunk_starts, i, side="right")) - 1
        value: str | None = self._text_chunks[column][k][i - int(self._chunk_starts[k])].as_py()
        return value

    def title_text(self, doc_idx: int) -> str:
        """Normalized title as tokenized by the writer (empty when the row had none)."""
        return self._text("title", doc_idx) or ""

    def body_text(self, doc_idx: int) -> str:
        return self._text("body", doc_idx) or ""

    def doc_id(self, doc_idx: int) -> str:
        value: str = self._doc_ids[self._check_doc(doc_idx)].as_py()
        return value

    def doc(self, doc_idx: int) -> Row:
        """The document as a ``DOC_SCHEMA`` row (texts normalized; token counts as stored)."""
        i = self._check_doc(doc_idx)
        row: Row = {}
        for name in DOC_SCHEMA.names:
            if name in TEXT_COLUMNS:
                row[name] = self._text(name, i)
            else:
                row[name] = self.docs.column(name)[i].as_py()
        return row

    def _bounds(self, part: str, doc_idx: int) -> tuple[int, int]:
        offsets = self._array(f"{part}_offsets")
        i = self._check_doc(doc_idx)
        return int(offsets[i]), int(offsets[i + 1])

    def title_forms(self, doc_idx: int) -> npt.NDArray[np.uint32]:
        a, b = self._bounds("title", doc_idx)
        return np.array(self._array("title_forms")[a:b], dtype=np.uint32)

    def body_forms(self, doc_idx: int) -> npt.NDArray[np.uint32]:
        a, b = self._bounds("body", doc_idx)
        return np.array(self._array("body_forms")[a:b], dtype=np.uint32)

    def body_case(self, doc_idx: int) -> npt.NDArray[np.uint8]:
        """Case class per body token: 0 lower, 1 first upper, 2 all upper, 3 mixed."""
        a, b = self._bounds("body", doc_idx)
        return np.array(self._array("body_case")[a:b], dtype=np.uint8)

    def n_body_tokens(self, doc_idx: int) -> int:
        a, b = self._bounds("body", doc_idx)
        return b - a

    def n_title_tokens(self, doc_idx: int) -> int:
        a, b = self._bounds("title", doc_idx)
        return b - a

    def split_indices(self, split: str) -> npt.NDArray[np.int64]:
        """Indices of the documents whose ``split`` column equals ``split``."""
        mask = pc.fill_null(pc.equal(self.docs.column("split"), split), False)
        flags = np.asarray(mask.to_numpy(zero_copy_only=False), dtype=np.bool_)
        return np.flatnonzero(flags).astype(np.int64)

    # -- forms and lemma views --------------------------------------------------------------

    def form(self, form_id: int) -> str:
        if not 0 <= form_id < len(self._forms):
            raise KeyError(f"unknown form id: {form_id}")
        text: str = self._forms[form_id].as_py()
        return text

    def form_id(self, form: str) -> int:
        """Id of a lowercase form, or ``-1`` (dictionary built on first use)."""
        if self._form_ids is None:
            self._form_ids = {text: i for i, text in enumerate(self._forms.to_pylist())}
        return self._form_ids.get(form, -1)

    def stop_mask(self, spec: StopwordSpec) -> npt.NDArray[np.bool_]:
        """``mask[form_id]``: whether the form is in the stopword set of ``spec`` (cached)."""
        key = canonical_json(spec)
        mask = self._stop_masks.get(key)
        if mask is None:
            words = pa.array(sorted(load_stopwords(spec).words), type=pa.string())
            hits = pc.is_in(self._forms, value_set=words)
            mask = np.asarray(hits.to_numpy(zero_copy_only=False), dtype=np.bool_)
            mask.flags.writeable = False
            self._stop_masks[key] = mask
        return mask

    @property
    def lemma_views(self) -> Mapping[str, Mapping[str, str]]:
        """View key -> ``{"name": ..., "content_hash": ...}`` as recorded in the manifest."""
        views: Mapping[str, Mapping[str, str]] = self.manifest.get("lemma_views", {})
        return views

    def lemma_view(self, lemmatizer: Lemmatizer | str) -> LemmaView:
        """The view of ``lemmatizer`` (an object or a registered name); clear error if absent."""
        if self._closed:
            raise ArtigoEnvError("the corpus store is closed")
        if isinstance(lemmatizer, str):
            key = self._view_keys.get(lemmatizer)
            if key is None:
                key = self._view_keys[lemmatizer] = get_lemmatizer(lemmatizer).content_hash()[:16]
            name = lemmatizer
        else:
            key = lemmatizer.content_hash()[:16]
            name = lemmatizer.name
        view = self._views.get(key)
        if view is not None:
            return view
        entry = self.lemma_views.get(key)
        if entry is None:
            available = ", ".join(
                f"{info['name']} ({view_key})"
                for view_key, info in sorted(self.lemma_views.items())
            )
            raise MissingLemmaViewError(
                f"the corpus build has no lemma view for lemmatizer {name!r} ({key}); "
                f"available: {available or 'none'}"
            )
        view = LemmaView(self.path / "lemma_views" / key, key, entry["name"], entry["content_hash"])
        self._views[key] = view
        return view

    # -- bridge to the engine ---------------------------------------------------------------

    def _sequence(
        self,
        part: str,
        doc_idx: int,
        text: str,
        view: LemmaView,
        stop_mask: npt.NDArray[np.bool_],
    ) -> TokenSeq:
        a, b = self._bounds(part, doc_idx)
        forms = np.array(self._array(f"{part}_forms")[a:b], dtype=np.uint32)
        spans = np.array(self._array(f"{part}_spans")[a:b], dtype=np.int64)
        flags = np.array(self._array(f"{part}_flags")[a:b], dtype=np.uint8)
        lemmas = view.form2lemma[forms].astype(np.int64)
        stop = stop_mask[forms]
        # Lengths come from the spans: the length of the spelling, exactly as the engine's
        # ``len(token.text)`` (lowercasing may change the length of a few characters).
        lengths = (spans[:, 1] - spans[:, 0]).astype(np.uint16)
        surface = tuple(
            map(text.__getitem__, map(slice, spans[:, 0].tolist(), spans[:, 1].tolist()))
        )
        return TokenSeq(forms, lemmas, lengths, flags, stop, surface)

    def article(self, doc_idx: int, ruleset: Ruleset) -> Article:
        """The engine ``Article`` of a stored document under ``ruleset``.

        Fails with ``IncompatibleRulesetError`` when the build used another hyphen mode
        (REQ-RUL-004) and with ``MissingLemmaViewError`` when the ruleset's lemmatizer has no
        view; ``UnplayableArticleError`` as ``assemble_article``.
        """
        ruleset.check_build(self.tokenizer_hyphen)
        view = self.lemma_view(ruleset.lemmatizer)
        stop_mask = self.stop_mask(ruleset.stopwords)
        i = self._check_doc(doc_idx)
        title = self._sequence("title", i, self.title_text(i), view, stop_mask)
        body = self._sequence("body", i, self.body_text(i), view, stop_mask)
        ref = sha256_hex(f"{self.build_id}:{self.doc_id(i)}")
        return assemble_article(ref, title, body, ruleset.title_parenthetical)
