"""Writer of the compiled corpus (SDD 06 §10, DEC-049).

The writer tokenizes each document once with the core tokenizer and appends its tokens to flat
arrays. Token arrays are streamed to raw files while documents are added, so memory stays
bounded by the form dictionary and the per-document metadata; ``finish`` turns them into
``.npy`` files and writes the tables and the manifest.

Output is byte-identical for the same inputs in the same order: form and lemma ids follow first
appearance, JSON is canonical, Parquet and Arrow files are written with fixed settings and no
timestamps.
"""

from __future__ import annotations

import contextlib
import os
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np
import numpy.typing as npt
import pyarrow as pa
import pyarrow.parquet as pq

from artigo_env.core.config import canonical_json
from artigo_env.corpus.types import DOC_SCHEMA, Row
from artigo_env.engine.types import (
    FLAG_HAS_DIGIT,
    FLAG_HEADER,
    FLAG_INSIDE_PARENTHESES,
    FLAG_PARAGRAPH_BREAK,
)
from artigo_env.text.guess import clean_guess
from artigo_env.text.lemmatizers.base import Lemmatizer
from artigo_env.text.normalize import lower, normalize
from artigo_env.text.tokenizer import Token, tokenize

FORMAT_VERSION = 1

CASE_LOWER = 0
CASE_FIRST_UPPER = 1
CASE_ALL_UPPER = 2
CASE_MIXED = 3

# Columns of docs.parquet: the document index, then every DOC_SCHEMA column except the texts,
# which live in texts.arrow.
TEXT_COLUMNS = ("title", "body")
DOCS_SCHEMA = pa.schema(
    [pa.field("doc_idx", pa.int64(), nullable=False)]
    + [field for field in DOC_SCHEMA if field.name not in TEXT_COLUMNS]
)
TEXTS_SCHEMA = pa.schema([pa.field("title", pa.string()), pa.field("body", pa.string())])
FORMS_SCHEMA = pa.schema(
    [
        pa.field("form_id", pa.uint32(), nullable=False),
        pa.field("form", pa.string(), nullable=False),
        pa.field("n_chars", pa.uint16(), nullable=False),
        pa.field("all_digits", pa.bool_(), nullable=False),
        pa.field("has_digit", pa.bool_(), nullable=False),
    ]
)
LEMMAS_SCHEMA = pa.schema(
    [
        pa.field("lemma_id", pa.uint32(), nullable=False),
        pa.field("lemma", pa.string(), nullable=False),
    ]
)

# Token arrays: name -> (dtype, number of columns). Everything is little-endian on disk.
_TOKEN_ARRAYS: Mapping[str, tuple[str, int]] = {
    "body_forms": ("<u4", 1),
    "body_spans": ("<u4", 2),
    "body_flags": ("u1", 1),
    "body_case": ("u1", 1),
    "title_forms": ("<u4", 1),
    "title_spans": ("<u4", 2),
    "title_flags": ("u1", 1),
}

_TEXT_BATCH_ROWS = 1024
_TEXT_BATCH_BYTES = 64 * 1024 * 1024
_COPY_CHUNK = 16 * 1024 * 1024
_DIGITS = frozenset("0123456789")
_MAX_SPAN = 2**32 - 1


def write_parquet(table: pa.Table, path: Path) -> None:
    """Write ``table`` with fixed settings so equal tables give identical bytes."""
    pq.write_table(
        table,
        path,
        compression="zstd",
        compression_level=3,
        use_dictionary=True,
        write_statistics=True,
        version="2.6",
        data_page_version="1.0",
        row_group_size=1 << 20,
    )


def token_case(text: str) -> int:
    """Case class of a token's spelling: 0 lower, 1 first upper, 2 all upper, 3 mixed.

    A token without cased characters (digits) is lowercase; a single capital letter is
    "first upper".
    """
    low = lower(text)
    if text == low:
        return CASE_LOWER
    if text[0] != low[0] and text[1:] == lower(text[1:]):
        return CASE_FIRST_UPPER
    if text == text.upper():
        return CASE_ALL_UPPER
    return CASE_MIXED


def token_flags(text: str, tokens: list[Token]) -> list[int]:
    """Flags exactly as ``engine.ArticleBuilder`` computes them (decision 8 of the F3 plan).

    bit0 header, bit1 paragraph break before, bit2 inside parentheses, bit3 has a digit.
    ``text`` is the normalized text the tokens refer to.
    """
    flags: list[int] = []
    depth = 0
    previous_end = 0
    for i, token in enumerate(tokens):
        gap = text[previous_end : token.start]
        depth = max(0, depth + gap.count("(") - gap.count(")"))
        flag = 0
        if token.header:
            flag |= FLAG_HEADER
        if i > 0 and "\n" in gap:
            flag |= FLAG_PARAGRAPH_BREAK
        if depth > 0:
            flag |= FLAG_INSIDE_PARENTHESES
        if not _DIGITS.isdisjoint(token.text):
            flag |= FLAG_HAS_DIGIT
        flags.append(flag)
        previous_end = token.end
    return flags


def _write_npy_from_raw(raw: Path, out: Path, dtype: str, columns: int) -> int:
    """Turn a raw little-endian dump into an ``.npy`` file; returns the number of rows."""
    itemsize = np.dtype(dtype).itemsize * columns
    size = raw.stat().st_size
    if size % itemsize:
        raise ValueError(f"corrupt raw token file: {raw.name}")
    rows = size // itemsize
    shape: tuple[int, ...] = (rows, columns) if columns > 1 else (rows,)
    header = {
        "descr": np.lib.format.dtype_to_descr(np.dtype(dtype)),
        "fortran_order": False,
        "shape": shape,
    }
    with out.open("wb") as dst, raw.open("rb") as src:
        np.lib.format.write_array_header_1_0(dst, header)
        shutil.copyfileobj(src, dst, _COPY_CHUNK)
    return rows


class CorpusWriter:
    """Compiles documents into a store directory (see ``reader.CorpusStore``).

    Usage: ``add`` every document, optionally ``add_lemma_view`` for each lemmatizer to serve,
    then ``finish``. Files are written under ``<out_dir>.tmp`` and renamed at the end.
    """

    def __init__(self, out_dir: Path, tokenizer_hyphen: str = "split") -> None:
        if tokenizer_hyphen not in ("split", "join"):
            raise ValueError(
                f"unknown hyphen mode: {tokenizer_hyphen!r} (expected 'split' or 'join')"
            )
        self.out_dir = Path(out_dir)
        if self.out_dir.exists():
            raise FileExistsError(f"corpus directory already exists: {self.out_dir}")
        self.tokenizer_hyphen = tokenizer_hyphen
        self._tmp = self.out_dir.with_name(self.out_dir.name + ".tmp")
        if self._tmp.exists():
            shutil.rmtree(self._tmp)
        self._tmp.mkdir(parents=True)
        self._raw: dict[str, BinaryIO] = {
            name: (self._tmp / f"{name}.bin").open("wb") for name in _TOKEN_ARRAYS
        }
        self._forms: dict[str, int] = {}
        self._body_offsets: list[int] = [0]
        self._title_offsets: list[int] = [0]
        self._docs: dict[str, list[Any]] = {name: [] for name in DOCS_SCHEMA.names}
        self._texts_sink = pa.OSFile(str(self._tmp / "texts.arrow"), "wb")
        self._texts_writer = pa.ipc.new_file(self._texts_sink, TEXTS_SCHEMA)
        self._pending_titles: list[str | None] = []
        self._pending_bodies: list[str] = []
        self._pending_bytes = 0
        self._views: dict[str, Lemmatizer] = {}
        self._finished = False

    # -- documents --------------------------------------------------------------------------

    @property
    def n_docs(self) -> int:
        return len(self._body_offsets) - 1

    def add(self, row: Row) -> int:
        """Tokenize and append one document (a ``DOC_SCHEMA`` row); returns its ``doc_idx``."""
        self._check_open()
        raw_title = row.get("title")
        title = None if raw_title is None else normalize(raw_title)
        body = normalize(row["body"])
        n_title = self._append_tokens("title", title or "")
        n_body = self._append_tokens("body", body)
        doc_idx = self.n_docs
        self._title_offsets.append(self._title_offsets[-1] + n_title)
        self._body_offsets.append(self._body_offsets[-1] + n_body)
        for name in DOCS_SCHEMA.names:
            if name == "doc_idx":
                value: Any = doc_idx
            elif name == "n_body_tokens":
                value = n_body
            elif name == "n_title_tokens":
                value = n_title
            else:
                value = row.get(name)
            self._docs[name].append(value)
        self._pending_titles.append(title)
        self._pending_bodies.append(body)
        self._pending_bytes += len(body) + len(title or "")
        if (
            len(self._pending_bodies) >= _TEXT_BATCH_ROWS
            or self._pending_bytes >= _TEXT_BATCH_BYTES
        ):
            self._flush_texts()
        return doc_idx

    def _append_tokens(self, part: str, text: str) -> int:
        tokens = tokenize(text, self.tokenizer_hyphen)
        n = len(tokens)
        if len(text) > _MAX_SPAN:
            raise ValueError("document text too long for uint32 spans")
        forms = np.empty(n, dtype="<u4")
        spans = np.empty((n, 2), dtype="<u4")
        for i, token in enumerate(tokens):
            low = lower(token.text)
            form_id = self._forms.get(low)
            if form_id is None:
                form_id = self._forms[low] = len(self._forms)
            forms[i] = form_id
            spans[i, 0] = token.start
            spans[i, 1] = token.end
        flags = np.asarray(token_flags(text, tokens), dtype="u1")
        self._raw[f"{part}_forms"].write(forms.tobytes())
        self._raw[f"{part}_spans"].write(spans.tobytes())
        self._raw[f"{part}_flags"].write(flags.tobytes())
        if part == "body":
            case = np.asarray([token_case(t.text) for t in tokens], dtype="u1")
            self._raw["body_case"].write(case.tobytes())
        return n

    def _flush_texts(self) -> None:
        if not self._pending_bodies:
            return
        batch = pa.RecordBatch.from_arrays(
            [
                pa.array(self._pending_titles, type=pa.string()),
                pa.array(self._pending_bodies, type=pa.string()),
            ],
            schema=TEXTS_SCHEMA,
        )
        self._texts_writer.write_batch(batch)
        self._pending_titles = []
        self._pending_bodies = []
        self._pending_bytes = 0

    # -- lemma views ------------------------------------------------------------------------

    def add_lemma_view(self, lemmatizer: Lemmatizer) -> str:
        """Register a form-to-lemma view; returns its key (first 16 hex of the content hash).

        The view is computed over every form of the build when ``finish`` runs, so documents may
        still be added afterwards. The lemma of a form is ``lemmatizer.lemma(clean_guess(form))``,
        as in the engine.
        """
        self._check_open()
        key = lemmatizer.content_hash()[:16]
        self._views.setdefault(key, lemmatizer)
        return key

    # -- output -----------------------------------------------------------------------------

    def finish(self, manifest: Mapping[str, Any]) -> None:
        """Write every file and rename ``<out_dir>.tmp`` to ``out_dir``.

        ``manifest`` must contain ``build_id``; the writer adds ``format_version``,
        ``tokenizer_hyphen``, ``n_docs``, ``n_body_tokens``, ``n_title_tokens``, ``n_forms`` and
        ``lemma_views`` (these keys override values of the same name).
        """
        self._check_open()
        if "build_id" not in manifest:
            raise ValueError("the corpus manifest needs a 'build_id'")
        self._flush_texts()
        self._texts_writer.close()
        self._texts_sink.close()
        for handle in self._raw.values():
            handle.close()
        tmp = self._tmp
        for name, (dtype, columns) in _TOKEN_ARRAYS.items():
            raw = tmp / f"{name}.bin"
            _write_npy_from_raw(raw, tmp / f"{name}.npy", dtype, columns)
            raw.unlink()
        np.save(tmp / "body_offsets.npy", np.asarray(self._body_offsets, dtype="<u8"))
        np.save(tmp / "title_offsets.npy", np.asarray(self._title_offsets, dtype="<u8"))

        forms = list(self._forms)  # insertion order = form_id
        write_parquet(self._forms_table(forms), tmp / "forms.parquet")
        write_parquet(pa.table(self._docs, schema=DOCS_SCHEMA), tmp / "docs.parquet")

        views: dict[str, dict[str, str]] = {}
        for key in sorted(self._views):
            lemmatizer = self._views[key]
            self._write_view(tmp / "lemma_views" / key, lemmatizer, forms)
            views[key] = {"name": lemmatizer.name, "content_hash": lemmatizer.content_hash()}

        full = dict(manifest)
        full.update(
            format_version=FORMAT_VERSION,
            tokenizer_hyphen=self.tokenizer_hyphen,
            n_docs=self.n_docs,
            n_body_tokens=self._body_offsets[-1],
            n_title_tokens=self._title_offsets[-1],
            n_forms=len(forms),
            lemma_views=views,
        )
        (tmp / "manifest.json").write_bytes((canonical_json(full) + "\n").encode("utf-8"))
        os.replace(tmp, self.out_dir)
        self._finished = True

    def abort(self) -> None:
        """Discard everything written so far."""
        if self._finished:
            return
        self._finished = True
        for handle in self._raw.values():
            handle.close()
        with contextlib.suppress(OSError, pa.ArrowException):
            self._texts_writer.close()
        self._texts_sink.close()
        shutil.rmtree(self._tmp, ignore_errors=True)

    def __enter__(self) -> CorpusWriter:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        if exc_type is not None:
            self.abort()

    @staticmethod
    def _forms_table(forms: list[str]) -> pa.Table:
        n_chars = np.fromiter((len(f) for f in forms), dtype=np.int64, count=len(forms))
        if len(forms) and int(np.max(n_chars)) > np.iinfo(np.uint16).max:
            raise ValueError("a form is longer than 65535 characters")
        all_digits = [f.isascii() and f.isdigit() for f in forms]
        has_digit = [not _DIGITS.isdisjoint(f) for f in forms]
        return pa.table(
            {
                "form_id": pa.array(np.arange(len(forms), dtype=np.uint32)),
                "form": pa.array(forms, type=pa.string()),
                "n_chars": pa.array(n_chars.astype(np.uint16)),
                "all_digits": pa.array(all_digits, type=pa.bool_()),
                "has_digit": pa.array(has_digit, type=pa.bool_()),
            },
            schema=FORMS_SCHEMA,
        )

    @staticmethod
    def _write_view(directory: Path, lemmatizer: Lemmatizer, forms: list[str]) -> None:
        directory.mkdir(parents=True)
        lemma_ids: dict[str, int] = {}
        form2lemma: npt.NDArray[np.uint32] = np.empty(len(forms), dtype="<u4")
        for form_id, form in enumerate(forms):
            lemma = lemmatizer.lemma(clean_guess(form))
            lemma_id = lemma_ids.get(lemma)
            if lemma_id is None:
                lemma_id = lemma_ids[lemma] = len(lemma_ids)
            form2lemma[form_id] = lemma_id
        np.save(directory / "form2lemma.npy", form2lemma)
        lemmas = pa.table(
            {
                "lemma_id": pa.array(np.arange(len(lemma_ids), dtype=np.uint32)),
                "lemma": pa.array(list(lemma_ids), type=pa.string()),
            },
            schema=LEMMAS_SCHEMA,
        )
        write_parquet(lemmas, directory / "lemmas.parquet")

    def _check_open(self) -> None:
        if self._finished:
            raise RuntimeError("the corpus writer is already finished")
