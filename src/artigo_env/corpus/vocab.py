"""Construction of the action vocabulary and its mapping onto compiled corpora (SDD 09 §2).

``build_vocab`` derives V from the training partition of a build plus an open lexicon, never from
an evaluation document (REQ-ENV-003), and writes a versioned, byte-deterministic artifact
(REQ-ENV-004) read by ``artigo_env.core.vocab.Vocabulary``. ``map_store`` turns the form ids of
any build into observation ids (decision 3 of the F4 plan) and ``title_coverage`` measures how
many articles of a split have every title lemma in V (REQ-ENV-005).

Counting is vectorized over the memory-mapped token arrays (``np.bincount`` for token
frequencies, ``np.unique`` over (document, key) codes for document frequencies); every distinct
form is lemmatized at most once (build forms come from the build's lemma view).
"""

from __future__ import annotations

import os
import shutil
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import regex

from artigo_env.core.config import ConfigBase, canonical_json, sha256_hex
from artigo_env.core.errors import ArtigoEnvError, IncompatibleRulesetError
from artigo_env.core.vocab import (
    ALL_LENGTHS,
    ARRAY_FILES,
    FORMS_FILE,
    LEMMAS_FILE,
    MANIFEST_FILE,
    MAPS_DIR,
    NUMERIC_PREFIX,
    STOP_FILE,
    VOCAB_FORMAT_VERSION,
    Vocabulary,
)
from artigo_env.corpus.store.reader import CorpusStore, LemmaView
from artigo_env.corpus.store.writer import write_parquet
from artigo_env.engine.rulesets.components import NUMBER_RULES
from artigo_env.engine.rulesets.ruleset import Ruleset
from artigo_env.engine.types import FLAG_INSIDE_PARENTHESES
from artigo_env.text.guess import clean_guess
from artigo_env.text.lemmatizers.registry import get_lemmatizer
from artigo_env.text.normalize import lower, normalize
from artigo_env.text.stopwords import load_stopwords

_DIGITS = regex.compile(r"[0-9]+")
_DIGITS_PATTERN = "^[0-9]+$"
_DOC_BATCH = 2048
MAP_FORMAT_VERSION = 1

LEMMAS_TABLE_SCHEMA = pa.schema(
    [
        pa.field("action_id", pa.int32(), nullable=False),
        pa.field("key", pa.string(), nullable=False),
        pa.field("is_numeric", pa.bool_(), nullable=False),
        pa.field("freq", pa.int64(), nullable=False),
        pa.field("doc_freq", pa.int64(), nullable=False),
    ]
)
STOP_TABLE_SCHEMA = pa.schema(
    [
        pa.field("stop_id", pa.int32(), nullable=False),
        pa.field("form", pa.string(), nullable=False),
    ]
)
FORMS_TABLE_SCHEMA = pa.schema(
    [
        pa.field("form_id", pa.int32(), nullable=False),
        pa.field("form", pa.string(), nullable=False),
        pa.field("freq", pa.int64(), nullable=False),
    ]
)


@dataclass(frozen=True)
class VocabConfig(ConfigBase):
    """Construction parameters of V (SDD 09 §2.2)."""

    min_doc_freq: int = 2
    max_size: int = 300_000
    numeric_max: int = 2100
    numeric_min_doc_freq: int = 5
    numeric_max_digits: int = 6
    max_forms: int = 2_000_000

    def __post_init__(self) -> None:
        for name in (
            "min_doc_freq",
            "max_size",
            "numeric_min_doc_freq",
            "numeric_max_digits",
            "max_forms",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")
        if (
            isinstance(self.numeric_max, bool)
            or not isinstance(self.numeric_max, int)
            or self.numeric_max < -1
        ):
            raise ValueError(f"numeric_max must be an integer >= -1, got {self.numeric_max!r}")


@dataclass(frozen=True)
class StoreVocabMap:
    """Observation ids of one build under one vocabulary (decision 3 of the F4 plan).

    ``lemma_obs[form_id]``: action id of the form's lemma, ``|V| + i`` for the i-th stopword,
    ``-1`` when the form is visible but outside V ∪ S. ``form_obs[form_id]``: id in F or ``-1``.
    ``action_lemma[action_id]``: lemma id of the key in the build's lemma view, ``-1`` when the
    build has no such lemma or the action is numeric.
    """

    build_id: str
    vocab_id: str
    lemma_obs: npt.NDArray[np.int32]
    form_obs: npt.NDArray[np.int32]
    action_lemma: npt.NDArray[np.int64]


@dataclass(frozen=True)
class CoverageReport:
    """Title coverage of a split (SDD 09 §2.3, REQ-ENV-005).

    An article is covered when every required title token (content token; outside parentheses
    when the ruleset ignores them) has its lemma, or numeric key, in V. ``uncovered`` lists the
    document indices of the other articles; ``missing`` the absent keys with the number of
    articles each one blocks, most frequent first.
    """

    build_id: str
    vocab_id: str
    split: str
    n_articles: int
    n_covered: int
    fraction: float
    uncovered: tuple[int, ...] = ()
    missing: tuple[tuple[str, int], ...] = field(default=())

    def to_dict(self) -> dict[str, Any]:
        return {
            "build_id": self.build_id,
            "vocab_id": self.vocab_id,
            "split": self.split,
            "n_articles": self.n_articles,
            "n_covered": self.n_covered,
            "fraction": self.fraction,
            "uncovered": list(self.uncovered),
            "missing": [[key, count] for key, count in self.missing],
        }


# -- helpers -----------------------------------------------------------------------------------


def _numbers_enabled(ruleset: Ruleset) -> bool:
    return bool(NUMBER_RULES.get(ruleset.numbers).enabled)


def _build_forms(store: CorpusStore) -> pa.StringArray:
    table = pq.read_table(store.path / "forms.parquet", columns=["form"], memory_map=False)
    forms: pa.StringArray = table.column("form").combine_chunks()
    if len(forms) != store.n_forms:
        raise ArtigoEnvError(f"inconsistent forms table in {store.path}")
    return forms


def _view_lemmas(store: CorpusStore, view: LemmaView) -> pa.StringArray:
    path = store.path / "lemma_views" / view.key / "lemmas.parquet"
    table = pq.read_table(path, columns=["lemma"], memory_map=False)
    lemmas: pa.StringArray = table.column("lemma").combine_chunks()
    return lemmas


def _form_classes(
    store: CorpusStore, ruleset: Ruleset, forms: pa.StringArray
) -> tuple[npt.NDArray[np.bool_], npt.NDArray[np.bool_]]:
    """``(numeric, stop)`` per build form, classified in the engine's order (numeric first)."""
    if _numbers_enabled(ruleset):
        hits = pc.match_substring_regex(forms, _DIGITS_PATTERN)
        numeric = np.asarray(hits.to_numpy(zero_copy_only=False), dtype=np.bool_)
    else:
        numeric = np.zeros(len(forms), dtype=np.bool_)
    stop = np.asarray(store.stop_mask(ruleset.stopwords), dtype=np.bool_) & ~numeric
    return numeric, stop


def _to_index(indices: pa.Array) -> npt.NDArray[np.int64]:
    """``pc.index_in`` result as int64 with ``-1`` for absent values."""
    filled = pc.fill_null(indices, -1)
    return np.asarray(filled.to_numpy(zero_copy_only=False), dtype=np.int64)


def _training_docs(store: CorpusStore) -> tuple[str, npt.NDArray[np.int64]]:
    docs = store.split_indices("train")
    if len(docs):
        return "train", docs
    docs = store.split_indices("demo")
    if len(docs):
        return "demo", docs
    raise ArtigoEnvError(
        f"the corpus build {store.build_id!r} has no 'train' (or 'demo') partition"
    )


def _count(
    store: CorpusStore,
    docs: npt.NDArray[np.int64],
    form2cand: npt.NDArray[np.int64],
    n_cand: int,
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """Token frequency per form and document frequency per candidate over ``docs``.

    A document is its title plus its body. Documents are processed in batches: the candidate of
    each token is encoded with its position in the batch, deduplicated with ``np.unique`` and
    counted with ``np.bincount``.
    """
    n_forms = store.n_forms
    tokens_per_form = np.zeros(n_forms, dtype=np.int64)
    doc_freq = np.zeros(n_cand, dtype=np.int64)
    for start in range(0, len(docs), _DOC_BATCH):
        batch = docs[start : start + _DOC_BATCH]
        parts: list[npt.NDArray[np.uint32]] = []
        sizes = np.empty(len(batch), dtype=np.int64)
        for j, doc_idx in enumerate(batch.tolist()):
            title = store.title_forms(doc_idx)
            body = store.body_forms(doc_idx)
            parts.append(title)
            parts.append(body)
            sizes[j] = len(title) + len(body)
        if not parts:
            continue
        tokens = np.concatenate(parts).astype(np.int64)
        if not len(tokens):
            continue
        tokens_per_form += np.bincount(tokens, minlength=n_forms)
        cand = form2cand[tokens]
        local = np.repeat(np.arange(len(batch), dtype=np.int64), sizes)
        keep = cand >= 0
        codes = np.unique(local[keep] * n_cand + cand[keep])
        doc_freq += np.bincount(codes % n_cand, minlength=n_cand)
    return tokens_per_form, doc_freq


def _normalize_lexicon(lexicon_forms: Iterable[str]) -> list[str]:
    forms = {lower(normalize(raw.strip())) for raw in lexicon_forms}
    forms.discard("")
    return sorted(forms)


def _length_bits(n_chars: npt.NDArray[np.int64]) -> npt.NDArray[np.uint64]:
    bits = np.clip(n_chars, 1, 64) - 1
    return np.left_shift(np.uint64(1), bits.astype(np.uint64))


def _write_npy(path: Path, array: npt.NDArray[Any]) -> None:
    np.save(path, np.ascontiguousarray(array), allow_pickle=False)


def _sorted_order(keys: list[str], freq: npt.NDArray[np.int64]) -> npt.NDArray[np.int64]:
    """Indices ordering by frequency descending, then key ascending (code-point order)."""
    table = pa.table({"key": pa.array(keys, type=pa.string()), "freq": pa.array(freq)})
    order = pc.sort_indices(table, sort_keys=[("freq", "descending"), ("key", "ascending")])
    return np.asarray(order.to_numpy(), dtype=np.int64)


# -- construction ------------------------------------------------------------------------------


def build_vocab(
    store: CorpusStore,
    ruleset: Ruleset,
    config: VocabConfig,
    out_root: Path,
    *,
    lexicon_forms: Iterable[str] = (),
) -> Path:
    """Build V for (training build, ruleset) and return ``<out_root>/<vocab_id>``.

    V is the union of (1) content lemmas of the ``train`` partition (``demo`` when there is no
    ``train``, then with ``min_doc_freq = 1``) with document frequency >= ``min_doc_freq``;
    (2) lemmas of the non-stopword lexicon forms; (3) ``n:0`` … ``n:<numeric_max>`` and the
    digit-only training tokens of at most ``numeric_max_digits`` digits with document frequency
    >= ``numeric_min_doc_freq`` (only when the ruleset's number rule is enabled; otherwise digit
    tokens are lexical). Order: training token frequency descending, then key in code-point
    order; cut at ``max_size`` keeping every ``n:0`` … ``n:<numeric_max>``. F (training forms ∪
    lexicon forms) uses the same order and is cut at ``max_forms``.

    If the directory already exists (same inputs, same id) it is returned unchanged.
    """
    ruleset.check_build(store.tokenizer_hyphen)
    numbers = _numbers_enabled(ruleset)
    view = store.lemma_view(ruleset.lemmatizer)
    lemmatizer = get_lemmatizer(ruleset.lemmatizer)
    stopset = load_stopwords(ruleset.stopwords)

    lexicon = _normalize_lexicon(lexicon_forms)
    lexicon_hash = sha256_hex("\n".join(lexicon))
    ruleset_hash = ruleset.content_hash()
    identity = {
        "build_id": store.build_id,
        "ruleset_hash": ruleset_hash,
        "config_hash": config.content_hash(),
        "lexicon_hash": lexicon_hash,
    }
    vocab_id = sha256_hex(canonical_json(identity))[:16]
    out_root = Path(out_root)
    out_dir = out_root / vocab_id
    if (out_dir / MANIFEST_FILE).is_file():
        return out_dir

    partition, docs = _training_docs(store)
    min_doc_freq = config.min_doc_freq if partition == "train" else 1

    # -- classify build forms and count the training partition ------------------------------
    forms = _build_forms(store)
    lemmas = _view_lemmas(store, view)
    n_lemmas = len(lemmas)
    form2lemma = np.asarray(view.form2lemma, dtype=np.int64)
    numeric, stop = _form_classes(store, ruleset, forms)
    content = ~(numeric | stop)
    numeric_ids = np.flatnonzero(numeric)
    form2cand = np.full(store.n_forms, -1, dtype=np.int64)
    form2cand[content] = form2lemma[content]
    form2cand[numeric_ids] = n_lemmas + np.arange(len(numeric_ids), dtype=np.int64)
    n_cand = n_lemmas + len(numeric_ids)
    tokens_per_form, cand_doc_freq = _count(store, docs, form2cand, n_cand)

    seen = tokens_per_form > 0
    train_content = np.flatnonzero(content & seen)
    lemma_freq = np.bincount(
        form2lemma[train_content], weights=tokens_per_form[train_content], minlength=n_lemmas
    )
    lemma_freq_i = np.rint(lemma_freq).astype(np.int64)
    lemma_doc_freq = cand_doc_freq[:n_lemmas]
    n_chars = np.asarray(
        pc.utf8_length(forms.take(pa.array(train_content))).to_numpy(zero_copy_only=False),
        dtype=np.int64,
    )
    lemma_mask = np.zeros(n_lemmas, dtype=np.uint64)
    np.bitwise_or.at(lemma_mask, form2lemma[train_content], _length_bits(n_chars))

    # -- candidate keys -------------------------------------------------------------------
    keys: list[str] = []
    key_freq: list[int] = []
    key_doc_freq: list[int] = []
    key_mask: list[int] = []
    key_numeric: list[bool] = []
    index: dict[str, int] = {}

    def add(key: str, freq: int, doc_freq: int, mask: int, is_numeric: bool) -> int:
        i = index.get(key)
        if i is None:
            i = index[key] = len(keys)
            keys.append(key)
            key_freq.append(freq)
            key_doc_freq.append(doc_freq)
            key_mask.append(mask)
            key_numeric.append(is_numeric)
        return i

    train_lemma_ids = np.flatnonzero(lemma_freq_i > 0)
    train_texts: list[str] = lemmas.take(pa.array(train_lemma_ids)).to_pylist()
    train_stats: dict[str, int] = {}
    for lemma_id, text in zip(train_lemma_ids.tolist(), train_texts, strict=True):
        if not text:
            continue
        train_stats[text] = lemma_id
        if int(lemma_doc_freq[lemma_id]) >= min_doc_freq:
            add(
                text,
                int(lemma_freq_i[lemma_id]),
                int(lemma_doc_freq[lemma_id]),
                int(lemma_mask[lemma_id]),
                False,
            )
    n_from_corpus = len(keys)

    # Rule (2): the open lexicon. Forms already in the build reuse the build's lemma view.
    lexicon_array = pa.array(lexicon, type=pa.string())
    in_build = _to_index(pc.index_in(lexicon_array, value_set=forms))
    lemma_texts: list[str] | None = None
    for form, build_form in zip(lexicon, in_build.tolist(), strict=True):
        if numbers and _DIGITS.fullmatch(form) is not None:
            continue
        if form in stopset:
            continue
        if build_form >= 0:
            if lemma_texts is None:
                lemma_texts = lemmas.to_pylist()
            lemma = lemma_texts[int(form2lemma[build_form])]
        else:
            lemma = lemmatizer.lemma(clean_guess(form))
        if not lemma:
            continue
        bit = 1 << (min(len(form), 64) - 1)
        i = index.get(lemma)
        if i is None:
            train_id = train_stats.get(lemma)
            if train_id is None:
                add(lemma, 0, 0, bit, False)
            else:
                add(
                    lemma,
                    int(lemma_freq_i[train_id]),
                    int(lemma_doc_freq[train_id]),
                    int(lemma_mask[train_id]) | bit,
                    False,
                )
        else:
            key_mask[i] |= bit
    n_from_lexicon = len(keys) - n_from_corpus

    # Rule (3): numeric lexemes.
    forced: set[str] = set()
    if numbers:
        digit_texts: list[str] = forms.take(pa.array(numeric_ids)).to_pylist()
        numeric_freq = tokens_per_form[numeric_ids]
        numeric_doc_freq = cand_doc_freq[n_lemmas:]
        digit_stats = {
            text: (int(numeric_freq[j]), int(numeric_doc_freq[j]))
            for j, text in enumerate(digit_texts)
            if numeric_doc_freq[j] > 0
        }
        for value in range(config.numeric_max + 1):
            digits = str(value)
            freq, doc_freq = digit_stats.get(digits, (0, 0))
            key = NUMERIC_PREFIX + digits
            add(key, freq, doc_freq, int(ALL_LENGTHS), True)
            forced.add(key)
        for digits, (freq, doc_freq) in digit_stats.items():
            if doc_freq >= config.numeric_min_doc_freq and len(digits) <= config.numeric_max_digits:
                add(NUMERIC_PREFIX + digits, freq, doc_freq, int(ALL_LENGTHS), True)
    if len(forced) > config.max_size:
        raise ValueError(
            f"max_size={config.max_size} is smaller than the {len(forced)} numeric lexemes "
            f"n:0..n:{config.numeric_max} that are never cut"
        )
    n_candidates = len(keys)

    # -- order and cut --------------------------------------------------------------------
    freq_arr = np.asarray(key_freq, dtype=np.int64)
    order = _sorted_order(keys, freq_arr)
    is_forced = np.fromiter((key in forced for key in keys), dtype=np.bool_, count=len(keys))
    free_slots = config.max_size - len(forced)
    ordered_forced = is_forced[order]
    free_rank = np.cumsum(~ordered_forced)
    selected = order[ordered_forced | ((~ordered_forced) & (free_rank <= free_slots))]

    v_keys = [keys[i] for i in selected.tolist()]
    v_freq = freq_arr[selected]
    v_doc_freq = np.asarray(key_doc_freq, dtype=np.int64)[selected]
    v_numeric = np.asarray(key_numeric, dtype=np.bool_)[selected]
    v_mask = np.asarray(key_mask, dtype=np.uint64)[selected]
    size = len(v_keys)

    # -- form vocabulary F ----------------------------------------------------------------
    train_forms_ids = np.flatnonzero(seen)
    train_forms = forms.take(pa.array(train_forms_ids))
    extra = pc.invert(pc.is_in(lexicon_array, value_set=train_forms))
    lexicon_only = pc.filter(lexicon_array, extra)
    f_forms = pa.concat_arrays([train_forms.cast(pa.string()), lexicon_only.cast(pa.string())])
    f_freq = np.concatenate(
        [tokens_per_form[train_forms_ids], np.zeros(len(lexicon_only), dtype=np.int64)]
    )
    f_table = pa.table({"form": f_forms, "freq": pa.array(f_freq)})
    f_order = pc.sort_indices(f_table, sort_keys=[("freq", "descending"), ("form", "ascending")])
    f_table = f_table.take(f_order).slice(0, config.max_forms)
    n_forms = f_table.num_rows

    stopwords = sorted(stopset.words)

    # -- write ----------------------------------------------------------------------------
    out_root.mkdir(parents=True, exist_ok=True)
    tmp = out_root / f"{vocab_id}.tmp"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir()
    try:
        write_parquet(
            pa.table(
                {
                    "action_id": pa.array(np.arange(size, dtype=np.int32)),
                    "key": pa.array(v_keys, type=pa.string()),
                    "is_numeric": pa.array(v_numeric),
                    "freq": pa.array(v_freq),
                    "doc_freq": pa.array(v_doc_freq),
                },
                schema=LEMMAS_TABLE_SCHEMA,
            ),
            tmp / LEMMAS_FILE,
        )
        write_parquet(
            pa.table(
                {
                    "stop_id": pa.array(np.arange(len(stopwords), dtype=np.int32)),
                    "form": pa.array(stopwords, type=pa.string()),
                },
                schema=STOP_TABLE_SCHEMA,
            ),
            tmp / STOP_FILE,
        )
        write_parquet(
            pa.table(
                {
                    "form_id": pa.array(np.arange(n_forms, dtype=np.int32)),
                    "form": f_table.column("form").combine_chunks(),
                    "freq": f_table.column("freq").combine_chunks(),
                },
                schema=FORMS_TABLE_SCHEMA,
            ),
            tmp / FORMS_FILE,
        )
        _write_npy(tmp / ARRAY_FILES["freq"], v_freq.astype("<i8"))
        _write_npy(tmp / ARRAY_FILES["doc_freq"], v_doc_freq.astype("<i8"))
        _write_npy(tmp / ARRAY_FILES["is_numeric"], v_numeric.astype(np.bool_))
        _write_npy(tmp / ARRAY_FILES["length_mask"], v_mask.astype("<u8"))
        manifest = {
            "format_version": VOCAB_FORMAT_VERSION,
            "vocab_id": vocab_id,
            "identity": identity,
            "build_id": store.build_id,
            "ruleset_name": ruleset.name,
            "ruleset": ruleset.to_dict(),
            "ruleset_hash": ruleset_hash,
            "lemmatizer": ruleset.lemmatizer,
            "lemma_view": view.key,
            "numbers_enabled": numbers,
            "config": config.to_dict(),
            "config_hash": config.content_hash(),
            "lexicon_hash": lexicon_hash,
            "n_lexicon_forms": len(lexicon),
            "partition": partition,
            "n_train_docs": len(docs),
            "min_doc_freq": min_doc_freq,
            "size": size,
            "n_stop": len(stopwords),
            "n_forms": n_forms,
            "n_numeric": int(v_numeric.sum()),
            "n_candidates": n_candidates,
            "n_candidates_corpus": n_from_corpus,
            "n_candidates_lexicon_only": n_from_lexicon,
            "n_form_candidates": len(f_forms),
        }
        (tmp / MANIFEST_FILE).write_bytes((canonical_json(manifest) + "\n").encode("utf-8"))
        os.replace(tmp, out_dir)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return out_dir


# -- mapping a build onto a vocabulary ---------------------------------------------------------


def _check_ruleset(vocab: Vocabulary, ruleset: Ruleset) -> None:
    expected = vocab.manifest.get("ruleset_hash")
    if expected != ruleset.content_hash():
        raise IncompatibleRulesetError(
            f"vocabulary {vocab.vocab_id} was built for ruleset "
            f"{vocab.manifest.get('ruleset_name')!r} ({str(expected)[:16]}), not for "
            f"{ruleset.name!r} ({ruleset.content_hash()[:16]})"
        )


def _map_manifest(vocab: Vocabulary, store: CorpusStore, ruleset: Ruleset) -> dict[str, Any]:
    return {
        "format_version": MAP_FORMAT_VERSION,
        "vocab_id": vocab.vocab_id,
        "build_id": store.build_id,
        "ruleset_hash": ruleset.content_hash(),
        "n_build_forms": store.n_forms,
        "size": vocab.size,
    }


def _load_map(directory: Path, expected: dict[str, Any]) -> StoreVocabMap | None:
    manifest_path = directory / MANIFEST_FILE
    if not manifest_path.is_file():
        return None
    if manifest_path.read_text(encoding="utf-8").strip() != canonical_json(expected):
        return None
    try:
        lemma_obs = np.load(directory / "lemma_obs.npy")
        form_obs = np.load(directory / "form_obs.npy")
        action_lemma = np.load(directory / "action_lemma.npy")
    except (OSError, ValueError):
        return None
    if (
        lemma_obs.shape != (expected["n_build_forms"],)
        or form_obs.shape != (expected["n_build_forms"],)
        or action_lemma.shape != (expected["size"],)
    ):
        return None
    return StoreVocabMap(
        expected["build_id"],
        expected["vocab_id"],
        lemma_obs.astype(np.int32, copy=False),
        form_obs.astype(np.int32, copy=False),
        action_lemma.astype(np.int64, copy=False),
    )


def map_store(
    vocab: Vocabulary, store: CorpusStore, ruleset: Ruleset, *, cache: bool = True
) -> StoreVocabMap:
    """Observation ids of every form of ``store`` and build lemma ids of every action.

    With ``cache=True`` the arrays are read from, or written to, ``<vocab>/maps/<build_id>/``.
    The arrays are loaded whole (no file stays open). V itself never changes (REQ-ENV-003).
    """
    _check_ruleset(vocab, ruleset)
    ruleset.check_build(store.tokenizer_hyphen)
    expected = _map_manifest(vocab, store, ruleset)
    directory = vocab.path / MAPS_DIR / store.build_id
    if cache:
        loaded = _load_map(directory, expected)
        if loaded is not None:
            return loaded

    view = store.lemma_view(ruleset.lemmatizer)
    forms = _build_forms(store)
    lemmas = _view_lemmas(store, view)
    form2lemma = np.asarray(view.form2lemma, dtype=np.int64)
    numeric, stop = _form_classes(store, ruleset, forms)
    keys = pa.array(vocab.keys(), type=pa.string())

    lemma2action = _to_index(pc.index_in(lemmas, value_set=keys))
    lemma_obs = lemma2action[form2lemma]
    if numeric.any():
        numeric_ids = np.flatnonzero(numeric)
        numeric_keys = pc.binary_join_element_wise(
            pa.scalar(NUMERIC_PREFIX), forms.take(pa.array(numeric_ids)), ""
        )
        lemma_obs[numeric_ids] = _to_index(pc.index_in(numeric_keys, value_set=keys))
    if stop.any():
        stop_ids = np.flatnonzero(stop)
        stopwords = pa.array(vocab.stopwords(), type=pa.string())
        index = _to_index(pc.index_in(forms.take(pa.array(stop_ids)), value_set=stopwords))
        lemma_obs[stop_ids] = np.where(index >= 0, vocab.size + index, -1)
    form_obs = _to_index(pc.index_in(forms, value_set=pa.array(vocab.forms(), type=pa.string())))
    action_lemma = _to_index(pc.index_in(keys, value_set=lemmas))
    action_lemma[np.asarray(vocab.is_numeric, dtype=np.bool_)] = -1

    result = StoreVocabMap(
        store.build_id,
        vocab.vocab_id,
        lemma_obs.astype(np.int32),
        form_obs.astype(np.int32),
        action_lemma.astype(np.int64),
    )
    if cache:
        _write_map(directory, expected, result)
    return result


def _write_map(directory: Path, manifest: dict[str, Any], result: StoreVocabMap) -> None:
    directory.parent.mkdir(parents=True, exist_ok=True)
    tmp = directory.with_name(directory.name + f".tmp{os.getpid()}")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir()
    try:
        _write_npy(tmp / "lemma_obs.npy", result.lemma_obs.astype("<i4"))
        _write_npy(tmp / "form_obs.npy", result.form_obs.astype("<i4"))
        _write_npy(tmp / "action_lemma.npy", result.action_lemma.astype("<i8"))
        (tmp / MANIFEST_FILE).write_bytes((canonical_json(manifest) + "\n").encode("utf-8"))
        if directory.exists():
            shutil.rmtree(directory)  # stale map (different manifest)
        os.replace(tmp, directory)
    except OSError:
        # Another process may have written the same map concurrently; the result stands.
        shutil.rmtree(tmp, ignore_errors=True)


# -- title coverage ----------------------------------------------------------------------------


def title_coverage(
    vocab: Vocabulary, store: CorpusStore, ruleset: Ruleset, split: str
) -> CoverageReport:
    """Fraction of the articles of ``split`` whose required title lemmas are all in V."""
    mapping = map_store(vocab, store, ruleset)
    docs = store.split_indices(split)
    ignore_parentheses = ruleset.title_parenthetical == "ignored"
    view = store.lemma_view(ruleset.lemmatizer)
    numbers = _numbers_enabled(ruleset)
    uncovered: list[int] = []
    missing: Counter[str] = Counter()
    for doc_idx in docs.tolist():
        title = store.title_forms(doc_idx).astype(np.int64)
        obs = mapping.lemma_obs[title]
        required = obs < vocab.size  # stopwords have ids >= |V|
        if ignore_parentheses and len(title):
            a, b = store._bounds("title", doc_idx)
            flags = np.asarray(store._array("title_flags")[a:b], dtype=np.uint8)
            required &= (flags & FLAG_INSIDE_PARENTHESES) == 0
        absent = title[required & (obs < 0)]
        if not len(absent):
            continue
        uncovered.append(int(doc_idx))
        keys: set[str] = set()
        for form_id in absent.tolist():
            form = store.form(form_id)
            if numbers and _DIGITS.fullmatch(form) is not None:
                keys.add(NUMERIC_PREFIX + form)
            else:
                keys.add(view.lemma_text(int(view.form2lemma[form_id])))
        missing.update(keys)
    n_articles = len(docs)
    n_covered = n_articles - len(uncovered)
    return CoverageReport(
        build_id=store.build_id,
        vocab_id=vocab.vocab_id,
        split=split,
        n_articles=n_articles,
        n_covered=n_covered,
        fraction=n_covered / n_articles if n_articles else 0.0,
        uncovered=tuple(uncovered),
        missing=tuple(sorted(missing.items(), key=lambda item: (-item[1], item[0]))),
    )


def with_max_size(config: VocabConfig, max_size: int | None) -> VocabConfig:
    """``config`` with ``max_size`` replaced when given (CLI helper)."""
    return config if max_size is None else replace(config, max_size=max_size)
