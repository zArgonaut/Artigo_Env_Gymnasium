"""Construction of the minimal knowledge store (SDD 10 §4.1, DEC-074; WP4.4).

``build_knowledge`` counts, over the training partition of a build (``train``, or ``demo`` when
the build has no ``train``), how often each action of a vocabulary V occurs: token frequency and
document frequency over title plus body, and the number of documents whose title contains it.
Each build form is mapped to its action id once (``map_store``: the action of its lemma, or of
``n:<digits>`` for a digit-only form); counting is vectorized with ``np.bincount`` over the
memory-mapped token arrays and ``np.unique`` over (document, action) codes.

The artifact is byte-deterministic (REQ-KNW-002): ``knowledge_id`` is the hash of (build,
ruleset, vocabulary, source configuration, regime, builder version) and every file is written in
a fixed format. A build whose manifest lists a Wikimedia source is refused under ``X->W``
(REQ-KNW-001).
"""

from __future__ import annotations

import os
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from artigo_env.core.config import canonical_json, sha256_hex
from artigo_env.core.errors import ArtigoEnvError
from artigo_env.core.vocab import Vocabulary
from artigo_env.corpus.store.reader import CorpusStore
from artigo_env.corpus.vocab import map_store
from artigo_env.engine.rulesets.ruleset import Ruleset
from artigo_env.services.knowledge import (
    KNOWLEDGE_FORMAT_VERSION,
    LEMMA_STATS_DTYPE,
    LEMMA_STATS_FILE,
    MANIFEST_FILE,
    REGIMES,
    SOURCE_CONFIGS,
    TITLE_STATS_FILE,
    KnowledgeMismatchError,
    file_sha256,
)

BUILDER_VERSION = 1
_DOC_BATCH = 2048
WIKIMEDIA_KINDS = frozenset({"wikipedia_extracts"})


class WikimediaSourceError(ArtigoEnvError, ValueError):
    """A Wikimedia source cannot feed the knowledge store of ``X->W`` (REQ-KNW-001)."""


def is_wikimedia_source(info: Mapping[str, Any]) -> bool:
    """Whether a source entry of a build manifest is marked as Wikimedia."""
    return bool(info.get("wikimedia")) or str(info.get("kind", "")) in WIKIMEDIA_KINDS


def wikimedia_sources(manifest: Mapping[str, Any]) -> list[str]:
    """Ids of the Wikimedia-marked sources of a build (an evaluation build counts as one)."""
    found = [
        str(info.get("id", "?"))
        for info in manifest.get("sources", []) or []
        if isinstance(info, Mapping) and is_wikimedia_source(info)
    ]
    if manifest.get("kind") == "evaluation":
        found.append(f"<evaluation build {manifest.get('build_id')}>")
    return found


def _training_docs(store: CorpusStore) -> tuple[str, npt.NDArray[np.int64]]:
    for partition in ("train", "demo"):
        docs = store.split_indices(partition)
        if len(docs):
            return partition, np.asarray(docs, dtype=np.int64)
    raise ArtigoEnvError(
        f"the corpus build {store.build_id!r} has no 'train' (or 'demo') partition"
    )


def count_actions(
    store: CorpusStore, docs: npt.NDArray[np.int64], form2action: npt.NDArray[np.int64], size: int
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """``(freq, doc_freq, title_doc_freq)`` of every action over ``docs`` (title + body)."""
    freq = np.zeros(size, dtype=np.int64)
    doc_freq = np.zeros(size, dtype=np.int64)
    title_doc_freq = np.zeros(size, dtype=np.int64)

    def unique_docs(local: npt.NDArray[np.int64], action: npt.NDArray[np.int64]) -> Any:
        keep = action >= 0
        codes = np.unique(local[keep] * size + action[keep])
        return np.bincount(codes % size, minlength=size)

    for start in range(0, len(docs), _DOC_BATCH):
        batch = docs[start : start + _DOC_BATCH].tolist()
        titles = [np.asarray(store.title_forms(d), dtype=np.int64) for d in batch]
        bodies = [np.asarray(store.body_forms(d), dtype=np.int64) for d in batch]
        local_ids = np.arange(len(batch), dtype=np.int64)
        title_tokens = np.concatenate(titles) if titles else np.zeros(0, np.int64)
        title_local = np.repeat(local_ids, [len(t) for t in titles])
        body_tokens = np.concatenate(bodies) if bodies else np.zeros(0, np.int64)
        body_local = np.repeat(local_ids, [len(b) for b in bodies])
        tokens = np.concatenate([title_tokens, body_tokens])
        local = np.concatenate([title_local, body_local])
        if not len(tokens):
            continue
        action = form2action[tokens]
        hit = action >= 0
        freq += np.bincount(action[hit], minlength=size)
        doc_freq += unique_docs(local, action)
        title_doc_freq += unique_docs(title_local, form2action[title_tokens])
    return freq, doc_freq, title_doc_freq


def _write_npy(path: Path, array: npt.NDArray[Any]) -> None:
    np.save(path, np.ascontiguousarray(array), allow_pickle=False)


def knowledge_identity(
    store: CorpusStore, ruleset: Ruleset, vocab: Vocabulary, source_config: str, regime: str
) -> dict[str, Any]:
    return {
        "build_id": store.build_id,
        "ruleset_hash": ruleset.content_hash(),
        "vocab_id": vocab.vocab_id,
        "source_config": source_config,
        "regime": regime,
        "builder_version": BUILDER_VERSION,
        "format_version": KNOWLEDGE_FORMAT_VERSION,
    }


def build_knowledge(
    store: CorpusStore,
    ruleset: Ruleset,
    vocab: Vocabulary,
    out_root: Path,
    *,
    source_config: str = "X->W",
    regime: str = "open",
) -> Path:
    """Build the knowledge store of (build, ruleset, V) and return ``<out_root>/<knowledge_id>``.

    If the directory already exists (same inputs, same id) it is returned unchanged.
    """
    if source_config not in SOURCE_CONFIGS:
        raise ValueError(f"unknown source configuration {source_config!r}; known: {SOURCE_CONFIGS}")
    if regime not in REGIMES:
        raise ValueError(f"unknown regime {regime!r}; known: {REGIMES}")
    if regime != "open":
        raise NotImplementedError(f"the {regime!r} regime arrives in phase F5; use 'open'")
    if source_config == "X->W":
        marked = wikimedia_sources(store.manifest)
        if marked:
            raise WikimediaSourceError(
                f"the corpus build {store.build_id!r} contains Wikimedia-marked sources "
                f"{marked}; they cannot feed a knowledge store of 'X->W' (REQ-KNW-001)"
            )
    ruleset_hash = ruleset.content_hash()
    if vocab.manifest.get("ruleset_hash") not in (None, ruleset_hash):
        raise KnowledgeMismatchError(
            f"vocabulary {vocab.vocab_id!r} was built for another ruleset than {ruleset.name!r}"
        )

    identity = knowledge_identity(store, ruleset, vocab, source_config, regime)
    knowledge_id = sha256_hex(canonical_json(identity))[:16]
    out_root = Path(out_root)
    out_dir = out_root / knowledge_id
    if (out_dir / MANIFEST_FILE).is_file():
        return out_dir

    partition, docs = _training_docs(store)
    maps = map_store(vocab, store, ruleset, cache=False)
    form2action = np.asarray(maps.lemma_obs, dtype=np.int64)
    form2action = np.where(form2action < vocab.size, form2action, -1)  # drop stopword ids
    freq, doc_freq, title_doc_freq = count_actions(store, docs, form2action, vocab.size)

    lemma_stats = np.zeros(vocab.size, dtype=LEMMA_STATS_DTYPE)
    lemma_stats["freq"] = freq.astype(np.float32)
    lemma_stats["doc_freq"] = doc_freq.astype(np.float32)
    present = doc_freq > 0
    mean = np.zeros(vocab.size, dtype=np.float64)
    mean[present] = freq[present] / doc_freq[present]
    lemma_stats["mean_count_present"] = mean.astype(np.float32)

    out_root.mkdir(parents=True, exist_ok=True)
    tmp = out_root / f"{knowledge_id}.tmp{os.getpid()}"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir()
    try:
        _write_npy(tmp / LEMMA_STATS_FILE, lemma_stats)
        _write_npy(tmp / TITLE_STATS_FILE, title_doc_freq.astype("<i8"))
        manifest = {
            "knowledge_id": knowledge_id,
            "format_version": KNOWLEDGE_FORMAT_VERSION,
            "builder_version": BUILDER_VERSION,
            "build_id": store.build_id,
            "build_kind": store.manifest.get("kind"),
            "sources": sorted(
                str(info.get("id", "?"))
                for info in store.manifest.get("sources", []) or []
                if isinstance(info, Mapping)
            ),
            "ruleset_name": ruleset.name,
            "ruleset_hash": ruleset_hash,
            "vocab_id": vocab.vocab_id,
            "vocab_size": vocab.size,
            "source_config": source_config,
            "regime": regime,
            "partition": partition,
            "n_docs": len(docs),
            "hashes": {
                LEMMA_STATS_FILE: file_sha256(tmp / LEMMA_STATS_FILE),
                TITLE_STATS_FILE: file_sha256(tmp / TITLE_STATS_FILE),
            },
        }
        (tmp / MANIFEST_FILE).write_bytes((canonical_json(manifest) + "\n").encode("utf-8"))
        if out_dir.exists():
            shutil.rmtree(out_dir)
        tmp.rename(out_dir)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return out_dir
