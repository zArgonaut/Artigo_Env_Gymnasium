"""Hand-built vocabulary and knowledge artifacts for the reference-agent tests."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import pytest

from artigo_env.core.vocab import ALL_LENGTHS, Vocabulary
from artigo_env.engine.rulesets import get_ruleset
from artigo_env.services.knowledge import LEMMA_STATS_DTYPE, KnowledgeStore


def length_mask(*lengths: int) -> int:
    mask = 0
    for n in lengths:
        mask |= 1 << (min(n, 64) - 1)
    return mask


def write_vocab(path: Path, freq: Sequence[int], masks: Sequence[int], numeric=()) -> Path:
    size = len(freq)
    path.mkdir(parents=True)
    is_numeric = np.zeros(size, dtype=np.bool_)
    is_numeric[list(numeric)] = True
    mask = np.array(
        [int(ALL_LENGTHS) if is_numeric[i] else m for i, m in enumerate(masks)], dtype=np.uint64
    )
    np.save(path / "freq.npy", np.asarray(freq, dtype=np.int64))
    np.save(path / "doc_freq.npy", np.asarray(freq, dtype=np.int64))
    np.save(path / "is_numeric.npy", is_numeric)
    np.save(path / "length_mask.npy", mask.astype(np.uint64))
    manifest = {"format_version": 1, "vocab_id": path.name, "size": size, "n_stop": 0}
    manifest["n_forms"] = 0
    (path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return path


def write_knowledge(
    path: Path,
    vocab_id: str,
    *,
    freq: Sequence[float],
    doc_freq: Sequence[float],
    title_doc_freq: Sequence[int],
    n_docs: int,
    ruleset_hash: str = "",
) -> Path:
    size = len(freq)
    path.mkdir(parents=True)
    stats = np.zeros(size, dtype=LEMMA_STATS_DTYPE)
    stats["freq"] = freq
    stats["doc_freq"] = doc_freq
    df = np.asarray(doc_freq, dtype=np.float64)
    stats["mean_count_present"] = np.where(df > 0, np.asarray(freq) / np.maximum(df, 1), 0)
    np.save(path / "lemma_stats.npy", stats)
    np.save(path / "title_stats.npy", np.asarray(title_doc_freq, dtype="<i8"))
    manifest = {
        "knowledge_id": path.name,
        "format_version": 1,
        "n_docs": n_docs,
        "vocab_id": vocab_id,
        "vocab_size": size,
        "build_id": "synthetic",
        "ruleset_hash": ruleset_hash,
        "source_config": "X->W",
        "regime": "open",
    }
    (path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return path


Artifacts = Callable[..., tuple[Vocabulary, KnowledgeStore]]


@pytest.fixture
def artifacts(tmp_path) -> Artifacts:
    """``artifacts(freq, masks, title_doc_freq, ...) -> (vocab, knowledge)``.

    ``doc_freq`` defaults to ``freq`` (one occurrence per document); ``n_docs`` to 100; the
    knowledge store records the hash of the ruleset preset ``ruleset``.
    """
    opened: list[Vocabulary] = []
    counter = iter(range(1000))

    def make(
        freq: Sequence[int],
        masks: Sequence[int] | None = None,
        title_doc_freq: Sequence[int] | None = None,
        *,
        doc_freq: Sequence[int] | None = None,
        numeric: Sequence[int] = (),
        n_docs: int = 100,
        ruleset: str = "canonical",
    ) -> tuple[Vocabulary, KnowledgeStore]:
        k = next(counter)
        masks = masks if masks is not None else [length_mask(5)] * len(freq)
        vocab = Vocabulary.open(write_vocab(tmp_path / f"v{k}", freq, masks, numeric))
        opened.append(vocab)
        knowledge = KnowledgeStore.open(
            write_knowledge(
                tmp_path / f"k{k}",
                vocab.vocab_id,
                freq=freq,
                doc_freq=doc_freq if doc_freq is not None else freq,
                title_doc_freq=title_doc_freq if title_doc_freq is not None else [0] * len(freq),
                n_docs=n_docs,
                ruleset_hash=get_ruleset(ruleset).content_hash(),
            ),
            vocab=vocab,
        )
        return vocab, knowledge

    yield make
    for vocab in opened:
        vocab.close()


def observation(
    title_lengths: Sequence[int] = (), body_lengths: Sequence[int] = (), *, visible_title=()
) -> dict[str, np.ndarray]:
    """Observation with every listed token hidden (lemma ``-2``), padded like the environment.

    ``visible_title`` lists title positions shown as visible (lemma 0, length 0).
    """
    title = np.full(32, -3, dtype=np.int32)
    title_len = np.zeros(32, dtype=np.int16)
    title[: len(title_lengths)] = -2
    title_len[: len(title_lengths)] = title_lengths
    for pos in visible_title:
        title[pos], title_len[pos] = 0, 0
    body = np.full(64, -3, dtype=np.int32)
    body_len = np.zeros(64, dtype=np.int16)
    body[: len(body_lengths)] = -2
    body_len[: len(body_lengths)] = body_lengths
    return {
        "title_lemma": title,
        "title_length": title_len,
        "body_lemma": body,
        "body_length": body_len,
    }


@pytest.fixture
def make_obs() -> Callable[..., dict[str, np.ndarray]]:
    return observation


@pytest.fixture
def lmask() -> Callable[..., int]:
    return length_mask
