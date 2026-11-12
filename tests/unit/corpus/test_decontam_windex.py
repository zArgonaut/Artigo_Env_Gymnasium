"""W index: persistence, content hash and the three-pass protocol (SDD 06 §7)."""

from __future__ import annotations

import json
import random
import time

import numpy as np
import pytest

from artigo_env.corpus.decontam import minhash
from artigo_env.corpus.decontam.coverage import (
    N_GRAM,
    DocFrequency,
    TrainDoc,
    WDoc,
    WGramIndex,
    decontaminate,
    hashes_of_interest,
    is_contaminated,
)
from artigo_env.corpus.decontam.grams import gram_hashes
from artigo_env.corpus.decontam.windex import WIndex, directory_hash

VOCAB = [f"w{i}" for i in range(400)] + ["coração", "ação", "pássaro", "árvore"]
PHRASE = [
    "segundo",
    "o",
    "instituto",
    "brasileiro",
    "de",
    "geografia",
    "e",
    "estatística",
    "a",
    "população",
    "era",
    "de",
]


def text(rng: random.Random, n: int) -> list[str]:
    return [rng.choice(VOCAB) for _ in range(n)]


def synthetic_batch(seed: int = 7) -> tuple[list[WDoc], list[TrainDoc]]:
    """W articles and a training batch with every barrier and the frequent-phrase trap."""
    rng = random.Random(seed)
    w = [WDoc(f"Artigo {k}", text(rng, 400)) for k in range(8)]
    w[0] = WDoc(w[0].title, w[0].tokens + PHRASE + text(rng, 30))
    train: list[TrainDoc] = []
    for k in range(40):
        train.append(TrainDoc(f"clean{k:02d}", f"Notícia {k}", text(rng, 300)))
    for k in range(12):  # frequent phrase also present in W
        train.append(TrainDoc(f"phrase{k:02d}", f"Censo {k}", text(rng, 100) + PHRASE))
    for k in range(8):  # excerpts of W from 10 to 17 tokens
        size = 10 + k
        start = rng.randrange(0, 300)
        body = text(rng, 100) + list(w[k].tokens[start : start + size]) + text(rng, 100)
        train.append(TrainDoc(f"excerpt{size}", f"Trecho {k}", body))
    train.append(TrainDoc("title", "artigo 3", text(rng, 200)))
    near = [t if rng.random() > 0.03 else rng.choice(VOCAB) for t in w[5].tokens]
    train.append(TrainDoc("near", "Quase", near))
    paraphrase = [t if i % 10 != 5 else rng.choice(VOCAB) for i, t in enumerate(w[6].tokens)]
    train.append(TrainDoc("paraphrase", "Paráfrase", paraphrase))
    train.append(TrainDoc("short", "Curto", text(rng, 3)))
    return w, train


def build(w: list[WDoc]) -> WIndex:
    return WIndex.build(((doc.title, doc.tokens) for doc in w), extra_titles=["Sem Texto"])


def three_pass(index: WIndex, train: list[TrainDoc]):
    interest = [hashes_of_interest(doc.tokens, index.grams) for doc in train]
    present = [h for h in interest if h is not None]
    union = np.concatenate(present) if present else np.empty(0, dtype=np.uint64)
    freq = DocFrequency(union)
    for doc in train:
        freq.add(doc.tokens)
    frequent = freq.frequent()
    return [index.judge(doc, frequent) for doc in train]


@pytest.mark.fast
@pytest.mark.req("REQ-COR-008", "REQ-COR-009")
def test_three_pass_protocol_agrees_with_in_memory():
    w, train = synthetic_batch()
    expected = decontaminate(train, w)
    got = three_pass(build(w), train)
    assert got == expected
    reasons = {v.doc_id: v.reason for v in got}
    # The batch exercises every barrier and the frequent-phrase rule.
    assert reasons["title"] == "w_title"
    assert reasons["near"] in {"w_13gram", "w_near_duplicate"}
    assert reasons["excerpt13"] == "w_13gram" and reasons["excerpt17"] == "w_13gram"
    assert reasons["excerpt10"] is None
    assert all(reasons[f"phrase{k:02d}"] is None for k in range(12))
    assert all(reasons[f"clean{k:02d}"] is None for k in range(40))
    assert reasons["paraphrase"] is None


@pytest.mark.fast
@pytest.mark.req("REQ-COR-009")
def test_near_duplicate_barrier_through_judge():
    # A W article syndicated in 11 training documents: all its 13-grams are frequent, so the
    # coverage rule ignores them and only the near-duplicate barrier removes the copies.
    rng = random.Random(3)
    w = [WDoc("Outro", text(rng, 300)), WDoc("A", text(rng, 600))]
    train = [TrainDoc(f"copy{k:02d}", f"Agência {k}", w[1].tokens) for k in range(11)]
    train.append(TrainDoc("clean", "Limpo", text(rng, 600)))
    expected = decontaminate(train, w)
    assert three_pass(build(w), train) == expected
    assert [(v.reason, v.covered, v.match) for v in expected[:11]] == [
        ("w_near_duplicate", 0, 1)
    ] * 11
    assert expected[11].reason is None


@pytest.mark.fast
@pytest.mark.req("DEC-047")
def test_save_load_roundtrip_and_content_hash(tmp_path):
    w, train = synthetic_batch()
    index = build(w)
    index.save(tmp_path / "w")
    loaded = WIndex.load(tmp_path / "w")
    assert len(loaded.titles) == 9 and "sem texto" in loaded.titles
    assert np.array_equal(loaded.grams.hashes, index.grams.hashes)
    assert len(loaded.near) == len(w)
    assert loaded.content_hash() == index.content_hash() == directory_hash(tmp_path / "w")
    assert three_pass(loaded, train) == three_pass(index, train)
    # Same input, same bytes.
    build(w).save(tmp_path / "again")
    for path in sorted((tmp_path / "w").iterdir()):
        assert path.read_bytes() == (tmp_path / "again" / path.name).read_bytes()
    # Any change of content changes the hash.
    changed = build(w[:-1])
    assert changed.content_hash() != index.content_hash()


@pytest.mark.fast
@pytest.mark.req("DEC-047")
def test_load_rejects_other_settings(tmp_path):
    w, _ = synthetic_batch()
    build(w).save(tmp_path)
    meta = json.loads((tmp_path / "meta.json").read_text(encoding="utf-8"))
    meta["n_gram"] = 12
    (tmp_path / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(ValueError, match="different settings"):
        WIndex.load(tmp_path)


@pytest.mark.fast
@pytest.mark.req("DEC-047")
def test_empty_index():
    index = WIndex.build([])
    assert len(index.titles) == len(index.grams) == len(index.near) == 0
    verdict = index.judge(TrainDoc("d", "T", ["a"] * 30), np.empty(0, dtype=np.uint64))
    assert verdict.reason is None


@pytest.mark.perf
@pytest.mark.req("REQ-COR-008", "REQ-COR-009")
@pytest.mark.parametrize("n_tokens", [300, 1000, 3000])
def test_cost_per_document(n_tokens, capsys):
    """Informative: ms per document for 13-grams (pass C) and MinHash (target <= 3 ms at 1,000)."""
    rng = random.Random(n_tokens)
    w = WGramIndex.build([text(rng, 3000) for _ in range(50)])
    docs = [text(rng, n_tokens) for _ in range(200)]
    frequent = np.unique(gram_hashes(docs[0], N_GRAM))
    start = time.perf_counter()
    for doc in docs:
        is_contaminated(doc, w, frequent)
    grams_ms = (time.perf_counter() - start) * 1000 / len(docs)
    start = time.perf_counter()
    for doc in docs:
        minhash.signature(minhash.shingle_hashes(doc))
    minhash_ms = (time.perf_counter() - start) * 1000 / len(docs)
    with capsys.disabled():
        print(f"\n{n_tokens} tokens: 13-grams {grams_ms:.3f} ms, MinHash {minhash_ms:.3f} ms")
    assert grams_ms > 0 and minhash_ms > 0
