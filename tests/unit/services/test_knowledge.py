"""Reader of the minimal knowledge store and its compatibility checks (SDD 10 §4, §6)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from artigo_env.core.errors import ArtigoEnvError
from artigo_env.core.resources import resource_path
from artigo_env.core.vocab import Vocabulary
from artigo_env.engine.rulesets import PRESETS, get_ruleset
from artigo_env.services.knowledge import (
    KnowledgeMismatchError,
    KnowledgeStore,
    find_knowledge,
)

pytestmark = pytest.mark.fast

DEMO = Path(str(resource_path("demo")))


def demo_vocabs() -> dict[str, Path]:
    found = {}
    for manifest in sorted((DEMO / "vocab").glob("*/manifest.json")):
        data = json.loads(manifest.read_text(encoding="utf-8"))
        found[data["ruleset_name"]] = manifest.parent
    return found


@pytest.mark.req("REQ-SVC-005")
def test_mismatched_vocabulary_is_refused():
    vocabs = demo_vocabs()
    with (
        Vocabulary.open(vocabs["canonical"]) as canonical,
        Vocabulary.open(vocabs["no-grouping"]) as other,
    ):
        path = find_knowledge(canonical.vocab_id)
        knowledge = KnowledgeStore.open(path, vocab=canonical)
        assert knowledge.vocab_id == canonical.vocab_id
        with pytest.raises(KnowledgeMismatchError, match="does not match") as error:
            KnowledgeStore.open(path, vocab=other)
        assert other.vocab_id in str(error.value) and canonical.vocab_id in str(error.value)
    with pytest.raises(KnowledgeMismatchError, match="ruleset"):
        knowledge.check(ruleset_hash=get_ruleset("hard").content_hash())
    with pytest.raises(KnowledgeMismatchError, match="source configuration"):
        knowledge.check(source_config="W->W")
    with pytest.raises(KnowledgeMismatchError, match="regime"):
        knowledge.check(regime="closed")
    knowledge.check(ruleset_hash=get_ruleset("canonical").content_hash(), regime="open")


@pytest.mark.req("REQ-KNW-002")
def test_one_packaged_store_per_demo_vocabulary():
    vocabs = demo_vocabs()
    assert set(vocabs) == set(PRESETS.names())
    for name, directory in vocabs.items():
        with Vocabulary.open(directory) as vocab:
            knowledge = KnowledgeStore.open(find_knowledge(vocab.vocab_id), vocab=vocab)
            assert knowledge.manifest["ruleset_name"] == name
            assert knowledge.ruleset_hash == get_ruleset(name).content_hash()
            assert knowledge.n_docs == 30 and knowledge.manifest["partition"] == "demo"
            stats = knowledge.lemma_stats
            # The vocabulary counts the same partition: frequencies must agree exactly.
            np.testing.assert_array_equal(stats["freq"], np.asarray(vocab.freq, np.float32))
            np.testing.assert_array_equal(stats["doc_freq"], np.asarray(vocab.doc_freq, np.float32))
            title = knowledge.title_stats["title_doc_freq"]
            assert (title <= stats["doc_freq"]).all() and title.sum() > 0


@pytest.mark.req("REQ-SVC-005")
def test_missing_or_unknown_store(tmp_path):
    with pytest.raises(ArtigoEnvError, match="not a knowledge directory"):
        KnowledgeStore.open(tmp_path)
    with pytest.raises(ArtigoEnvError, match="no knowledge store"):
        find_knowledge("0000000000000000", root=tmp_path)
