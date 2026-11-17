"""``AgentCard`` round trip and identity (SDD 12 §1)."""

from __future__ import annotations

import pytest

from artigo_env.agents import AGENTS, AgentCard, make_agent

pytestmark = [pytest.mark.fast, pytest.mark.req("REQ-AGT-001")]


def test_round_trip_and_hash(tmp_path):
    card = AgentCard(
        name="a2",
        algorithm="a2-deductive",
        hyperparams={"lam": 1.0, "alpha": 0.5},
        services=("lemma_stats", "title_stats"),
        seed=7,
        vocab_id="v",
        knowledge_id="k",
    )
    assert AgentCard.from_dict(card.to_dict()) == card
    path = tmp_path / "card.json"
    card.save(path)
    loaded = AgentCard.load(path)
    assert loaded == card and loaded.content_hash() == card.content_hash()
    assert len(card.content_hash()) == 64
    other = AgentCard.from_dict({**card.to_dict(), "seed": 8})
    assert other.content_hash() != card.content_hash()
    changed = AgentCard.from_dict({**card.to_dict(), "hyperparams": {"lam": 2.0, "alpha": 0.5}})
    assert changed.content_hash() != card.content_hash()


def test_card_validation():
    with pytest.raises(ValueError, match="interface"):
        AgentCard(name="x", algorithm="y", interface="pixels")
    with pytest.raises(ValueError, match="regime"):
        AgentCard(name="x", algorithm="y", regime="half-open")


def test_registry_builds_cards(artifacts):
    vocab, knowledge = artifacts([5, 3, 1])
    assert set(AGENTS.names()) == {"a0", "a1", "a2"}
    for name in AGENTS.names():
        agent = make_agent(name, vocab=vocab, knowledge=knowledge, ruleset="canonical", seed=3)
        card = agent.card
        assert card.name == name and card.seed == 3 and card.ruleset == "canonical"
        assert card.vocab_id == vocab.vocab_id and card.knowledge_id == knowledge.knowledge_id
        assert card.source_config == "X->W" and card.regime == "open"
    a2 = make_agent("a2", vocab=vocab, knowledge=knowledge, ruleset="canonical", lam=2.0)
    assert a2.card.hyperparams == {"lam": 2.0, "alpha": 0.5, "numeric_title": True}
    with pytest.raises(ValueError, match="unknown hyperparameters"):
        make_agent("a1", vocab=vocab, knowledge=knowledge, ruleset="canonical", lam=2.0)
