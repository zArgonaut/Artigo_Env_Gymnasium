"""Registry and factory of the reference agents (SDD 12 §2)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from artigo_env.agents.a0_random import A0RandomFreq
from artigo_env.agents.a1_greedy import A1GreedyFreq
from artigo_env.agents.a2_deductive import A2Deductive
from artigo_env.agents.base import Agent
from artigo_env.agents.card import AgentCard
from artigo_env.core.registry import Registry
from artigo_env.core.vocab import Vocabulary
from artigo_env.engine.rulesets.presets import get_ruleset
from artigo_env.engine.rulesets.ruleset import Ruleset
from artigo_env.services.knowledge import KnowledgeStore

AgentFactory = Callable[..., Agent]
AGENTS: Registry[AgentFactory] = Registry("agent")

ALGORITHMS = {"a0": "a0-random-freq", "a1": "a1-greedy-freq", "a2": "a2-deductive"}
SERVICES = {"a0": ("lemma_stats",), "a1": ("lemma_stats",), "a2": ("lemma_stats", "title_stats")}
DEFAULTS: dict[str, dict[str, Any]] = {
    "a0": {"smoothing": 1.0},
    "a1": {},
    "a2": {"lam": 1.0, "alpha": 0.5, "numeric_title": True},
}
CARD_FIELDS = ("reward_preset", "scenario")


def make_card(
    name: str,
    *,
    vocab: Vocabulary,
    knowledge: KnowledgeStore,
    ruleset: Ruleset,
    seed: int,
    hyperparams: dict[str, Any],
    **card_fields: Any,
) -> AgentCard:
    unknown = set(hyperparams) - set(DEFAULTS[name])
    if unknown:
        raise ValueError(
            f"unknown hyperparameters for {name}: {sorted(unknown)}; "
            f"accepted: {sorted(DEFAULTS[name])}"
        )
    return AgentCard(
        name=name,
        algorithm=ALGORITHMS[name],
        interface="core",
        hyperparams={**DEFAULTS[name], **hyperparams},
        services=SERVICES[name],
        ruleset=ruleset.name,
        source_config=knowledge.source_config,
        regime=knowledge.regime,
        seed=int(seed),
        vocab_id=vocab.vocab_id,
        knowledge_id=knowledge.knowledge_id,
        **card_fields,
    )


def _split(hyper: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    fields = {key: hyper.pop(key) for key in CARD_FIELDS if key in hyper}
    return hyper, fields


def _register(name: str) -> None:
    def factory(
        *,
        vocab: Vocabulary,
        knowledge: KnowledgeStore,
        ruleset: Ruleset | str,
        seed: int = 0,
        **hyperparams: Any,
    ) -> Agent:
        rules = get_ruleset(ruleset) if isinstance(ruleset, str) else ruleset
        knowledge.check(vocab=vocab, ruleset_hash=rules.content_hash())
        hyper, fields = _split(dict(hyperparams))
        card = make_card(
            name,
            vocab=vocab,
            knowledge=knowledge,
            ruleset=rules,
            seed=seed,
            hyperparams=hyper,
            **fields,
        )
        if name == "a0":
            return A0RandomFreq(card, knowledge)
        if name == "a1":
            return A1GreedyFreq(card, knowledge)
        return A2Deductive(card, knowledge, vocab, rules)

    AGENTS.register(name, factory)


for _name in ALGORITHMS:
    _register(_name)


def make_agent(
    name: str,
    *,
    vocab: Vocabulary,
    knowledge: KnowledgeStore,
    ruleset: Ruleset | str,
    seed: int = 0,
    **hyperparams: Any,
) -> Agent:
    """Build agent ``name`` (``a0``, ``a1``, ``a2``).

    ``hyperparams`` may also set the card fields ``reward_preset`` and ``scenario``.
    """
    return AGENTS.get(name)(
        vocab=vocab, knowledge=knowledge, ruleset=ruleset, seed=seed, **hyperparams
    )
