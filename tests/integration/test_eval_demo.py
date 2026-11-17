"""A0, A1 and A2 play the 30 demo articles through the harness (WP4.5 integration).

The ordering A2 <= A1 <= A0 in N mean is recorded in the test output, not required.
"""

from __future__ import annotations

import pytest

from artigo_env.envs import ArtigoCoreEnv
from artigo_env.envs.core_env import resolve_corpus, resolve_vocab
from artigo_env.evaluation import evaluate

agents = pytest.importorskip("artigo_env.agents")
knowledge_mod = pytest.importorskip("artigo_env.services.knowledge")

pytestmark = pytest.mark.slow

AGENT_NAMES = ("a0", "a1", "a2")


def run_agent(name: str, out_dir, seed: int = 0):
    from artigo_env.core.vocab import Vocabulary
    from artigo_env.corpus.store import CorpusStore
    from artigo_env.engine.rulesets import get_ruleset

    ruleset = get_ruleset("canonical")
    with CorpusStore.open(resolve_corpus("demo")) as store:
        build_id = store.build_id
    vocab_path = resolve_vocab(None, "demo", build_id, ruleset)
    with Vocabulary.open(vocab_path) as vocab:
        knowledge = knowledge_mod.KnowledgeStore.open(
            knowledge_mod.find_knowledge(vocab.vocab_id), vocab=vocab
        )

        def agent_for(s: int):
            return agents.make_agent(
                name, vocab=vocab, knowledge=knowledge, ruleset=ruleset, seed=s
            )

        return evaluate(
            agent_for,
            lambda: ArtigoCoreEnv(corpus="demo", scenario="demo", ruleset=ruleset),
            "demo",
            [seed],
            out_dir=out_dir,
        )


@pytest.mark.req("REQ-EVL-001", "REQ-AGT-001", "REQ-AGT-003")
def test_reference_agents_on_the_demo_corpus(tmp_path, capsys):
    results = {name: run_agent(name, tmp_path / name) for name in AGENT_NAMES}
    for result in results.values():
        assert len(result.episodes) == 30
        card = result.summary["agent_card"]
        assert card is not None and card.get("content_hash")
    n_mean = {name: r.summary["metrics"]["n_mean"]["value"] for name, r in results.items()}
    with capsys.disabled():
        ordered = n_mean["a2"] <= n_mean["a1"] <= n_mean["a0"]
        print(f"\nN mean on demo: {n_mean} (A2 <= A1 <= A0: {ordered})")
    again = run_agent("a2", tmp_path / "again")
    assert [e.to_dict() for e in again.episodes] == [e.to_dict() for e in results["a2"].episodes]
