"""A0–A2 play a full demo episode through ``gym.make`` (REQ-AGT-002, REQ-AGT-003).

The test loop owns the environment: it hands each agent only ``obs``, the allow-listed ``info``
and the action mask; the agent never touches ``env``.
"""

from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import pytest

import artigo_env  # noqa: F401  (registers ArtigoEnv-v0)
from artigo_env.agents import make_agent
from artigo_env.core.resources import resource_path
from artigo_env.core.vocab import Vocabulary
from artigo_env.envs.core_env import INFO_KEYS
from artigo_env.services.knowledge import KnowledgeStore, find_knowledge

pytestmark = [pytest.mark.fast, pytest.mark.req("REQ-AGT-002"), pytest.mark.req("REQ-AGT-003")]


def play_episode(env, agent, article_index: int) -> tuple[int, bool]:
    obs, info = env.reset(seed=0, options={"article_index": article_index})
    agent.reset(obs, info)
    steps, terminated, truncated = 0, False, False
    while not (terminated or truncated):
        assert set(info) <= INFO_KEYS
        mask = env.unwrapped.action_masks()
        action = agent.act(obs, info, mask)
        assert mask[action]
        obs, _, terminated, truncated, info = env.step(action)
        steps += 1
    return steps, bool(terminated)


@pytest.mark.parametrize("name", ["a0", "a1", "a2"])
def test_agent_plays_a_demo_episode(name):
    env = gym.make("ArtigoEnv-v0")
    try:
        vocab_dir = Path(str(resource_path("demo", "vocab", env.unwrapped.vocab_id)))
        with Vocabulary.open(vocab_dir) as vocab:
            knowledge = KnowledgeStore.open(find_knowledge(vocab.vocab_id), vocab=vocab)
            agent = make_agent(name, vocab=vocab, knowledge=knowledge, ruleset="canonical", seed=3)
            first = play_episode(env, agent, 0)
            assert first[0] >= 1
            assert play_episode(env, agent, 0) == first  # deterministic, also after reset
    finally:
        env.close()
