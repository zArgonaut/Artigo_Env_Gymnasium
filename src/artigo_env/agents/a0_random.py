"""A0: random guesses weighted by training frequency (SDD 12 §2)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import numpy.typing as npt

from artigo_env.agents.base import Obs, ReferenceAgent, allowed_actions
from artigo_env.agents.card import AgentCard
from artigo_env.services.knowledge import KnowledgeStore


class A0RandomFreq(ReferenceAgent):
    """Samples an allowed action with probability proportional to ``freq + smoothing``.

    ``smoothing`` (default 1) keeps actions never seen in training reachable.
    """

    def __init__(self, card: AgentCard, knowledge: KnowledgeStore) -> None:
        super().__init__(card)
        smoothing = float(card.hyperparams.get("smoothing", 1.0))
        if not smoothing > 0:
            raise ValueError(f"smoothing must be positive, got {smoothing!r}")
        self.weights = knowledge.lemma_stats["freq"].astype(np.float64) + smoothing

    def act(
        self, obs: Obs, info: Mapping[str, Any], action_mask: npt.NDArray[np.bool_] | None
    ) -> int:
        allowed = allowed_actions(action_mask, len(self.weights))
        cumulative = np.cumsum(np.where(allowed, self.weights, 0.0))
        total = float(cumulative[-1])
        draw = self.rng.random() * total
        action = int(np.searchsorted(cumulative, draw, side="right"))
        action = min(action, len(cumulative) - 1)
        if not allowed[action]:  # guard against a draw on a float boundary
            action = int(np.flatnonzero(allowed)[-1])
        return action
