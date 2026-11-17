"""A1: greedy by training frequency (SDD 12 §2)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import numpy.typing as npt

from artigo_env.agents.base import Obs, ReferenceAgent, allowed_actions
from artigo_env.agents.card import AgentCard
from artigo_env.services.knowledge import KnowledgeStore


class A1GreedyFreq(ReferenceAgent):
    """The allowed action of highest ``freq``; ties go to the smallest id."""

    def __init__(self, card: AgentCard, knowledge: KnowledgeStore) -> None:
        super().__init__(card)
        self.freq = knowledge.lemma_stats["freq"].astype(np.float64)

    def act(
        self, obs: Obs, info: Mapping[str, Any], action_mask: npt.NDArray[np.bool_] | None
    ) -> int:
        allowed = allowed_actions(action_mask, len(self.freq))
        return int(np.argmax(np.where(allowed, self.freq, -np.inf)))  # argmax: first maximum
