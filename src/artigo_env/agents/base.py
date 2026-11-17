"""Agent protocol and the shared base of the reference agents (SDD 12 §1, DEC-082).

Agents see only what a player sees: the observation dictionary produced by the environment
(NumPy arrays), the allow-listed ``info`` and the boolean action mask (REQ-AGT-002). They never
receive the environment itself. Reference agents read per-action statistics from an opened
``KnowledgeStore`` and the action vocabulary ``Vocabulary``.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

from artigo_env.agents.card import AgentCard
from artigo_env.core.errors import ArtigoEnvError

Obs = Mapping[str, npt.NDArray[Any]]

HIDDEN = -2  # observation code of a hidden token (SDD 09 §3)


class NoActionAvailableError(ArtigoEnvError):
    """The action mask allows no action."""


@runtime_checkable
class Agent(Protocol):
    card: AgentCard

    def reset(self, obs: Obs, info: Mapping[str, Any]) -> None: ...

    def act(
        self, obs: Obs, info: Mapping[str, Any], action_mask: npt.NDArray[np.bool_] | None
    ) -> int | str: ...

    def observe(self, transition: Any) -> None: ...

    def save(self, path: Path) -> None: ...


def allowed_actions(mask: npt.NDArray[np.bool_] | None, size: int) -> npt.NDArray[np.bool_]:
    """The mask as ``bool[size]`` (all allowed when ``None``); fails when nothing is allowed."""
    if mask is None:
        allowed = np.ones(size, dtype=np.bool_)
    else:
        allowed = np.asarray(mask, dtype=np.bool_)
        if allowed.shape != (size,):
            raise ValueError(f"action mask of shape {allowed.shape}, expected ({size},)")
    if not allowed.any():
        raise NoActionAvailableError("the action mask allows no action")
    return allowed


class ReferenceAgent:
    """Common state of A0–A2: a card, a seed and an own generator re-seeded by ``reset``.

    ``reset`` re-seeds the generator from the seed alone, so the decisions of an episode depend
    only on the seed and on what the agent observes in that episode: a fresh agent and a reused
    one act identically (REQ-AGT-003). Episodes still differ because their articles differ.
    """

    def __init__(self, card: AgentCard) -> None:
        self.card = card
        self.seed = int(card.seed)
        self.rng = np.random.default_rng(self.seed)

    def reset(self, obs: Obs, info: Mapping[str, Any]) -> None:
        self.rng = np.random.default_rng(self.seed)

    def observe(self, transition: Any) -> None:
        """Reference agents do not learn online."""

    def save(self, path: Path) -> None:
        """Write the card; reference agents have no other state (rebuild with ``make_agent``)."""
        self.card.save(path)

    def act(
        self, obs: Obs, info: Mapping[str, Any], action_mask: npt.NDArray[np.bool_] | None
    ) -> int:
        raise NotImplementedError
