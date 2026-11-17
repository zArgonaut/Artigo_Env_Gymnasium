"""``AgentCard``: the configuration sheet saved with every result (SDD 12 §1, REQ-AGT-001)."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from artigo_env.core.config import ConfigBase, canonical_json

INTERFACES = ("core", "candidates", "text")
REGIMES = ("open", "closed", "oracle")


@dataclass(frozen=True)
class AgentCard(ConfigBase):
    """Identity of an agent configuration; ``content_hash`` names it in reports (REQ-REPO-004).

    ``vocab_id`` and ``knowledge_id`` record the artifacts the agent read (empty when unused).
    ``candidates`` stays ``None`` until the candidate interface of phase F5 exists.
    """

    name: str
    algorithm: str
    interface: str = "core"
    hyperparams: Mapping[str, Any] = field(default_factory=dict)
    services: tuple[str, ...] = ()
    candidates: Mapping[str, Any] | None = None
    reward_preset: str = "R0"
    ruleset: str = "canonical"
    scenario: str = "demo"
    source_config: str = "X->W"
    regime: str = "open"
    seed: int = 0
    vocab_id: str = ""
    knowledge_id: str = ""

    def __post_init__(self) -> None:
        if self.interface not in INTERFACES:
            raise ValueError(f"interface must be one of {INTERFACES}, got {self.interface!r}")
        if self.regime not in REGIMES:
            raise ValueError(f"regime must be one of {REGIMES}, got {self.regime!r}")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError(f"seed must be an integer, got {self.seed!r}")
        object.__setattr__(self, "hyperparams", dict(self.hyperparams))
        object.__setattr__(self, "services", tuple(self.services))

    def to_json(self) -> str:
        return canonical_json(self.to_dict())

    def save(self, path: Path) -> None:
        Path(path).write_bytes((self.to_json() + "\n").encode("utf-8"))

    @classmethod
    def load(cls, path: Path) -> AgentCard:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
