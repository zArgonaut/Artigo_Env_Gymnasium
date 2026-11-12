"""Registration of ``ArtigoEnv-v0`` with Gymnasium (SDD 09 §8).

Importing this module registers the environment by entry-point string only: no corpus, no
vocabulary and not even the environment module is loaded until ``gym.make`` is called.
"""

from __future__ import annotations

import gymnasium as gym

ENV_ID = "ArtigoEnv-v0"
ENTRY_POINT = "artigo_env.envs:ArtigoCoreEnv"
MAX_EPISODE_STEPS = 400  # T_max ([LOCAL-TMAX-1]); equals ArtigoCoreEnv's max_steps default
DEFAULT_KWARGS = {"corpus": "demo", "ruleset": "canonical", "scenario": "demo"}


def register() -> None:
    """Register ``ArtigoEnv-v0`` once per process (idempotent)."""
    if ENV_ID in gym.registry:
        return
    gym.register(
        id=ENV_ID,
        entry_point=ENTRY_POINT,
        max_episode_steps=MAX_EPISODE_STEPS,
        kwargs=dict(DEFAULT_KWARGS),
    )


register()
