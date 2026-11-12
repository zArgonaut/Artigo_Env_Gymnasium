"""Ruleset presets (SDD 08 §A.3). Python is the source; YAML mirrors come with the runner."""

from __future__ import annotations

from dataclasses import replace

from artigo_env.core.registry import Registry
from artigo_env.engine.rulesets.ruleset import Ruleset

PRESETS: Registry[Ruleset] = Registry("ruleset")

_CANONICAL = Ruleset(name="canonical")

PRESETS.register("canonical", _CANONICAL)
PRESETS.register("no-grouping", replace(_CANONICAL, name="no-grouping", lemmatizer="identity"))
PRESETS.register("no-lengths", replace(_CANONICAL, name="no-lengths", lengths_visible="none"))
PRESETS.register("strict-noops", replace(_CANONICAL, name="strict-noops", noop_cost=1))
PRESETS.register(
    "hard",
    replace(_CANONICAL, name="hard", lemmatizer="identity", lengths_visible="none", noop_cost=1),
)


def get_ruleset(name: str) -> Ruleset:
    """Return the preset registered as ``name``."""
    return PRESETS.get(name)
