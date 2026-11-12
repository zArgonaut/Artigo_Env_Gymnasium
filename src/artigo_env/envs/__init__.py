"""ArtigoEnv ``envs`` layer (see docs/sdd/02, section B.2): the Gymnasium environments.

Importing the package registers ``ArtigoEnv-v0``; the environment class and its dependencies
(corpus store, engine, scenarios) are imported only on first access to ``ArtigoCoreEnv``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from artigo_env.envs import registration as registration

if TYPE_CHECKING:
    from artigo_env.envs.core_env import ArtigoCoreEnv
    from artigo_env.envs.record import EpisodeNotFinishedError, EpisodeRecord

__all__ = ["ArtigoCoreEnv", "EpisodeNotFinishedError", "EpisodeRecord"]

_LAZY = {
    "ArtigoCoreEnv": "artigo_env.envs.core_env",
    "EpisodeRecord": "artigo_env.envs.record",
    "EpisodeNotFinishedError": "artigo_env.envs.record",
}


def __getattr__(name: str) -> Any:
    module_name = _LAZY.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    value = getattr(importlib.import_module(module_name), name)
    globals()[name] = value
    return value
