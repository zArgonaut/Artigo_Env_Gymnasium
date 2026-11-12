"""ArtigoEnv: a Gymnasium research platform reproducing the artigo.app game offline."""

__version__ = "0.1.0"

# Registers ArtigoEnv-v0 with Gymnasium (entry-point string only; nothing is loaded).
from artigo_env.envs import registration as _registration  # noqa: E402, F401
