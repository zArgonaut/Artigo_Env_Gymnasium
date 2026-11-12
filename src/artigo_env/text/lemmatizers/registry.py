"""Registry of named lemmatizers (REQ-RUL-003)."""

from __future__ import annotations

from collections.abc import Callable

from artigo_env.core.errors import ArtigoEnvError
from artigo_env.core.paths import data_path
from artigo_env.core.registry import Registry
from artigo_env.core.resources import resource_path
from artigo_env.text.lemmatizers.base import Lemmatizer, RuleLemmatizer
from artigo_env.text.lemmatizers.identity import IdentityLemmatizer
from artigo_env.text.lemmatizers.tables import load_tables

LEMMATIZERS: Registry[Callable[[], Lemmatizer]] = Registry("lemmatizer")

_OPEN_TABLES = ("lemmatizers", "open-tables", "tables.json")
_REFERENCE_LOCAL = ("reference", "lemmatizer", "tables.json")
_CACHE: dict[str, Lemmatizer] = {}


def _identity() -> Lemmatizer:
    return IdentityLemmatizer()


def _open_tables() -> Lemmatizer:
    node = resource_path(*_OPEN_TABLES)
    if not node.is_file():
        raise ArtigoEnvError(
            "open-tables lemmatizer data not found: expected "
            f"data/{'/'.join(_OPEN_TABLES)} inside the package"
        )
    return RuleLemmatizer("open-tables", load_tables(node))


def _reference_local() -> Lemmatizer:
    path = data_path(*_REFERENCE_LOCAL)
    if not path.is_file():
        raise ArtigoEnvError(
            "reference-local lemmatizer tables not found: expected "
            f"{'/'.join(_REFERENCE_LOCAL)} under ARTIGO_DATA_ROOT ({path})"
        )
    return RuleLemmatizer("reference-local", load_tables(path))


LEMMATIZERS.register("identity", _identity)
LEMMATIZERS.register("open-tables", _open_tables)
LEMMATIZERS.register("reference-local", _reference_local)


def get_lemmatizer(name: str) -> Lemmatizer:
    """Return the lemmatizer registered as ``name`` (built once per name per process)."""
    cached = _CACHE.get(name)
    if cached is None:
        cached = _CACHE[name] = LEMMATIZERS.get(name)()
    return cached
