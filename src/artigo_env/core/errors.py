"""Exception hierarchy shared by every layer."""

from __future__ import annotations


class ArtigoEnvError(Exception):
    """Base class for all errors raised by the package."""


class DataRootNotConfiguredError(ArtigoEnvError):
    """``ARTIGO_DATA_ROOT`` is missing, empty or not an absolute path."""


class DuplicateNameError(ArtigoEnvError):
    """A component was registered under a name that is already taken."""


class UnknownNameError(ArtigoEnvError, KeyError):
    """A component name was requested that is not registered."""

    def __str__(self) -> str:
        return str(self.args[0]) if self.args else ""


class UnplayableArticleError(ArtigoEnvError):
    """An article has no content token in its title (it can never be won)."""


class InvalidActionError(ArtigoEnvError):
    """An action was applied where it is not allowed (for example after the game was won)."""


class IncompatibleRulesetError(ArtigoEnvError):
    """A ruleset cannot be used with the given build or component combination."""
