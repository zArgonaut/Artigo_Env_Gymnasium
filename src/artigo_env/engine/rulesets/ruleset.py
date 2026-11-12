"""The ruleset: how the game works (SDD 08 §A.2, DEC-059, DEC-060)."""

from __future__ import annotations

from dataclasses import dataclass, field

from artigo_env.core.config import ConfigBase, canonical_json, sha256_hex
from artigo_env.core.errors import IncompatibleRulesetError
from artigo_env.engine.rulesets.components import (
    HINT_RULES,
    HYPHEN_RULES,
    LENGTH_RULES,
    NUMBER_RULES,
    SEPARATOR_RULES,
    STOPWORD_KINDS,
    TITLE_RULES,
    HintSpec,
)
from artigo_env.text.lemmatizers.registry import LEMMATIZERS, get_lemmatizer
from artigo_env.text.stopwords import StopwordSpec, load_stopwords


@dataclass(frozen=True)
class Ruleset(ConfigBase):
    """Declarative ruleset; every lever is the name of a registered component.

    Construction validates names against the registries only; lemmatizer tables are loaded
    lazily (by ``ArticleBuilder``, ``Engine`` and ``content_hash``).
    """

    name: str
    version: int = 1
    tokenizer_hyphen: str = "split"
    guess_separator: str = "strip"
    stopwords: StopwordSpec = field(default_factory=StopwordSpec)
    lemmatizer: str = "open-tables"
    numbers: str = "substring"
    lengths_visible: str = "all"
    noop_cost: int = 0
    title_parenthetical: str = "required"
    hint: HintSpec = field(default_factory=HintSpec)

    def __post_init__(self) -> None:
        checks = (
            ("tokenizer_hyphen", self.tokenizer_hyphen, HYPHEN_RULES),
            ("guess_separator", self.guess_separator, SEPARATOR_RULES),
            ("stopwords.kind", self.stopwords.kind, STOPWORD_KINDS),
            ("numbers", self.numbers, NUMBER_RULES),
            ("lengths_visible", self.lengths_visible, LENGTH_RULES),
            ("title_parenthetical", self.title_parenthetical, TITLE_RULES),
            ("hint.kind", self.hint.kind, HINT_RULES),
            ("lemmatizer", self.lemmatizer, LEMMATIZERS),
        )
        for lever, value, registry in checks:
            if value not in registry:
                registry.get(value)  # raises UnknownNameError listing the available names
                raise ValueError(f"invalid value for {lever}: {value!r}")
        if isinstance(self.noop_cost, bool) or self.noop_cost not in (0, 1):
            raise ValueError(f"noop_cost must be 0 or 1, got {self.noop_cost!r}")

    def content_hash(self) -> str:
        """Identity including the stopword list and the lemmatizer data (REQ-RUL-001)."""
        payload = {
            "config": self.to_dict(),
            "stopwords": load_stopwords(self.stopwords).content_hash(),
            "lemmatizer": get_lemmatizer(self.lemmatizer).content_hash(),
        }
        return sha256_hex(canonical_json(payload))

    def check_build(self, tokenizer_hyphen: str) -> None:
        """Fail when the corpus build used a different hyphen mode (REQ-RUL-004)."""
        if tokenizer_hyphen != self.tokenizer_hyphen:
            raise IncompatibleRulesetError(
                f"ruleset {self.name!r} needs tokenizer_hyphen={self.tokenizer_hyphen!r} "
                f"but the build used {tokenizer_hyphen!r}"
            )
