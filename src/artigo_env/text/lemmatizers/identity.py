"""The identity lemmatizer (no grouping of word variants)."""

from __future__ import annotations

from artigo_env.core.config import sha256_hex


class IdentityLemmatizer:
    """Lemma equals the form (the ``no-grouping`` lever)."""

    name = "identity"

    def lemma(self, form: str) -> str:
        return form

    def content_hash(self) -> str:
        return sha256_hex("identity:1")
