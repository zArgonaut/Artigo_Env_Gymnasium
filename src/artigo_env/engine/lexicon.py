"""Lexicon: ties forms and lemmas to integer ids (SDD 07 §3, SDD 09)."""

from __future__ import annotations

from typing import Protocol


class Lexicon(Protocol):
    def lemma_id(self, lemma: str) -> int:
        """Id of ``lemma``, or ``-1`` when it is not in the lexicon."""
        ...

    def form_id(self, form: str) -> int:
        """Id of ``form``, or ``-1`` when it is not in the lexicon."""
        ...

    def lemma_text(self, lemma_id: int) -> str: ...


class InMemoryLexicon:
    """Growable lexicon with ids assigned in order of first appearance, from 0."""

    def __init__(self) -> None:
        self._forms: dict[str, int] = {}
        self._lemmas: dict[str, int] = {}
        self._lemma_texts: list[str] = []

    def intern_form(self, form: str) -> int:
        return self._forms.setdefault(form, len(self._forms))

    def intern_lemma(self, lemma: str) -> int:
        known = self._lemmas.get(lemma)
        if known is None:
            known = self._lemmas[lemma] = len(self._lemma_texts)
            self._lemma_texts.append(lemma)
        return known

    def lemma_id(self, lemma: str) -> int:
        return self._lemmas.get(lemma, -1)

    def form_id(self, form: str) -> int:
        return self._forms.get(form, -1)

    def lemma_text(self, lemma_id: int) -> str:
        if not 0 <= lemma_id < len(self._lemma_texts):
            raise KeyError(f"unknown lemma id: {lemma_id}")
        return self._lemma_texts[lemma_id]
