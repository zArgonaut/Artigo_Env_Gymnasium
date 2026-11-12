"""Engine lexicon backed by a compiled corpus (SDD 07 §3)."""

from __future__ import annotations

from artigo_env.corpus.store.reader import CorpusStore, LemmaView


class StoreLexicon:
    """Implements ``engine.Lexicon`` over a store and one of its lemma views.

    Ids are the build's global form and lemma ids, the same ones ``CorpusStore.article`` puts in
    the token sequences. The text-to-id dictionaries are built on first use.
    """

    def __init__(self, store: CorpusStore, lemma_view: LemmaView) -> None:
        self.store = store
        self.view = lemma_view

    def lemma_id(self, lemma: str) -> int:
        return self.view.lemma_id(lemma)

    def form_id(self, form: str) -> int:
        return self.store.form_id(form)

    def lemma_text(self, lemma_id: int) -> str:
        return self.view.lemma_text(lemma_id)
