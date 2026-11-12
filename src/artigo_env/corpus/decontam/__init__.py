"""Decontamination of training documents against the evaluation set W (SDD 06 §7, DEC-047).

Three barriers, in order: same title (:mod:`.titles`), shared 13-grams under the coverage rule
(:mod:`.coverage`) and near duplicates by MinHash/LSH with an exact Jaccard decision
(:mod:`.minhash`). :class:`.windex.WIndex` bundles W's data for the pipeline; the in-memory
:func:`.coverage.decontaminate` is the reference implementation of the decision.
"""

from __future__ import annotations

from artigo_env.corpus.decontam.coverage import (
    FREQ_LIMIT,
    MIN_NEW,
    N_GRAM,
    DocFrequency,
    TrainDoc,
    Verdict,
    WDoc,
    WGramIndex,
    covered_new_tokens,
    decontaminate,
    hashes_of_interest,
    is_contaminated,
    judge,
)
from artigo_env.corpus.decontam.grams import fold_accents, gram_hashes, gram_tokens
from artigo_env.corpus.decontam.minhash import (
    JACCARD_MIN,
    LSHIndex,
    NearDuplicateIndex,
    jaccard,
    shingle_hashes,
    signature,
)
from artigo_env.corpus.decontam.titles import TitleBarrier, normalize_title
from artigo_env.corpus.decontam.windex import WIndex

__all__ = [
    "FREQ_LIMIT",
    "JACCARD_MIN",
    "MIN_NEW",
    "N_GRAM",
    "DocFrequency",
    "LSHIndex",
    "NearDuplicateIndex",
    "TitleBarrier",
    "TrainDoc",
    "Verdict",
    "WDoc",
    "WGramIndex",
    "WIndex",
    "covered_new_tokens",
    "decontaminate",
    "fold_accents",
    "gram_hashes",
    "gram_tokens",
    "hashes_of_interest",
    "is_contaminated",
    "jaccard",
    "judge",
    "normalize_title",
    "shingle_hashes",
    "signature",
]
