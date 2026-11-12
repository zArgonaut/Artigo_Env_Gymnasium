"""Per-document pipeline stages S2-S5, S8 and S9 (SDD 06 §3).

Each stage takes and returns a ``DOC_SCHEMA`` record batch, keeps row order, fills only its own
columns and reports removals with the reasons listed under its name in ``REASONS``.
"""

from __future__ import annotations

from artigo_env.corpus.stages.annotate import AnnotateStage
from artigo_env.corpus.stages.context import BuildContext
from artigo_env.corpus.stages.normalize import Cleaner, NormalizeStage
from artigo_env.corpus.stages.origin import OriginStage, load_blocklist
from artigo_env.corpus.stages.partition import PartitionStage, split_of
from artigo_env.corpus.stages.playable import PlayableStage
from artigo_env.corpus.stages.tokenize import TokenizeStage

__all__ = [
    "AnnotateStage",
    "BuildContext",
    "Cleaner",
    "NormalizeStage",
    "OriginStage",
    "PartitionStage",
    "PlayableStage",
    "TokenizeStage",
    "load_blocklist",
    "split_of",
]
