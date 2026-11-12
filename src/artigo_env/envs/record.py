"""The evaluation record of one finished episode (SDD 09 §6, DEC-070).

``ArtigoCoreEnv.episode_record()`` is the only door to hidden data: it returns an
``EpisodeRecord`` (article identity, source, domain, full text, trajectory and final state) and
raises ``EpisodeNotFinishedError`` while the episode is running. Only the evaluation harness may
call it; agents never do (architecture test).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

from artigo_env.core.errors import ArtigoEnvError


class EpisodeNotFinishedError(ArtigoEnvError, RuntimeError):
    """``episode_record()`` was called before the episode terminated or was truncated."""


@dataclass(frozen=True)
class EpisodeRecord:
    """Everything about one finished episode, hidden data included.

    ``actions``, ``outcomes``, ``counts`` and ``counted`` have one entry per ``step`` call;
    ``outcomes`` holds ``hit``, ``miss``, ``noop_stopword``, ``noop_repeat`` or ``invalid``
    (an action outside the mask in tolerant mode, which counts as a guess). ``title_oov`` is
    true when a lemma required for victory is outside V (the episode cannot be won through the
    discrete interface). ``final_state`` is ``Engine.serialize`` of the last state.
    """

    article_ref: str
    doc_idx: int
    source_id: str
    domain: str | None
    split: str
    title: str
    body: str
    n_title_content: int
    n_body_tokens: int
    title_oov: bool
    actions: tuple[int, ...]
    outcomes: tuple[str, ...]
    counts: tuple[int, ...]
    counted: tuple[bool, ...]
    n_counted: int
    won: bool
    truncated: bool
    final_state: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Plain dictionary (lists instead of tuples), ready for JSON or Parquet."""
        data = asdict(self)
        for key in ("actions", "outcomes", "counts", "counted"):
            data[key] = list(data[key])
        data["final_state"] = dict(self.final_state)
        return data
