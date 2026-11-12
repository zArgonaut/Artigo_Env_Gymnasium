"""Numeric observation of ``ArtigoCoreEnv`` (SDD 09 §3, DEC-065; decisions 3 and 4 of the F4 plan).

Per episode, ``prepare`` computes the full per-position id vectors once (``lemma_obs`` and
``form_obs`` of every token, from the build-to-vocabulary map). Per step, ``encode`` hides what
the player cannot see with ``np.where`` over ``Engine.screen_arrays``: a hidden position gets
``-2`` whatever its word, so hidden words can only reach the observation through what the
screen itself exposes (lengths when the ruleset shows them, layout flags, partial reveals).
Every call returns new arrays.

Ids: ``-3`` padding; ``-2`` hidden; ``-1`` visible but outside V ∪ S (outside F for the form
arrays); ``0 .. |V|-1`` a lemma of V; ``|V| + i`` the i-th stopword of the ruleset.

Flags: bit0 header, bit1 paragraph break before, bit2 partially revealed, bit3 stopword, bit4
visible; title tokens add bit5 inside parentheses.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
from gymnasium import spaces

from artigo_env.core.errors import ArtigoEnvError
from artigo_env.engine.screen import ScreenArrays, render_partial
from artigo_env.engine.types import (
    FLAG_HEADER,
    FLAG_INSIDE_PARENTHESES,
    FLAG_PARAGRAPH_BREAK,
    Article,
)

TITLE_TOKENS = 32  # Tt
DEFAULT_BODY_TOKENS = 16_384  # provisional L_max ([LOCAL-OBS-1])
MAX_LENGTH = 255

PAD = -3
HIDDEN = -2
OUTSIDE = -1

OBS_HEADER = 1 << 0
OBS_PARAGRAPH_BREAK = 1 << 1
OBS_PARTIAL = 1 << 2
OBS_STOPWORD = 1 << 3
OBS_VISIBLE = 1 << 4
OBS_PARENTHESES = 1 << 5

OUTCOME_CODES: dict[str, int] = {
    "hit": 0,
    "miss": 1,
    "noop_stopword": 2,
    "noop_repeat": 3,
    "invalid": 4,
}
HISTORY_HIGH = 2**31 - 1

_LAYOUT = FLAG_HEADER | FLAG_PARAGRAPH_BREAK  # same bits in the engine and in the observation
assert OBS_HEADER == FLAG_HEADER and OBS_PARAGRAPH_BREAK == FLAG_PARAGRAPH_BREAK


class ObservationSizeError(ArtigoEnvError, ValueError):
    """A document does not fit the fixed observation (texts are never truncated, REQ-ENV-008)."""


def observation_space(
    *,
    n_actions: int,
    n_stop: int,
    n_forms: int,
    body_tokens: int,
    history_rows: int,
    title_tokens: int = TITLE_TOKENS,
) -> spaces.Dict:
    """The ``Dict`` space of SDD 09 §3 for the given sizes."""
    lemma_high = n_actions + n_stop - 1
    form_high = max(n_forms - 1, OUTSIDE)

    def ids(high: int, n: int) -> spaces.Box:
        return spaces.Box(PAD, high, (n,), np.int32)

    return spaces.Dict(
        {
            "body_lemma": ids(lemma_high, body_tokens),
            "body_form": ids(form_high, body_tokens),
            "body_length": spaces.Box(0, MAX_LENGTH, (body_tokens,), np.int16),
            "body_flags": spaces.Box(0, 255, (body_tokens,), np.uint8),
            "title_lemma": ids(lemma_high, title_tokens),
            "title_form": ids(form_high, title_tokens),
            "title_length": spaces.Box(0, MAX_LENGTH, (title_tokens,), np.int16),
            "title_flags": spaces.Box(0, 255, (title_tokens,), np.uint8),
            "n_tokens": spaces.Box(0, body_tokens, (2,), np.int32),
            "history": spaces.Box(-1, max(n_actions, HISTORY_HIGH), (history_rows, 3), np.int32),
            "step": spaces.Box(0, history_rows, (1,), np.int32),
        }
    )


@dataclass(frozen=True, eq=False)
class EpisodeArrays:
    """Per-episode vectors over unified positions (title first, then body).

    ``lemma`` and ``form`` hold the observation id of every token as if it were visible; they
    never leave this object except through ``encode``, which masks hidden positions. ``outside``
    lists the positions whose form is outside F (their spelling goes to ``info["screen"]`` once
    visible).
    """

    article: Article
    n_title: int
    n_body: int
    lemma: npt.NDArray[np.int32]
    form: npt.NDArray[np.int32]
    stop: npt.NDArray[np.bool_]
    outside: npt.NDArray[np.int64]
    body_tokens: int
    title_tokens: int
    show_title_lengths: bool
    show_body_lengths: bool


def check_fits(n_title: int, n_body: int, *, body_tokens: int, title_tokens: int) -> None:
    """Raise ``ObservationSizeError`` when a document does not fit the observation."""
    if n_body > body_tokens:
        raise ObservationSizeError(
            f"the body has {n_body} tokens but the observation holds L={body_tokens}; "
            "texts are never truncated (raise max_body_tokens or exclude the document)"
        )
    if n_title > title_tokens:
        raise ObservationSizeError(
            f"the title has {n_title} tokens but the observation holds Tt={title_tokens}"
        )


def prepare(
    article: Article,
    lemma_obs: npt.NDArray[np.int32],
    form_obs: npt.NDArray[np.int32],
    *,
    body_tokens: int,
    title_tokens: int = TITLE_TOKENS,
    show_title_lengths: bool = True,
    show_body_lengths: bool = True,
) -> EpisodeArrays:
    """Pre-compute the id vectors of ``article`` (``*_obs`` are indexed by build form id)."""
    n_title, n_body = len(article.title), len(article.body)
    check_fits(n_title, n_body, body_tokens=body_tokens, title_tokens=title_tokens)
    forms = np.concatenate([article.title.forms, article.body.forms]).astype(np.int64)
    form = np.asarray(form_obs[forms], dtype=np.int32)
    return EpisodeArrays(
        article=article,
        n_title=n_title,
        n_body=n_body,
        lemma=np.asarray(lemma_obs[forms], dtype=np.int32),
        form=form,
        stop=np.asarray(article.stop, dtype=np.bool_),
        outside=np.flatnonzero(form < 0).astype(np.int64),
        body_tokens=body_tokens,
        title_tokens=title_tokens,
        show_title_lengths=show_title_lengths,
        show_body_lengths=show_body_lengths,
    )


def _padded(values: npt.NDArray[Any], size: int, fill: int, dtype: Any) -> npt.NDArray[Any]:
    out = np.full(size, fill, dtype=dtype)
    out[: len(values)] = values
    return out


def encode(
    episode: EpisodeArrays,
    screen: ScreenArrays,
    history: npt.NDArray[np.int32],
    n_steps: int,
) -> dict[str, npt.NDArray[Any]]:
    """The observation dictionary for the current screen (all arrays are new objects)."""
    visible = screen.visible
    lemma = np.where(visible, episode.lemma, HIDDEN).astype(np.int32)
    form = np.where(visible, episode.form, HIDDEN).astype(np.int32)
    length = np.minimum(screen.length, MAX_LENGTH).astype(np.int16)
    exposed = screen.flags
    flags = (exposed & _LAYOUT).astype(np.uint8)
    flags |= np.where(visible, OBS_VISIBLE, 0).astype(np.uint8)
    flags |= np.where(visible & episode.stop, OBS_STOPWORD, 0).astype(np.uint8)
    if screen.partial:
        partial = np.fromiter(screen.partial.keys(), dtype=np.int64, count=len(screen.partial))
        flags[partial] |= OBS_PARTIAL
    t = episode.n_title
    title_flags = flags[:t] | np.where(
        (exposed[:t] & FLAG_INSIDE_PARENTHESES) != 0, OBS_PARENTHESES, 0
    ).astype(np.uint8)
    L, Tt = episode.body_tokens, episode.title_tokens
    return {
        "body_lemma": _padded(lemma[t:], L, PAD, np.int32),
        "body_form": _padded(form[t:], L, PAD, np.int32),
        "body_length": _padded(length[t:], L, 0, np.int16),
        "body_flags": _padded(flags[t:], L, 0, np.uint8),
        "title_lemma": _padded(lemma[:t], Tt, PAD, np.int32),
        "title_form": _padded(form[:t], Tt, PAD, np.int32),
        "title_length": _padded(length[:t], Tt, 0, np.int16),
        "title_flags": _padded(title_flags, Tt, 0, np.uint8),
        "n_tokens": np.array([episode.n_body, episode.n_title], dtype=np.int32),
        "history": np.array(history, dtype=np.int32, copy=True),
        "step": np.array([n_steps], dtype=np.int32),
    }


def screen_info(episode: EpisodeArrays, screen: ScreenArrays) -> dict[str, dict[int, str]]:
    """Non-numeric screen data (``info["screen"]``), keyed by unified position.

    ``forms``: spelling of every visible token whose form is outside F. ``partial``: the partial
    reveal of hidden tokens as displayed (``█`` for hidden characters, collapsed to one when the
    ruleset hides the length). Both are on screen, so neither leaks anything.
    """
    article = episode.article
    visible = screen.visible
    forms = {int(p): article.surface_at(int(p)) for p in episode.outside if visible[p]}
    partial: dict[int, str] = {}
    for pos in sorted(screen.partial):
        show = episode.show_title_lengths if pos < episode.n_title else episode.show_body_lengths
        partial[int(pos)] = render_partial(article.surface_at(pos), screen.partial[pos], show)
    return {"forms": forms, "partial": partial}
