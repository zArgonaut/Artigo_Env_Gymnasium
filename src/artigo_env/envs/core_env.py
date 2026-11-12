"""``ArtigoCoreEnv`` (``ArtigoEnv-v0``): the game as a Gymnasium environment (SDD 09).

Action ``a`` guesses key ``a`` of the vocabulary V: a lemma (a ``LexicalGuess`` of the build's
lemma with that text, a counted miss when the build has no such lemma) or ``n:<digits>`` (a
``NumericGuess``). The observation is a ``Dict`` of fixed-size arrays (``observation.py``); the
mask forbids keys already guessed; ``info`` holds only allow-listed keys (``INFO_KEYS``); hidden
data leaves the environment only through ``episode_record()`` after the episode ends.

Name resolution:

- ``corpus``: ``"demo"`` (the build packaged in ``artigo_env/assets/demo/corpus``), a directory
  holding a compiled build, or a build id under ``ARTIGO_DATA_ROOT/corpus/``;
- ``ruleset`` / ``scenario``: a preset name or a ``Ruleset`` / ``Scenario`` object;
- ``vocab``: ``None`` (the vocabulary of (build, ruleset): for ``demo`` the packaged one, else
  the only directory under ``ARTIGO_DATA_ROOT/vocab/`` whose manifest has this ``build_id`` and
  this ruleset hash), a directory, or a vocabulary id under ``ARTIGO_DATA_ROOT/vocab/``.

Seeding: ``reset(seed=s)`` seeds ``np_random`` with ``s`` and rebuilds the article sampler with
the shared base seed ``s - worker_index`` (vectorized environments pass ``seed + i`` to worker
``i``, so every worker derives the same epoch permutations and takes a disjoint stride of them);
the curriculum stage is kept. ``reset()`` without a seed continues the current sequence; on the
very first reset without a seed the base seed is drawn from the (randomly seeded) ``np_random``.
The generator only picks the source of each draw; the permutations depend on the base seed.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any, SupportsFloat

import gymnasium as gym
import numpy as np
import numpy.typing as npt
from gymnasium import spaces
from gymnasium.error import ResetNeeded

from artigo_env.core.errors import ArtigoEnvError, InvalidActionError
from artigo_env.core.paths import data_path
from artigo_env.core.resources import resource_path
from artigo_env.core.vocab import MANIFEST_FILE, NUMERIC_PREFIX, Vocabulary
from artigo_env.corpus.store import CorpusStore, StoreLexicon
from artigo_env.corpus.vocab import StoreVocabMap, map_store
from artigo_env.engine.engine import Engine
from artigo_env.engine.rulesets import Ruleset, get_ruleset
from artigo_env.engine.rulesets.components import LENGTH_RULES, TITLE_RULES
from artigo_env.engine.types import (
    Article,
    GameState,
    Guess,
    LexicalGuess,
    NumericGuess,
    TitleProgress,
)
from artigo_env.envs.observation import (
    DEFAULT_BODY_TOKENS,
    OUTCOME_CODES,
    TITLE_TOKENS,
    EpisodeArrays,
    ObservationSizeError,
    encode,
    observation_space,
    prepare,
    screen_info,
)
from artigo_env.envs.record import EpisodeNotFinishedError, EpisodeRecord
from artigo_env.envs.registration import ENV_ID, MAX_EPISODE_STEPS
from artigo_env.envs.render import render_ansi
from artigo_env.scenarios import (
    DocColumns,
    FixedListSampler,
    Sampler,
    Scenario,
    apply_filters,
    get_scenario,
    make_sampler,
)
from artigo_env.scenarios.sampler import N_TITLE_BUCKETS

ENV_VERSION = ENV_ID
DEMO_CORPUS = "demo"
DEFAULT_MAX_STEPS = MAX_EPISODE_STEPS  # T_max ([LOCAL-TMAX-1]); same as the TimeLimit

# Allow-list of info keys (SDD 09 §6). ``reward_components`` is added by the reward wrapper and
# ``action_mask`` only with ``mask_in_info=True`` (SDD 09 §4).
INFO_KEYS = frozenset(
    {
        "outcome",
        "count",
        "counted",
        "newly_revealed",
        "title_progress",
        "screen",
        "invalid_action",
        "curriculum_stage",
        "ruleset",
        "corpus_build",
        "env_version",
        "reward_components",
        "action_mask",
    }
)
RESET_OPTIONS = frozenset({"article_index", "curriculum_stage"})

_DOC_COLUMNS_STR = ("source_id", "split", "domain", "variant")
_DOC_COLUMNS_INT = ("n_body_tokens", "n_title_content", "n_title_tokens")
_DOC_COLUMNS_BOOL = ("title_in_body",)

Obs = dict[str, npt.NDArray[Any]]


# -- name resolution ---------------------------------------------------------------------------


def _demo_dir(*parts: str) -> Path:
    path = Path(str(resource_path("demo", *parts)))
    if not path.is_dir():
        raise ArtigoEnvError(
            f"the packaged demo data is missing ({path}); rebuild it with "
            "'python -m tools.demo.build_demo' from a source checkout"
        )
    return path


def resolve_corpus(corpus: str | os.PathLike[str]) -> Path:
    """Directory of the compiled build named by ``corpus`` (see the module docstring)."""
    if isinstance(corpus, str) and corpus == DEMO_CORPUS:
        return _demo_dir("corpus")
    path = Path(corpus)
    if (path / MANIFEST_FILE).is_file():
        return path
    if isinstance(corpus, str) and path.name == corpus:  # a bare build id
        candidate = data_path("corpus", corpus)
        if (candidate / MANIFEST_FILE).is_file():
            return candidate
        raise ArtigoEnvError(f"no compiled corpus {corpus!r} under {candidate.parent}")
    raise ArtigoEnvError(f"not a compiled corpus directory (no {MANIFEST_FILE}): {path}")


def resolve_ruleset(ruleset: str | Ruleset) -> Ruleset:
    if isinstance(ruleset, Ruleset):
        return ruleset
    if isinstance(ruleset, str):
        return get_ruleset(ruleset)
    raise TypeError(f"ruleset must be a preset name or a Ruleset, got {type(ruleset).__name__}")


def resolve_scenario(scenario: str | Scenario) -> Scenario:
    if isinstance(scenario, Scenario):
        return scenario
    if isinstance(scenario, str):
        return get_scenario(scenario)
    raise TypeError(f"scenario must be a preset name or a Scenario, got {type(scenario).__name__}")


def _matching_vocabs(root: Path, build_id: str, ruleset_hash: str) -> list[Path]:
    found: list[Path] = []
    if not root.is_dir():
        return found
    for manifest in sorted(root.glob(f"*/{MANIFEST_FILE}")):
        try:
            data = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if data.get("build_id") == build_id and data.get("ruleset_hash") == ruleset_hash:
            found.append(manifest.parent)
    return found


def resolve_vocab(
    vocab: str | os.PathLike[str] | None, corpus: Any, build_id: str, ruleset: Ruleset
) -> Path:
    """Directory of the vocabulary named by ``vocab`` (see the module docstring)."""
    if vocab is not None:
        path = Path(vocab)
        if (path / MANIFEST_FILE).is_file():
            return path
        if isinstance(vocab, str) and path.name == vocab:
            candidate = data_path("vocab", vocab)
            if (candidate / MANIFEST_FILE).is_file():
                return candidate
            raise ArtigoEnvError(f"no vocabulary {vocab!r} under {candidate.parent}")
        raise ArtigoEnvError(f"not a vocabulary directory (no {MANIFEST_FILE}): {path}")
    ruleset_hash = ruleset.content_hash()
    if isinstance(corpus, str) and corpus == DEMO_CORPUS:
        root = _demo_dir("vocab")
        where = "the packaged demo vocabularies"
    else:
        root = data_path("vocab")
        where = str(root)
    found = _matching_vocabs(root, build_id, ruleset_hash)
    if len(found) == 1:
        return found[0]
    if not found:
        raise ArtigoEnvError(
            f"no vocabulary for build {build_id!r} and ruleset {ruleset.name!r} "
            f"({ruleset_hash[:16]}) in {where}; build one with 'artigo-env vocab build' or pass "
            "vocab=<id or path> (evaluation builds use the vocabulary of the training build)"
        )
    names = ", ".join(p.name for p in found)
    raise ArtigoEnvError(
        f"several vocabularies for build {build_id!r} and ruleset {ruleset.name!r}: {names}; "
        "pass vocab=<id or path>"
    )


def _load_map(vocab: Vocabulary, store: CorpusStore, ruleset: Ruleset) -> StoreVocabMap:
    try:
        return map_store(vocab, store, ruleset, cache=True)
    except OSError:  # read-only vocabulary directory without a cached map
        return map_store(vocab, store, ruleset, cache=False)


def doc_columns(store: CorpusStore) -> tuple[DocColumns, npt.NDArray[np.int64]]:
    """Scenario metadata of every document and the title token counts (nulls filled)."""
    table = store.docs
    data: dict[str, Any] = {}
    for name in _DOC_COLUMNS_STR:
        values = table.column(name).to_pylist() if name in table.column_names else []
        data[name] = ["" if v is None else str(v) for v in values] or [""] * store.n_docs
    for name in _DOC_COLUMNS_INT:
        values = table.column(name).to_pylist() if name in table.column_names else []
        data[name] = np.asarray(
            [0 if v is None else int(v) for v in values] or [0] * store.n_docs, dtype=np.int64
        )
    for name in _DOC_COLUMNS_BOOL:
        values = table.column(name).to_pylist() if name in table.column_names else []
        data[name] = np.asarray([bool(v) for v in values] or [False] * store.n_docs, dtype=np.bool_)
    return DocColumns.from_mapping(data), data["n_title_tokens"]


# -- environment -------------------------------------------------------------------------------


class ArtigoCoreEnv(gym.Env[Obs, np.int64]):
    """Discrete-action environment over the action vocabulary V (SDD 09, DEC-063)."""

    metadata: dict[str, Any] = {"render_modes": ["ansi"], "render_fps": 4}

    def __init__(
        self,
        corpus: str | os.PathLike[str] = DEMO_CORPUS,
        ruleset: str | Ruleset = "canonical",
        scenario: str | Scenario = "demo",
        vocab: str | os.PathLike[str] | None = None,
        strict_actions: bool = True,
        mask_noops: bool = True,
        mask_in_info: bool = False,
        render_mode: str | None = None,
        worker_index: int = 0,
        num_workers: int = 1,
        max_body_tokens: int | None = None,
        max_steps: int = DEFAULT_MAX_STEPS,
    ) -> None:
        if render_mode == "rgb_array":
            raise NotImplementedError(
                "render_mode='rgb_array' is not available yet (it needs the category-1 "
                "services of a later phase); use 'ansi' or None"
            )
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"unknown render_mode {render_mode!r}; supported: 'ansi' or None")
        if isinstance(num_workers, bool) or not isinstance(num_workers, int) or num_workers < 1:
            raise ValueError(f"num_workers must be a positive integer, got {num_workers!r}")
        if (
            isinstance(worker_index, bool)
            or not isinstance(worker_index, int)
            or not 0 <= worker_index < num_workers
        ):
            raise ValueError(f"worker_index must be in [0, {num_workers}), got {worker_index!r}")
        if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps < 1:
            raise ValueError(f"max_steps must be a positive integer, got {max_steps!r}")

        self.render_mode = render_mode
        self.strict_actions = bool(strict_actions)
        self.mask_noops = bool(mask_noops)
        self.mask_in_info = bool(mask_in_info)
        self.worker_index = worker_index
        self.num_workers = num_workers
        self.max_steps = max_steps
        self.ruleset = resolve_ruleset(ruleset)
        self.scenario = resolve_scenario(scenario)

        self._store: CorpusStore | None = None
        self._vocab: Vocabulary | None = None
        store = CorpusStore.open(resolve_corpus(corpus))
        self._store = store
        try:
            self._setup(corpus, vocab, max_body_tokens)
        except BaseException:
            self.close()
            raise

    def _setup(
        self,
        corpus: Any,
        vocab: str | os.PathLike[str] | None,
        max_body_tokens: int | None,
    ) -> None:
        store = self._require_store()
        ruleset = self.ruleset
        self.corpus_build = store.build_id
        self._vocab = Vocabulary.open(resolve_vocab(vocab, corpus, store.build_id, ruleset))
        vocabulary = self._vocab
        self.vocab_id = vocabulary.vocab_id
        maps = _load_map(vocabulary, store, ruleset)
        self._lemma_obs = maps.lemma_obs
        self._form_obs = maps.form_obs
        self._action_lemma = np.asarray(maps.action_lemma, dtype=np.int64)
        self._is_numeric = np.array(vocabulary.is_numeric, dtype=np.bool_)
        self._engine = Engine(ruleset, StoreLexicon(store, store.lemma_view(ruleset.lemmatizer)))
        lengths = LENGTH_RULES.get(ruleset.lengths_visible)
        self._show_title_lengths = bool(lengths.visible(True))
        self._show_body_lengths = bool(lengths.visible(False))
        self._title_rule = TITLE_RULES.get(ruleset.title_parenthetical)

        if max_body_tokens is None:
            l_max = store.manifest.get("l_max")
            body_tokens = int(l_max) if l_max is not None else DEFAULT_BODY_TOKENS
        else:
            body_tokens = int(max_body_tokens)
        if body_tokens < 1:
            raise ValueError(f"max_body_tokens must be positive, got {max_body_tokens!r}")
        self.body_tokens = body_tokens
        self.title_tokens = TITLE_TOKENS

        self._columns, n_title_tokens = doc_columns(store)
        self._stage_masks = self._eligibility(self._columns)
        self._check_sizes(self._columns, n_title_tokens)
        self._target_shares = self._title_shares(self._columns)

        self.action_space = spaces.Discrete(vocabulary.size)
        self.observation_space: spaces.Dict = observation_space(  # type: ignore[assignment]
            n_actions=vocabulary.size,
            n_stop=vocabulary.n_stop,
            n_forms=vocabulary.n_forms,
            body_tokens=body_tokens,
            history_rows=self.max_steps,
        )

        self._sampler: Sampler | None = None
        self._stage = 0
        self._guessed = np.zeros(vocabulary.size, dtype=np.bool_)
        self._history = np.full((self.max_steps, 3), -1, dtype=np.int32)
        self._doc_idx = -1
        self._article: Article | None = None
        self._episode: EpisodeArrays | None = None
        self._state: GameState | None = None
        self._progress: TitleProgress | None = None
        self._n_steps = 0
        self._n_invalid = 0
        self._outcomes: list[str] = []
        self._counted: list[bool] = []
        self._terminated = False
        self._truncated = False
        self._last_message: str | None = None

    # -- construction helpers -----------------------------------------------------------------

    def _require_store(self) -> CorpusStore:
        if self._store is None:
            raise ArtigoEnvError("the environment is closed")
        return self._store

    def _require_vocab(self) -> Vocabulary:
        if self._vocab is None:
            raise ArtigoEnvError("the environment is closed")
        return self._vocab

    def _eligibility(self, columns: DocColumns) -> list[npt.NDArray[np.bool_]]:
        scenario = self.scenario
        base = (columns.split == scenario.split) & apply_filters(scenario.filters, columns)
        return [base & apply_filters(stage.filters, columns) for stage in scenario.stages]

    def _check_sizes(self, columns: DocColumns, n_title_tokens: npt.NDArray[np.int64]) -> None:
        """Every document the scenario can draw must fit the observation (REQ-ENV-008)."""
        eligible = np.any(np.stack(self._stage_masks), axis=0)
        too_long = eligible & (columns.n_body_tokens > self.body_tokens)
        if too_long.any():
            worst = int(np.max(columns.n_body_tokens[too_long]))
            raise ObservationSizeError(
                f"{int(too_long.sum())} document(s) of scenario {self.scenario.name!r} have more "
                f"than L={self.body_tokens} body tokens (largest: {worst}); texts are never "
                "truncated: raise max_body_tokens or filter them out"
            )
        too_wide = eligible & (n_title_tokens > self.title_tokens)
        if too_wide.any():
            worst = int(np.max(n_title_tokens[too_wide]))
            raise ObservationSizeError(
                f"{int(too_wide.sum())} document(s) of scenario {self.scenario.name!r} have more "
                f"than Tt={self.title_tokens} title tokens (largest: {worst})"
            )

    def _title_shares(self, columns: DocColumns) -> list[float] | None:
        """Title-length shares of the ``w-pool`` split, needed by ``match-w`` scenarios."""
        if self.scenario.title_distribution != "match-w" or self.scenario.is_evaluation:
            return None
        pool = columns.split == "w-pool"
        if not pool.any():
            raise ArtigoEnvError(
                f"scenario {self.scenario.name!r} uses title_distribution='match-w', which needs "
                f"the 'w-pool' split in the corpus build {self.corpus_build!r}; it has none"
            )
        buckets = np.clip(columns.n_title_content[pool], 1, N_TITLE_BUCKETS) - 1
        counts = np.bincount(buckets, minlength=N_TITLE_BUCKETS)
        return [float(c) for c in counts / counts.sum()]

    def _new_sampler(self, base_seed: int) -> Sampler:
        sampler = make_sampler(
            self._columns,
            self.scenario,
            base_seed=base_seed,
            worker_index=self.worker_index,
            num_workers=self.num_workers,
            target_title_shares=self._target_shares,
        )
        if self._stage:
            sampler.set_stage(self._stage)
        return sampler

    # -- Gymnasium API ------------------------------------------------------------------------

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Obs, dict[str, Any]]:
        super().reset(seed=seed)
        store = self._require_store()
        options = dict(options or {})
        unknown = set(options) - RESET_OPTIONS
        if unknown:
            raise ValueError(
                f"unknown reset options {sorted(unknown)}; accepted: {sorted(RESET_OPTIONS)}"
            )
        if seed is not None or self._sampler is None:
            if seed is not None:
                base_seed = int(seed) - self.worker_index
                if base_seed < 0:
                    raise ValueError(
                        f"reset(seed={seed}) on worker {self.worker_index}: the shared sampler "
                        "seed is seed - worker_index and must be >= 0 (vectorized environments "
                        "pass seed + worker_index to each worker)"
                    )
            else:
                base_seed = int(self.np_random.integers(0, 2**31 - 1))
            self._sampler = self._new_sampler(base_seed)
        sampler = self._sampler
        if "curriculum_stage" in options:
            self.set_curriculum_stage(options["curriculum_stage"])
        if "article_index" in options:
            doc_idx = self._article_at(options["article_index"])
        else:
            doc_idx = int(sampler.next(self.np_random))

        article = store.article(doc_idx, self.ruleset)
        self._doc_idx = doc_idx
        self._article = article
        self._episode = prepare(
            article,
            self._lemma_obs,
            self._form_obs,
            body_tokens=self.body_tokens,
            title_tokens=self.title_tokens,
            show_title_lengths=self._show_title_lengths,
            show_body_lengths=self._show_body_lengths,
        )
        self._state = self._engine.new_game(article)
        required = self._title_rule.required_mask(article.title)
        letters = int(article.title.lengths[required].sum())
        self._progress = TitleProgress(0, int(required.sum()), 0, letters)
        self._guessed[:] = False
        self._history[:] = -1
        self._n_steps = 0
        self._n_invalid = 0
        self._outcomes = []
        self._counted = []
        self._terminated = False
        self._truncated = False
        self._last_message = None
        return self._observe(None)

    def step(self, action: Any) -> tuple[Obs, SupportsFloat, bool, bool, dict[str, Any]]:
        state, article = self._state, self._article
        if state is None or article is None:
            raise ResetNeeded("call reset() before step()")
        if self._terminated or self._truncated:
            raise InvalidActionError("the episode is over; call reset()")
        a = self._action_id(action)
        valid = not (self.mask_noops and self._guessed[a])
        if valid:
            state, outcome = self._engine.apply(state, article, self._guess(a))
            self._state = state
            self._progress = outcome.title_progress
            self._guessed[a] = True
            kind, counted, count = outcome.kind.value, outcome.counted, outcome.count
            fresh = set(outcome.newly_revealed.tolist()) | set(outcome.newly_partial)
            newly = sorted(int(p) for p in fresh)
            won = outcome.won
        else:
            if self.strict_actions:
                raise InvalidActionError(
                    f"action {a} ({self._key(a)!r}) is masked: it was already guessed"
                )
            kind, counted, count, newly, won = "invalid", True, 0, [], False
            self._n_invalid += 1

        self._history[self._n_steps] = (a, count, OUTCOME_CODES[kind])
        self._n_steps += 1
        self._outcomes.append(kind)
        self._counted.append(counted)
        self._terminated = bool(won)
        self._truncated = not self._terminated and self._n_steps >= self.max_steps
        if self._sampler is not None:
            self._sampler.notify_steps(1)
        self._last_message = f"{self._key(a)}: {kind}" + (f" ({count})" if count else "")
        reward = -1.0 if counted else 0.0
        obs, info = self._observe(
            {
                "outcome": kind,
                "count": int(count),
                "counted": bool(counted),
                "newly_revealed": newly,
                "invalid_action": not valid,
            }
        )
        return obs, reward, self._terminated, self._truncated, info

    def action_masks(self) -> npt.NDArray[np.bool_]:
        """``(|V|,)`` booleans: ``False`` for keys already guessed (SDD 09 §4)."""
        if not self.mask_noops:
            return np.ones(len(self._guessed), dtype=np.bool_)
        return np.logical_not(self._guessed)

    def set_curriculum_stage(self, k: int) -> None:
        """Switch the scenario's curriculum stage (used by the runner through ``env_method``)."""
        n_stages = len(self.scenario.stages)
        if isinstance(k, bool) or not isinstance(k, int | np.integer) or not 0 <= k < n_stages:
            raise ValueError(f"curriculum stage must be in [0, {n_stages}), got {k!r}")
        self._stage = int(k)
        if self._sampler is not None:
            self._sampler.set_stage(self._stage)

    @property
    def curriculum_stage(self) -> int:
        return self._sampler.stage if self._sampler is not None else self._stage

    def episode_record(self) -> EpisodeRecord:
        """Hidden data of the finished episode (SDD 09 §6); raises while it is running."""
        state, article = self._state, self._article
        if state is None or article is None or not (self._terminated or self._truncated):
            raise EpisodeNotFinishedError(
                "episode_record() is available only after the episode terminated or was truncated"
            )
        store = self._require_store()
        vocabulary = self._require_vocab()
        row = store.doc(self._doc_idx)
        required = self._title_rule.required_mask(article.title)
        title_ids = self._lemma_obs[article.title.forms.astype(np.int64)]
        title_oov = bool(((title_ids < 0) | (title_ids >= vocabulary.size))[required].any())
        n = self._n_steps
        return EpisodeRecord(
            article_ref=article.article_ref,
            doc_idx=int(self._doc_idx),
            source_id=str(row.get("source_id") or ""),
            domain=row.get("domain"),
            split=str(row.get("split") or ""),
            title=store.title_text(self._doc_idx),
            body=store.body_text(self._doc_idx),
            n_title_content=int(required.sum()),
            n_body_tokens=len(article.body),
            title_oov=title_oov,
            actions=tuple(int(x) for x in self._history[:n, 0]),
            outcomes=tuple(self._outcomes),
            counts=tuple(int(x) for x in self._history[:n, 1]),
            counted=tuple(self._counted),
            n_counted=state.n_counted + self._n_invalid,
            won=state.status == "won",
            truncated=self._truncated,
            final_state=self._engine.serialize(state),
        )

    def render(self) -> str | None:
        if self.render_mode is None:
            gym.logger.warn("render() called without a render_mode; pass render_mode='ansi'")
            return None
        state, article = self._state, self._article
        if state is None or article is None:
            raise ResetNeeded("call reset() before render()")
        return render_ansi(self._engine.screen(state, article), self._last_message)

    def close(self) -> None:
        if self._vocab is not None:
            self._vocab.close()
            self._vocab = None
        if self._store is not None:
            self._store.close()
            self._store = None
        super().close()

    # -- internals ----------------------------------------------------------------------------

    def _article_at(self, article_index: Any) -> int:
        sampler = self._sampler
        if isinstance(sampler, FixedListSampler):
            return sampler.index_of(article_index)
        if isinstance(article_index, bool) or not isinstance(article_index, int | np.integer):
            raise TypeError(f"article_index must be an integer, got {article_index!r}")
        eligible = np.flatnonzero(self._stage_masks[self.curriculum_stage])
        if not 0 <= article_index < len(eligible):
            raise IndexError(
                f"article_index {article_index} out of range for scenario "
                f"{self.scenario.name!r} (valid: 0..{len(eligible) - 1})"
            )
        return int(eligible[int(article_index)])

    def _action_id(self, action: Any) -> int:
        try:
            value = np.asarray(action)
            if value.size != 1 or value.dtype.kind not in "iu":
                raise TypeError
            a = int(value.reshape(()))
        except (TypeError, ValueError):
            raise InvalidActionError(f"action must be an integer, got {action!r}") from None
        if not 0 <= a < len(self._guessed):
            raise InvalidActionError(f"action {a} outside Discrete({len(self._guessed)})")
        return a

    def _key(self, a: int) -> str:
        return self._require_vocab().key(a)

    def _guess(self, a: int) -> Guess:
        key = self._key(a)
        if self._is_numeric[a]:
            return NumericGuess(key[len(NUMERIC_PREFIX) :])
        lemma_id = int(self._action_lemma[a])
        if lemma_id >= 0:
            return self._engine.guess_from_lemma(lemma_id)
        return LexicalGuess(-1, key, key)  # no form of this build has the lemma: a miss

    def _observe(self, step_info: Mapping[str, Any] | None) -> tuple[Obs, dict[str, Any]]:
        """Observation and info of the current state (one ``screen_arrays`` call)."""
        episode, state, article = self._episode, self._state, self._article
        progress = self._progress
        assert episode is not None and state is not None and article is not None
        assert progress is not None
        screen = self._engine.screen_arrays(state, article)
        obs = encode(episode, screen, self._history, self._n_steps)
        # Letter totals of hidden title words are not on screen when title lengths are hidden.
        letters = progress.letter_fraction if self._show_title_lengths else None
        info: dict[str, Any] = dict(step_info or {})
        info.update(
            title_progress={"tokens": progress.token_fraction, "letters": letters},
            screen=screen_info(episode, screen),
            curriculum_stage=self.curriculum_stage,
            ruleset=self.ruleset.name,
            corpus_build=self.corpus_build,
            env_version=ENV_VERSION,
        )
        if self.mask_in_info:
            info["action_mask"] = self.action_masks()
        return obs, info
