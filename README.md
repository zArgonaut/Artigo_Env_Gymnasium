# ArtigoEnv

A [Gymnasium](https://gymnasium.farama.org/) environment that reproduces, offline, the Portuguese
word-reveal game [artigo.app](https://artigo.app/): an encyclopedia-style article is shown with
every content word hidden, each guess reveals all occurrences of that word, and the episode ends
when every content word of the **title** is revealed. The goal is to do it in as few guesses as
possible.

ArtigoEnv is a research platform for Reinforcement Learning (UFRRJ, 2026.2). It is under active
development; the section [Status](#status) says what works today.

| | |
|---|---|
| Environment id | `ArtigoEnv-v0` |
| Action space | `Discrete(\|V\|)` — one action per lemma of a fixed vocabulary, with an action mask |
| Observation space | `Dict` of fixed-shape integer arrays (see below) |
| Reward | `-1` for every counted guess, `0` otherwise |
| Termination | every content word of the title is revealed |
| Truncation | 400 steps |

## Installation

The project uses [uv](https://docs.astral.sh/uv/) and Python 3.12.

```bash
git clone https://github.com/zArgonaut/Artigo_Env_Gynmasium.git
cd Artigo_Env_Gynmasium
uv sync
```

Nothing is downloaded at install time. A small demonstration corpus (30 synthetic articles
written for this project) ships inside the package, so the environment works right away.

## Quick start

```python
import gymnasium as gym
import numpy as np

import artigo_env  # registers ArtigoEnv-v0

env = gym.make("ArtigoEnv-v0")
obs, info = env.reset(seed=0)

rng = np.random.default_rng(0)
terminated = truncated = False
while not (terminated or truncated):
    mask = env.unwrapped.action_masks()  # False for lemmas already guessed
    action = int(rng.choice(np.flatnonzero(mask)))
    obs, reward, terminated, truncated, info = env.step(action)
```

To play as a human in the terminal:

```bash
uv run artigo-env play --title "Rio Taquirama" --text-file my_article.txt --ruleset no-grouping
```

## The game

- A **token** is a maximal run of letters and digits; hyphens, apostrophes and punctuation
  separate tokens and are never hidden.
- **Stopwords** (a fixed list of 227 function words) are visible from the start and guessing one
  does not count.
- A guess is lower-cased and stripped of everything that is not a letter or a digit, then mapped
  to a **lemma**: guessing `anos` also reveals `ano`; guessing `nasceu` reveals `nascer`,
  `nasceram`, `nascido`.
- A **numeric** guess is searched as a substring of every token that contains a digit and reveals
  only the matching characters (`1` reveals the first character of `1822`).
- A guess with no occurrence still counts; repeating a guess does not.

## Observation

`L` is the maximum body length of the corpus build, `Tt = 32` title tokens, `T = 400` steps.

| Key | Shape | Content |
|---|---|---|
| `body_lemma`, `title_lemma` | `(L,)`, `(Tt,)` | `-3` padding, `-2` hidden, `-1` visible but outside the vocabulary, `0..\|V\|-1` lemma id, `\|V\|..` stopword |
| `body_form`, `title_form` | `(L,)`, `(Tt,)` | same coding with ids of the form vocabulary |
| `body_length`, `title_length` | `(L,)`, `(Tt,)` | length in characters; `0` for padding and for hidden tokens when the ruleset hides lengths |
| `body_flags`, `title_flags` | `(L,)`, `(Tt,)` | bit 0 header, bit 1 paragraph break, bit 2 partly revealed, bit 3 stopword, bit 4 visible, bit 5 inside parentheses (title) |
| `n_tokens` | `(2,)` | number of body and title tokens |
| `history` | `(T, 3)` | per guess: action id, number of occurrences, outcome code |
| `step` | `(1,)` | guesses made so far |

Nothing about hidden words reaches the observation, the action mask or `info`. `info` only carries
the outcome of the last guess, the title progress, non-numeric screen data and configuration
identifiers. Full episode data (the article and the trajectory) is available through
`env.unwrapped.episode_record()` **after** the episode ends, for evaluation code only.

## Rules and scenarios

Rules are declarative **rulesets**; each lever is a registered component.

| Preset | Difference from `canonical` |
|---|---|
| `canonical` | the rules of the original game, with the open lemmatizer |
| `no-grouping` | no lemmatization: each form is its own guess |
| `no-lengths` | hidden words do not show their length |
| `strict-noops` | stopword and repeated guesses count |
| `hard` | the three above together |

Scenarios choose which articles are drawn: `demo`, `train-default`, `train-curriculum`,
`eval-w-val`, `eval-w-test`, `eval-ext-val`. Training scenarios sample without replacement inside
an epoch and are reproducible from the seed.

```python
env = gym.make("ArtigoEnv-v0", corpus="demo", ruleset="no-lengths", scenario="demo")
```

## Corpora

Real corpora are built locally from sources declared in `src/artigo_env/assets/sources.yaml`;
no corpus text is redistributed with this repository. Set `ARTIGO_DATA_ROOT` (copy `.env.example`
to `.env`) and use the command line:

```bash
uv run --env-file .env artigo-env data fetch <source>          # download with SHA-256 check
uv run --env-file .env artigo-env data build configs/corpus/<name>.yaml
uv run --env-file .env artigo-env data report <build_id>
uv run --env-file .env artigo-env vocab build --corpus <build_id> --ruleset canonical
```

The build pipeline filters by origin, normalizes text, keeps only playable documents (title with
one to six content words, all of them present in the body), removes anything that overlaps the
evaluation set (titles, shared 13-grams, near duplicates), deduplicates and compiles the result
into memory-mapped arrays. Builds are reproducible byte for byte.

The evaluation set comes from Portuguese Wikipedia; training text never does.

## Tests

```bash
uv run pytest -m fast        # quick suite, no network or local data needed
uv run pytest -m slow        # browser, subprocess and multi-process tests
uv run ruff check . && uv run mypy && uv run lint-imports
```

The suite covers the game rules, determinism, the Gymnasium contract (`check_env` on every
ruleset preset) and leakage: permutation tests check that two articles differing only in hidden
words produce identical observations, masks and `info`.

## Repository layout

```
src/artigo_env/   core, text, engine, corpus, scenarios, envs, cli (+ assets/ with packaged data)
configs/          corpus build configurations
tools/            recorder and calibration, lemmatizer tables, evaluation-set snapshot, demo build
tests/            unit, contract, determinism, leakage, fidelity, integration
```

Layers only import layers to their left: `core → text → engine → {corpus | scenarios} → services →
envs → wrappers → agents → evaluation → analysis`. The contract is checked on every commit.

## Status

Working today: text core and lemmatizers, game engine and rulesets, corpus pipeline and compiled
store, scenarios, action vocabulary, `ArtigoEnv-v0` with the demonstration corpus, terminal play.

In progress: reference agents and the evaluation harness, a one-command self-test, observation
services and reward wrappers, experiment orchestration, a free-text environment.

## License

Code: MIT. Lexical resources derived from third parties are listed in `THIRD_PARTY_NOTICES.md`.
Data sources keep their own licenses and are never redistributed.
