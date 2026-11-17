"""Self-test checks of the Gymnasium environment, determinism, leakage and the harness."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from artigo_env.selftest.checks import expect, expect_equal
from artigo_env.selftest.registry import Context, check

RANDOM_STEPS = 25


def _env(**kwargs: Any) -> Any:
    from artigo_env.envs.core_env import ArtigoCoreEnv

    return ArtigoCoreEnv(corpus="demo", scenario="demo", **kwargs)


def _random_action(env: Any, rng: np.random.Generator) -> int:
    allowed = np.flatnonzero(env.action_masks())
    return int(rng.choice(allowed))


# -- env ------------------------------------------------------------------------------------------


@check("env", "check_env on every ruleset preset")
def check_env_presets(ctx: Context) -> str:
    from gymnasium.utils.env_checker import check_env

    from artigo_env.engine.rulesets import PRESETS

    names = sorted(PRESETS.names())
    for name in names:
        env = _env(ruleset=name)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                check_env(env, skip_render_check=True)
        finally:
            env.close()
    return ", ".join(names)


@check("env", "observations inside the space")
def observations_in_space(ctx: Context) -> str:
    env = _env()
    try:
        rng = np.random.default_rng(0)
        obs, _ = env.reset(seed=0)
        n = 0
        expect(env.observation_space.contains(obs), "reset observation outside the space")
        for _ in range(RANDOM_STEPS):
            obs, _, terminated, truncated, _ = env.step(_random_action(env, rng))
            n += 1
            expect(env.observation_space.contains(obs), f"observation outside the space at {n}")
            if terminated or truncated:
                break
    finally:
        env.close()
    return f"{n} random masked steps"


@check("env", "info keys within the allow-list")
def info_allow_list(ctx: Context) -> str:
    from artigo_env.envs.core_env import INFO_KEYS

    env = _env(mask_in_info=True)
    try:
        rng = np.random.default_rng(1)
        _, info = env.reset(seed=1)
        seen = set(info)
        for _ in range(RANDOM_STEPS):
            _, _, terminated, truncated, info = env.step(_random_action(env, rng))
            seen |= set(info)
            if terminated or truncated:
                break
    finally:
        env.close()
    extra = sorted(seen - INFO_KEYS)
    expect(not extra, f"info keys outside the allow-list: {extra}")
    return f"{len(seen)} keys seen, all allowed"


@check("env", "episode_record refused before the end")
def record_guard(ctx: Context) -> str:
    from artigo_env.envs.record import EpisodeNotFinishedError

    env = _env(max_steps=3)
    try:
        env.reset(seed=0)
        rng = np.random.default_rng(0)
        env.step(_random_action(env, rng))
        try:
            env.episode_record()
        except EpisodeNotFinishedError:
            pass
        else:
            raise AssertionError("episode_record() answered while the episode was running")
        done = False
        while not done:
            _, _, terminated, truncated, _ = env.step(_random_action(env, rng))
            done = terminated or truncated
        record = env.episode_record()
        expect(record.truncated or record.won, "finished episode neither won nor truncated")
    finally:
        env.close()
    return "refused while running, available after the end"


# -- determinism ----------------------------------------------------------------------------------


def trajectory_digest(seed: int = 3, steps: int = 40) -> str:
    """SHA-256 of a seeded random masked trajectory on the demo corpus (obs, rewards, info)."""
    env = _env()
    digest = hashlib.sha256()
    try:
        rng = np.random.default_rng(seed)
        obs, info = env.reset(seed=seed)
        frames = [(obs, 0.0, info)]
        for _ in range(steps):
            obs, reward, terminated, truncated, info = env.step(_random_action(env, rng))
            frames.append((obs, float(reward), info))
            if terminated or truncated:
                obs, info = env.reset()
                frames.append((obs, 0.0, info))
        for obs, reward, info in frames:
            for key in sorted(obs):
                digest.update(key.encode())
                digest.update(np.ascontiguousarray(obs[key]).tobytes())
            digest.update(repr(reward).encode())
            digest.update(json.dumps(info, sort_keys=True, default=repr).encode())
    finally:
        env.close()
    return digest.hexdigest()


@check("determinism", "same seed, same trajectory")
def same_trajectory(ctx: Context) -> str:
    first, second = trajectory_digest(), trajectory_digest()
    expect_equal(second, first, "trajectory digest of the second run")
    return f"digest {first[:12]}"


@check("determinism", "same trajectory in a subprocess", full_only=True)
def same_trajectory_subprocess(ctx: Context) -> str:
    code = "from artigo_env.selftest import checks_env; print(checks_env.trajectory_digest())"
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=300, check=False
    )
    expect(result.returncode == 0, f"subprocess failed: {result.stderr.strip()[-200:]}")
    expect_equal(result.stdout.strip(), trajectory_digest(), "digest computed in a subprocess")
    return "digests match across processes"


# -- leakage --------------------------------------------------------------------------------------

ARTICLE_A = ("Rato preto", "O rato preto dorme na casa azul.\n\n== Vida ==\nEm 1990 o rato fugiu.")
ARTICLE_B = ("Pato preto", "O pato preto corre na casa anil.\n\n== Vida ==\nEm 1991 o pato fugiu.")
GUESSED_FORMS = ("preto", "casa", "vida", "fugiu")
NUMERIC_KEYS = ("n:9",)


def _tiny_corpus(path: Path, name: str, article: tuple[str, str]) -> Path:
    from artigo_env.corpus.store import CorpusWriter
    from artigo_env.corpus.types import DOC_SCHEMA
    from artigo_env.text.lemmatizers import get_lemmatizer

    row: dict[str, Any] = dict.fromkeys(DOC_SCHEMA.names)
    row.update(
        source_id="synthetic",
        doc_id=name,
        title=article[0],
        body=article[1],
        license_id="MIT",
        provenance="{}",
        domain="synthetic",
        split="demo",
    )
    out = path / name
    writer = CorpusWriter(out)
    writer.add(row)
    for lemmatizer in ("identity", "open-tables"):
        writer.add_lemma_view(get_lemmatizer(lemmatizer))
    writer.finish({"build_id": f"selftest-{name}"})
    return out


def _frames(corpus: Path, vocab: Path, preset: str, actions: list[int]) -> list[Any]:
    from artigo_env.envs.core_env import ArtigoCoreEnv

    env = ArtigoCoreEnv(corpus=str(corpus), vocab=str(vocab), ruleset=preset, mask_in_info=True)
    try:
        obs, info = env.reset(seed=0)
        frames = [_frame(obs, 0.0, info, env.action_masks())]
        for action in actions:
            obs, reward, terminated, _, info = env.step(action)
            expect(not terminated, "the leakage episode ended early")
            frames.append(_frame(obs, float(reward), info, env.action_masks()))
    finally:
        env.close()
    return frames


def _frame(obs: Any, reward: float, info: Any, mask: Any) -> tuple[Any, ...]:
    comparable = {k: v for k, v in info.items() if k != "corpus_build"}  # differs by design
    comparable["action_mask"] = np.asarray(comparable["action_mask"]).tolist()
    arrays = {k: (str(v.dtype), np.ascontiguousarray(v).tobytes()) for k, v in obs.items()}
    return (arrays, reward, repr(comparable), np.asarray(mask).tobytes())


@check("leakage", "hidden words do not reach observation, mask or info")
def hidden_words(ctx: Context) -> str:
    from artigo_env.core.vocab import Vocabulary
    from artigo_env.corpus.store import CorpusStore
    from artigo_env.corpus.vocab import VocabConfig, build_vocab
    from artigo_env.engine.rulesets import PRESETS, get_ruleset
    from artigo_env.text.guess import clean_guess
    from artigo_env.text.lemmatizers import get_lemmatizer

    presets = ["canonical"] if ctx.quick else sorted(PRESETS.names())
    for preset in presets:
        work = ctx.workdir(f"leak-{preset}")
        ruleset = get_ruleset(preset)
        corpus_a = _tiny_corpus(work, "a", ARTICLE_A)
        corpus_b = _tiny_corpus(work, "b", ARTICLE_B)
        with CorpusStore.open(corpus_a) as store:
            vocab = build_vocab(store, ruleset, VocabConfig(numeric_max=10), work / "vocab")
        lemmatizer = get_lemmatizer(ruleset.lemmatizer)
        keys = [lemmatizer.lemma(clean_guess(f)) for f in GUESSED_FORMS] + list(NUMERIC_KEYS)
        with Vocabulary.open(vocab) as vocabulary:
            actions = [vocabulary.action_id(k) for k in keys]
        expect(min(actions) >= 0, f"a guessed key is outside V: {keys}")
        frames_a = _frames(corpus_a, vocab, preset, actions)
        frames_b = _frames(corpus_b, vocab, preset, actions)
        expect_equal(len(frames_b), len(frames_a), "number of frames")
        for step, (fa, fb) in enumerate(zip(frames_a, frames_b, strict=True)):
            for part, label in enumerate(("observation", "reward", "info", "mask")):
                expect(fa[part] == fb[part], f"{preset}: {label} differs at step {step}")
    return f"identical after {len(GUESSED_FORMS) + len(NUMERIC_KEYS)} actions: " + ", ".join(
        presets
    )


# -- eval -----------------------------------------------------------------------------------------


class _FirstAllowed:
    """Trivial agent: the first allowed action of V."""

    def reset(self, obs: Any, info: Any) -> None:
        return None

    def act(self, obs: Any, info: Any, action_mask: Any) -> int:
        return int(np.argmax(action_mask))


@check("eval", "harness on three demo articles")
def harness_demo(ctx: Context) -> str:
    import pyarrow.parquet as pq

    from artigo_env.evaluation import EPISODES_SCHEMA, STEPS_SCHEMA, evaluate
    from artigo_env.evaluation.records import EPISODES_FILE, STEPS_FILE, read_summary

    out = ctx.workdir("eval")
    factory: Callable[[], Any] = lambda: _env(max_steps=60)  # noqa: E731
    result = evaluate(_FirstAllowed(), factory, "demo", [0], limit=3, out_dir=out, resamples=1000)
    episodes = pq.read_table(out / EPISODES_FILE)
    steps = pq.read_table(out / STEPS_FILE)
    expect(episodes.schema.equals(EPISODES_SCHEMA), "episodes.parquet schema differs")
    expect(steps.schema.equals(STEPS_SCHEMA), "steps.parquet schema differs")
    expect_equal(episodes.num_rows, 3, "episode rows")
    expect_equal(steps.num_rows, sum(e.n_steps for e in result.episodes), "step rows")
    summary = read_summary(out)
    for key in ("env_id", "ruleset", "corpus_build", "calibration_version", "code_version"):
        expect(key in summary["identifiers"], f"summary.json lacks identifier {key}")
    n_mean = summary["metrics"]["n_mean"]["value"]
    return f"3 episodes, {steps.num_rows} steps, N mean {n_mean:.1f}"
