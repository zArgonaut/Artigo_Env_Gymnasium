"""Contract of ArtigoCoreEnv with Gymnasium (SDD 09, SDD 15 §1)."""

from __future__ import annotations

import warnings

import gymnasium as gym
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env
from hypothesis import given, settings
from hypothesis import strategies as st

import artigo_env  # noqa: F401  (registers ArtigoEnv-v0)
from artigo_env.corpus.store import CorpusStore, CorpusWriter
from artigo_env.corpus.types import DOC_SCHEMA
from artigo_env.corpus.vocab import VocabConfig, build_vocab
from artigo_env.engine.rulesets import PRESETS, get_ruleset
from artigo_env.envs import ArtigoCoreEnv
from artigo_env.envs.observation import PAD, ObservationSizeError
from artigo_env.text.lemmatizers import get_lemmatizer

PRESET_NAMES = sorted(PRESETS.names())


def _play_until_end(env, seed=0, pick=lambda mask: int(np.flatnonzero(mask)[0])):
    env.reset(seed=seed)
    rewards, terminated, truncated, n = [], False, False, 0
    while not (terminated or truncated):
        _, reward, terminated, truncated, _ = env.step(pick(env.action_masks()))
        rewards.append(reward)
        n += 1
    return rewards, terminated, truncated, n


@pytest.mark.fast
@pytest.mark.req("REQ-ENV-001")
@pytest.mark.parametrize("preset", PRESET_NAMES)
def test_check_env_on_every_preset(preset):
    env = gym.make("ArtigoEnv-v0", ruleset=preset, render_mode="ansi")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        check_env(env.unwrapped)
    env.close()


@pytest.mark.fast
@pytest.mark.req("REQ-ENV-001")
def test_registration_defaults():
    spec = gym.spec("ArtigoEnv-v0")
    assert spec.max_episode_steps == 400
    assert spec.kwargs == {"corpus": "demo", "ruleset": "canonical", "scenario": "demo"}
    env = gym.make("ArtigoEnv-v0")
    assert env.unwrapped.max_steps == spec.max_episode_steps
    env.close()


@pytest.mark.fast
@pytest.mark.req("REQ-ENV-001")
def test_rgb_array_is_not_available():
    with pytest.raises(NotImplementedError, match="rgb_array"):
        ArtigoCoreEnv(render_mode="rgb_array")


@pytest.mark.fast
@pytest.mark.req("REQ-ENV-001")
def test_ansi_render_shows_the_screen():
    env = ArtigoCoreEnv(render_mode="ansi")
    env.reset(seed=1)
    env.step(0)
    text = env.render()
    assert isinstance(text, str) and "█" in text and "Guesses: 1" in text
    env.close()


@pytest.mark.fast
@pytest.mark.req("REQ-ENV-006")
@settings(max_examples=15, deadline=None)
@given(seed=st.integers(0, 10_000), preset=st.sampled_from(PRESET_NAMES))
def test_observations_stay_in_space(seed, preset):
    env = ArtigoCoreEnv(ruleset=preset, max_steps=60)
    rng = np.random.default_rng(seed)
    obs, _ = env.reset(seed=seed)
    assert env.observation_space.contains(obs)
    previous = obs
    done = False
    while not done:
        action = int(rng.choice(np.flatnonzero(env.action_masks())))
        obs, _, terminated, truncated, _ = env.step(action)
        assert env.observation_space.contains(obs)
        for key in obs:  # every observation is a new object
            assert obs[key] is not previous[key]
        previous = obs
        done = terminated or truncated
    env.close()


@pytest.mark.fast
@pytest.mark.req("REQ-ENV-013", "REQ-ENV-014")
def test_reward_and_victory():
    env = ArtigoCoreEnv()
    rewards, terminated, truncated, n = _play_until_end(env, seed=3)
    assert terminated and not truncated
    assert set(rewards) <= {-1.0, 0.0}
    assert env.episode_record().won
    # Repeats in canonical cost nothing; in strict-noops they count.
    env.close()
    for preset, expected in (("canonical", 0.0), ("strict-noops", -1.0)):
        env = ArtigoCoreEnv(ruleset=preset, mask_noops=False)
        env.reset(seed=0)
        env.step(5)
        _, reward, *_ = env.step(5)
        assert reward == expected
        env.close()


@pytest.mark.fast
@pytest.mark.req("REQ-ENV-014")
def test_truncation_only_by_the_limit():
    env = ArtigoCoreEnv(max_steps=5)
    env.reset(seed=0)
    numeric = np.flatnonzero(env.unwrapped._is_numeric)[-7:].tolist()  # demo titles: no digits
    flags = [env.step(a)[2:4] for a in numeric[:5]]
    assert flags[:-1] == [(False, False)] * 4
    assert flags[-1] == (False, True)
    env.close()
    wrapped = gym.make("ArtigoEnv-v0", max_steps=7, max_episode_steps=7)
    wrapped.reset(seed=0)
    for i, a in enumerate(numeric):
        _, _, terminated, truncated, _ = wrapped.step(a)
        assert not terminated and truncated == (i == 6)
    wrapped.close()


@pytest.mark.fast
@pytest.mark.req("REQ-ENV-015")
def test_spec_recreates_the_environment():
    first = gym.make("ArtigoEnv-v0", ruleset="no-lengths", strict_actions=False)
    second = gym.make(first.spec)
    for env in (first, second):
        assert env.unwrapped.ruleset.name == "no-lengths" and not env.unwrapped.strict_actions
    obs_a, _ = first.reset(seed=11)
    obs_b, _ = second.reset(seed=11)
    for action in (3, 50, 3, 2050, 7):
        a = first.step(action)
        b = second.step(action)
        for key in a[0]:
            np.testing.assert_array_equal(a[0][key], b[0][key])
        assert a[1:] == b[1:]
    first.close()
    second.close()


# -- REQ-ENV-008: never truncated ----------------------------------------------------------------


def _corpus_with_body(tmp_path, n_body: int, name: str):
    words = [f"termo{chr(97 + i % 26)}{chr(97 + (i // 26) % 26)}" for i in range(n_body)]
    row = dict.fromkeys(DOC_SCHEMA.names)
    row.update(
        source_id="synthetic",
        doc_id=f"{name}-0",
        title="Termoaa",
        body=" ".join(words),
        license_id="MIT",
        provenance="{}",
        domain="synthetic",
        split="demo",
    )
    corpus = tmp_path / name
    writer = CorpusWriter(corpus)
    writer.add(row)
    writer.add_lemma_view(get_lemmatizer("open-tables"))
    writer.finish({"build_id": f"{name}-build"})
    with CorpusStore.open(corpus) as store:
        vocab = build_vocab(
            store, get_ruleset("canonical"), VocabConfig(numeric_max=0), tmp_path / "vocab"
        )
    return str(corpus), str(vocab)


@pytest.mark.fast
@pytest.mark.req("REQ-ENV-008")
def test_body_of_exactly_l_tokens_is_whole_and_one_more_fails(tmp_path):
    L = 64
    corpus, vocab = _corpus_with_body(tmp_path, L, "exact")
    env = ArtigoCoreEnv(corpus=corpus, vocab=vocab, max_body_tokens=L)
    obs, _ = env.reset(seed=0)
    assert obs["n_tokens"][0] == L
    assert obs["body_lemma"].shape == (L,) and not (obs["body_lemma"] == PAD).any()
    assert (obs["body_length"] > 0).all()
    env.close()
    corpus, vocab = _corpus_with_body(tmp_path, L + 1, "longer")
    with pytest.raises(ObservationSizeError, match="never truncated"):
        ArtigoCoreEnv(corpus=corpus, vocab=vocab, max_body_tokens=L)
