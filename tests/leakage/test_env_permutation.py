"""Permutation test for ArtigoCoreEnv (SDD 15 §2, REQ-ENV-007).

Two corpora hold one article each with the same layout and the same revealed words; they differ
only in words that stay hidden. With the same vocabulary and the same actions, the observation,
the action mask and ``info`` must be bit-identical. With hidden lengths, hidden words may also
differ in length.
"""

from __future__ import annotations

import numpy as np
import pytest

from artigo_env.corpus.store import CorpusStore, CorpusWriter
from artigo_env.corpus.types import DOC_SCHEMA
from artigo_env.corpus.vocab import VocabConfig, build_vocab
from artigo_env.engine.rulesets import PRESETS, get_ruleset
from artigo_env.envs import ArtigoCoreEnv
from artigo_env.text.guess import clean_guess
from artigo_env.text.lemmatizers import get_lemmatizer

ARTICLE_A = ("Rato preto", "O rato preto dorme na casa azul.\n\n== Vida ==\nEm 1990 o rato fugiu.")
ARTICLE_B = ("Pato preto", "O pato preto corre na casa anil.\n\n== Vida ==\nEm 1991 o pato fugiu.")
ARTICLE_C = (
    "Jacaré preto",
    "O jacaré preto vai na casa lilás.\n\n== Vida ==\nEm 19911 o jacaré fugiu.",
)
GUESSED_FORMS = ["preto", "casa", "vida", "fugiu"]
NUMERIC_KEYS = ["n:9"]


def _corpus(tmp_path, name, article):
    row = dict.fromkeys(DOC_SCHEMA.names)
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
    path = tmp_path / name
    writer = CorpusWriter(path)
    writer.add(row)
    for lemmatizer in ("identity", "open-tables"):
        writer.add_lemma_view(get_lemmatizer(lemmatizer))
    writer.finish({"build_id": f"perm-{name}"})
    return path


def _run(corpus, vocab, preset, actions_of):
    env = ArtigoCoreEnv(corpus=str(corpus), vocab=str(vocab), ruleset=preset, mask_in_info=True)
    obs, info = env.reset(seed=0)
    frames = [(obs, info, env.action_masks())]
    for action in actions_of(env):
        obs, reward, terminated, truncated, info = env.step(action)
        assert not terminated
        frames.append((obs, (reward, info), env.action_masks()))
    env.close()
    return frames


def _assert_same(frames_a, frames_b):
    assert len(frames_a) == len(frames_b)
    for (obs_a, info_a, mask_a), (obs_b, info_b, mask_b) in zip(frames_a, frames_b, strict=True):
        assert obs_a.keys() == obs_b.keys()
        for key in obs_a:
            assert obs_a[key].dtype == obs_b[key].dtype
            assert obs_a[key].tobytes() == obs_b[key].tobytes(), key
        np.testing.assert_array_equal(mask_a, mask_b)
        assert repr(_comparable(info_a)) == repr(_comparable(info_b))


def _comparable(info):
    """``info`` without ``corpus_build``: a configuration id that differs by construction."""
    if isinstance(info, tuple):
        return tuple(_comparable(x) for x in info)
    if isinstance(info, dict):
        out = {k: v for k, v in info.items() if k != "corpus_build"}
        if "action_mask" in out:
            out["action_mask"] = out["action_mask"].tolist()
        return out
    return info


@pytest.mark.fast
@pytest.mark.req("REQ-ENV-007")
@pytest.mark.parametrize("preset", sorted(PRESETS.names()))
def test_hidden_words_do_not_reach_observation_mask_or_info(tmp_path, preset):
    ruleset = get_ruleset(preset)
    corpus_a = _corpus(tmp_path, "a", ARTICLE_A)
    with CorpusStore.open(corpus_a) as store:
        vocab = build_vocab(store, ruleset, VocabConfig(numeric_max=10), tmp_path / "vocab")
    lemmatizer = get_lemmatizer(ruleset.lemmatizer)

    def actions_of(env):
        vocabulary = env._require_vocab()
        keys = [lemmatizer.lemma(clean_guess(f)) for f in GUESSED_FORMS] + NUMERIC_KEYS
        ids = [vocabulary.action_id(k) for k in keys]
        assert min(ids) >= 0
        return ids

    frames_a = _run(corpus_a, vocab, preset, actions_of)
    others = [ARTICLE_B]
    if ruleset.lengths_visible == "none":
        others.append(ARTICLE_C)
    for i, article in enumerate(others):
        corpus = _corpus(tmp_path, f"other{i}", article)
        _assert_same(frames_a, _run(corpus, vocab, preset, actions_of))


@pytest.mark.fast
@pytest.mark.req("REQ-ENV-007")
def test_the_permutation_test_detects_visible_differences(tmp_path):
    """Sanity check: different hidden lengths are caught when the ruleset shows lengths."""
    ruleset = get_ruleset("canonical")
    corpus_a = _corpus(tmp_path, "a", ARTICLE_A)
    corpus_c = _corpus(tmp_path, "c", ARTICLE_C)
    with CorpusStore.open(corpus_a) as store:
        vocab = build_vocab(store, ruleset, VocabConfig(numeric_max=10), tmp_path / "vocab")
    frames_a = _run(corpus_a, vocab, "canonical", lambda env: [])
    frames_c = _run(corpus_c, vocab, "canonical", lambda env: [])
    with pytest.raises(AssertionError):
        _assert_same(frames_a, frames_c)
