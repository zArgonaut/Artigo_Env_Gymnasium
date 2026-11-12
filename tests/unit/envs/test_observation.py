"""Observation encoding of ArtigoCoreEnv (SDD 09 §3, DEC-065)."""

from __future__ import annotations

import time
from dataclasses import replace

import numpy as np
import pytest

from artigo_env.core.errors import ArtigoEnvError
from artigo_env.corpus.store import CorpusWriter
from artigo_env.corpus.types import DOC_SCHEMA
from artigo_env.corpus.vocab import VocabConfig, build_vocab
from artigo_env.engine import ArticleBuilder, Engine, NumericGuess
from artigo_env.engine.rulesets import get_ruleset
from artigo_env.envs.observation import (
    HIDDEN,
    OBS_HEADER,
    OBS_PARENTHESES,
    OBS_PARTIAL,
    OBS_STOPWORD,
    OBS_VISIBLE,
    PAD,
    ObservationSizeError,
    encode,
    observation_space,
    prepare,
    screen_info,
)
from artigo_env.text.lemmatizers import get_lemmatizer

TITLE = "Rato (animal)"
BODY = "O rato dorme em 1990.\n\n== Vida ==\nO rato fugiu."


def _game(ruleset_name="canonical", body_tokens=32):
    ruleset = replace(get_ruleset(ruleset_name), lemmatizer="identity")
    article, lexicon = ArticleBuilder(ruleset).from_text(TITLE, BODY)
    engine = Engine(ruleset, lexicon)
    n_forms = int(max(article.title.forms.max(), article.body.forms.max())) + 1
    # Observation ids: form id doubles as lemma id; the form of "fugiu" is outside F.
    lemma_obs = np.arange(n_forms, dtype=np.int32)
    form_obs = np.arange(n_forms, dtype=np.int32)
    fugiu = article.body.forms[article.body.surface.index("fugiu")]
    form_obs[fugiu] = -1
    lengths = ruleset.lengths_visible != "none"
    episode = prepare(
        article,
        lemma_obs,
        form_obs,
        body_tokens=body_tokens,
        show_title_lengths=lengths,
        show_body_lengths=lengths,
    )
    return engine, article, episode, n_forms


def _obs(engine, article, episode, state, steps=0):
    screen = engine.screen_arrays(state, article)
    history = np.full((4, 3), -1, dtype=np.int32)
    return encode(episode, screen, history, steps), screen_info(episode, screen)


@pytest.mark.fast
@pytest.mark.req("REQ-ENV-006")
def test_hidden_padding_and_flags():
    engine, article, episode, n_forms = _game()
    state = engine.new_game(article)
    obs, info = _obs(engine, article, episode, state)
    nb, nt = len(article.body), len(article.title)
    assert obs["n_tokens"].tolist() == [nb, nt]
    assert (obs["body_lemma"][nb:] == PAD).all() and (obs["body_length"][nb:] == 0).all()
    assert (obs["title_lemma"][nt:] == PAD).all()
    stop = article.body.stop
    assert (obs["body_lemma"][:nb][~stop] == HIDDEN).all()
    assert (obs["body_flags"][:nb][stop] & (OBS_VISIBLE | OBS_STOPWORD) != 0).all()
    assert (obs["body_flags"][:nb][~stop] & OBS_VISIBLE == 0).all()
    # Lengths are visible in canonical; the header bit survives on hidden tokens.
    assert obs["body_length"][:nb].tolist() == article.body.lengths.tolist()
    vida = article.body.surface.index("Vida")
    assert obs["body_flags"][vida] & OBS_HEADER
    # "animal" is inside parentheses in the title.
    assert obs["title_flags"][1] & OBS_PARENTHESES
    assert not obs["title_flags"][0] & OBS_PARENTHESES
    assert info == {"forms": {}, "partial": {}}
    space = observation_space(
        n_actions=n_forms, n_stop=0, n_forms=n_forms, body_tokens=32, history_rows=4
    )
    assert space.contains(obs)


@pytest.mark.fast
@pytest.mark.req("REQ-ENV-006", "REQ-ENV-007")
def test_partial_reveal_and_forms_outside_f():
    engine, article, episode, _ = _game()
    state = engine.new_game(article)
    state, _ = engine.apply(state, article, NumericGuess("9"))
    lemma_fugiu = int(article.body.lemmas[article.body.surface.index("fugiu")])
    state, _ = engine.apply(state, article, engine.guess_from_lemma(lemma_fugiu))
    obs, info = _obs(engine, article, episode, state, steps=2)
    nt = len(article.title)
    year = article.body.surface.index("1990")
    assert obs["body_lemma"][year] == HIDDEN
    assert obs["body_flags"][year] & OBS_PARTIAL
    assert info["partial"] == {nt + year: "█99█"}
    fugiu = article.body.surface.index("fugiu")
    assert obs["body_form"][fugiu] == -1 and obs["body_lemma"][fugiu] >= 0
    assert info["forms"] == {nt + fugiu: "fugiu"}
    assert obs["step"].tolist() == [2]


@pytest.mark.fast
@pytest.mark.req("REQ-ENV-007")
def test_hidden_lengths_are_zero_without_length_rule():
    engine, article, episode, _ = _game("no-lengths")
    state = engine.new_game(article)
    state, _ = engine.apply(state, article, NumericGuess("9"))
    obs, info = _obs(engine, article, episode, state)
    nb = len(article.body)
    hidden = ~article.body.stop
    assert (obs["body_length"][:nb][hidden] == 0).all()
    nt = len(article.title)
    assert info["partial"] == {nt + article.body.surface.index("1990"): "█99█"}


@pytest.mark.fast
@pytest.mark.req("REQ-ENV-008")
def test_prepare_refuses_documents_that_do_not_fit():
    engine, article, _, n_forms = _game()
    ids = np.arange(n_forms, dtype=np.int32)
    with pytest.raises(ObservationSizeError):
        prepare(article, ids, ids, body_tokens=len(article.body) - 1)
    with pytest.raises(ObservationSizeError):
        prepare(article, ids, ids, body_tokens=64, title_tokens=len(article.title) - 1)
    assert issubclass(ObservationSizeError, ArtigoEnvError)


@pytest.mark.fast
@pytest.mark.req("REQ-ENV-006")
def test_every_call_returns_new_arrays():
    engine, article, episode, _ = _game()
    state = engine.new_game(article)
    first, _ = _obs(engine, article, episode, state)
    second, _ = _obs(engine, article, episode, state)
    for key in first:
        assert first[key] is not second[key]
        assert not np.shares_memory(first[key], second[key])
        np.testing.assert_array_equal(first[key], second[key])


# -- performance (informative) -----------------------------------------------------------------


def _long_corpus(tmp_path, n_tokens: int):
    words = [f"palavra{chr(97 + i % 26)}{chr(97 + (i // 26) % 26)}" for i in range(n_tokens - 2)]
    row = dict.fromkeys(DOC_SCHEMA.names)
    row.update(
        source_id="perf",
        doc_id="perf-0",
        title="Palavraaa palavrabb",
        body=" ".join(words) + " fim fim",
        license_id="MIT",
        provenance="{}",
        domain="perf",
        split="demo",
    )
    corpus = tmp_path / "corpus"
    writer = CorpusWriter(corpus)
    writer.add(row)
    writer.add_lemma_view(get_lemmatizer("open-tables"))
    writer.finish({"build_id": "perf-build"})
    from artigo_env.corpus.store import CorpusStore

    with CorpusStore.open(corpus) as store:
        vocab = build_vocab(store, get_ruleset("canonical"), VocabConfig(numeric_max=10), tmp_path)
    return corpus, vocab


@pytest.mark.perf
@pytest.mark.req("REQ-ENV-001")
@pytest.mark.parametrize("body_tokens", [4096, 16384])
def test_step_and_reset_time(tmp_path, body_tokens):
    from artigo_env.envs import ArtigoCoreEnv

    corpus, vocab = _long_corpus(tmp_path, 3000)
    env = ArtigoCoreEnv(corpus=str(corpus), vocab=str(vocab), max_body_tokens=body_tokens)
    keys = env._require_vocab().keys()
    title = {"palavraaa", "palavrabb"}
    actions = [i for i, k in enumerate(keys) if not k.startswith("n:") and k not in title][:40]
    resets, steps = [], []
    for seed in range(20):
        start = time.perf_counter()
        env.reset(seed=seed)
        resets.append(time.perf_counter() - start)
        for a in actions:
            start = time.perf_counter()
            env.step(a)
            steps.append(time.perf_counter() - start)
    env.close()
    reset_ms = float(np.median(resets) * 1e3)
    step_ms = float(np.median(steps) * 1e3)
    print(f"\nL={body_tokens}: reset median {reset_ms:.3f} ms, step median {step_ms:.3f} ms")
    assert reset_ms > 0 and step_ms > 0
