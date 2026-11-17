"""Self-test checks of the installation, the text layer, the engine and the corpus store."""

from __future__ import annotations

import importlib
import importlib.metadata
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from artigo_env.selftest.registry import Context, check

DEPENDENCIES = {  # distribution -> import name
    "numpy": "numpy",
    "pyarrow": "pyarrow",
    "regex": "regex",
    "xxhash": "xxhash",
    "PyYAML": "yaml",
    "gymnasium": "gymnasium",
}
PINNED = {"gymnasium": "1.4.0"}
STOPWORD_COUNT = 227
LEMMA_EXAMPLES = {
    "anos": "ano",
    "nasceu": "nascer",
    "cidades": "cidade",
    "casas": "casa",
    "maiores": "maior",
}


def expect(condition: object, message: str) -> None:
    """Fail the current check with ``message`` (unlike ``assert``, survives ``python -O``)."""
    if not condition:
        raise AssertionError(message)


def expect_equal(actual: Any, expected: Any, what: str) -> None:
    if actual != expected:
        raise AssertionError(f"{what}: expected {expected!r}, got {actual!r}")


def demo_dir(*parts: str) -> Path:
    from artigo_env.core.resources import resource_path

    path = Path(str(resource_path("demo", *parts)))
    expect(path.exists(), f"packaged demo data missing: {path}")
    return path


# -- install --------------------------------------------------------------------------------------


@check("install", "python and dependencies")
def python_and_dependencies(ctx: Context) -> str:
    expect(sys.version_info >= (3, 11), f"Python >= 3.11 required, found {sys.version.split()[0]}")
    found = []
    for dist, module in DEPENDENCIES.items():
        version = importlib.metadata.version(dist)
        importlib.import_module(module)
        if dist in PINNED:
            expect_equal(version, PINNED[dist], f"{dist} version")
        found.append(f"{module} {version}")
    return f"Python {sys.version.split()[0]}; " + ", ".join(found)


@check("install", "stopword list")
def stopword_list(ctx: Context) -> str:
    from artigo_env.text.stopwords import StopwordSpec, load_stopwords

    words = load_stopwords(StopwordSpec())
    expect_equal(len(words), STOPWORD_COUNT, "number of stopwords")
    return f"{len(words)} words"


@check("install", "lemmatizer tables")
def lemmatizer_tables(ctx: Context) -> str:
    from artigo_env.engine.rulesets import get_ruleset
    from artigo_env.text.lemmatizers import get_lemmatizer

    name = get_ruleset("canonical").lemmatizer
    lemmatizer = get_lemmatizer(name)
    digest = lemmatizer.content_hash()
    return f"{name} loaded ({digest[:12]})"


@check("install", "sources.yaml")
def sources_yaml(ctx: Context) -> str:
    import yaml

    from artigo_env.core.resources import resource_path

    data = yaml.safe_load(resource_path("sources.yaml").read_text(encoding="utf-8"))
    expect(isinstance(data, dict) and data, "sources.yaml is not a non-empty mapping")
    return f"{len(data)} top-level keys"


@check("install", "demo corpus and vocabularies")
def demo_data(ctx: Context) -> str:
    from artigo_env.core.vocab import Vocabulary
    from artigo_env.corpus.store import CorpusStore

    with CorpusStore.open(demo_dir("corpus")) as store:
        n_docs, build_id = store.n_docs, store.build_id
        expect_equal(n_docs, int(store.manifest["n_docs"]), "documents in the demo corpus")
    lines = demo_dir("articles.jsonl").read_text(encoding="utf-8").splitlines()
    expect_equal(len([x for x in lines if x.strip()]), n_docs, "articles in articles.jsonl")
    vocabs = sorted(p for p in demo_dir("vocab").iterdir() if p.is_dir())
    expect(vocabs, "no packaged demo vocabulary")
    for path in vocabs:
        with Vocabulary.open(path) as vocab:
            expect(vocab.size > 0, f"empty vocabulary {path.name}")
            expect_equal(vocab.manifest.get("build_id"), build_id, f"build of {path.name}")
    return f"{n_docs} articles, {len(vocabs)} vocabularies"


# -- text -----------------------------------------------------------------------------------------


@check("text", "tokenizer")
def tokenizer(ctx: Context) -> str:
    from artigo_env.text.tokenizer import tokenize

    tokens = tokenize("O gato-preto (1990) dorme.\n\n== Vida ==\nFim.")
    expect_equal(
        [t.text for t in tokens],
        ["O", "gato", "preto", "1990", "dorme", "Vida", "Fim"],
        "tokens",
    )
    expect_equal([t.header for t in tokens][5:], [True, False], "header flags")
    return f"{len(tokens)} tokens, header detected"


@check("text", "guess clean-up and classification")
def guesses(ctx: Context) -> str:
    from artigo_env.text.guess import GuessKind, clean_guess, process_guess
    from artigo_env.text.stopwords import StopwordSpec, load_stopwords

    expect_equal(clean_guess("  Ação! "), "ação", "clean_guess")
    stop = load_stopwords(StopwordSpec())
    cases = {
        "Casa": GuessKind.LEXICAL,
        "de": GuessKind.STOPWORD,
        "1990": GuessKind.NUMERIC,
        "?!": GuessKind.INVALID,
    }
    for raw, kind in cases.items():
        expect_equal(process_guess(raw, stopwords=stop).kind, kind, f"kind of {raw!r}")
    return f"{len(cases) + 1} examples"


@check("text", "lemmatizer")
def lemmatizer(ctx: Context) -> str:
    from artigo_env.engine.rulesets import get_ruleset
    from artigo_env.text.lemmatizers import get_lemmatizer

    lem = get_lemmatizer(get_ruleset("canonical").lemmatizer)
    for form, lemma in LEMMA_EXAMPLES.items():
        expect_equal(lem.lemma(form), lemma, f"lemma of {form!r}")
    return ", ".join(f"{f}->{lemma}" for f, lemma in list(LEMMA_EXAMPLES.items())[:3]) + ", ..."


# -- engine ---------------------------------------------------------------------------------------

GAME_TITLE = "Gato preto"
GAME_BODY = "O gato preto dorme na casa.\n\nEm 1990 o gato fugiu de casa."


def _game() -> tuple[Any, Any, Any]:
    from artigo_env.engine import ArticleBuilder, Engine
    from artigo_env.engine.rulesets import get_ruleset

    ruleset = get_ruleset("canonical")
    article, lexicon = ArticleBuilder(ruleset).from_text(GAME_TITLE, GAME_BODY)
    engine = Engine(ruleset, lexicon)
    return engine, article, engine.new_game(article)


@check("engine", "scripted game to victory")
def scripted_game(ctx: Context) -> str:
    engine, article, state = _game()
    script = [  # guess, outcome, counted, count
        ("zebra", "miss", True, 0),
        ("de", "noop_stopword", False, 0),
        ("gato", "hit", True, 3),
        ("gatos", "noop_repeat", False, 0),
        ("casa", "hit", True, 2),
        ("preto", "hit", True, 2),
    ]
    outcome = None
    for raw, kind, counted, count in script:
        expect(state.status == "playing", f"game ended before {raw!r}")
        state, outcome = engine.apply(state, article, engine.parse_guess(raw))
        got = (outcome.kind.value, outcome.counted, outcome.count)
        expect_equal(got, (kind, counted, count), f"outcome of {raw!r}")
    expect(outcome is not None and outcome.won, "the game was not won")
    expect_equal(state.status, "won", "final status")
    expect_equal(state.n_counted, 4, "counted guesses")
    return f"won in {state.n_counted} counted guesses ({len(script)} inputs)"


@check("engine", "numeric partial reveal")
def numeric_partial(ctx: Context) -> str:
    engine, article, state = _game()
    state, outcome = engine.apply(state, article, engine.parse_guess("19"))
    year = article.surfaces.index("1990")
    expect_equal((outcome.kind.value, outcome.count), ("hit", 1), "outcome of '19'")
    expect_equal(dict(outcome.newly_partial), {year: 0b0011}, "partial bits")
    expect(not state.revealed[year], "1990 fully revealed by '19'")
    state, outcome = engine.apply(state, article, engine.parse_guess("90"))
    expect(bool(state.revealed[year]), "1990 not revealed after '19' and '90'")
    return "'19' then '90' reveal 1990"


@check("engine", "state serialization round trip")
def serialization(ctx: Context) -> str:
    engine, article, state = _game()
    for raw in ("gato", "19", "zebra"):
        state, _ = engine.apply(state, article, engine.parse_guess(raw))
    data = engine.serialize(state)
    restored = engine.deserialize(json.loads(json.dumps(data)))
    expect_equal(engine.serialize(restored), data, "serialized state after round trip")
    expect(np.array_equal(restored.revealed, state.revealed), "revealed mask differs")
    return f"{len(json.dumps(data))} bytes of JSON"


@check("engine", "purity")
def purity(ctx: Context) -> str:
    engine, article, state = _game()
    revealed = state.revealed.copy()
    before = engine.serialize(state)
    engine.apply(state, article, engine.parse_guess("gato"))
    expect(np.array_equal(state.revealed, revealed), "apply() changed the input mask")
    expect_equal(engine.serialize(state), before, "input state after apply()")
    return "input state untouched"


# -- corpus ---------------------------------------------------------------------------------------


@check("corpus", "demo store is memory-mapped")
def demo_store_mmap(ctx: Context) -> str:
    from artigo_env.corpus.store import CorpusStore

    with CorpusStore.open(demo_dir("corpus")) as store:
        arrays = store._arrays  # token arrays of the store (np.load with mmap_mode="r")
        expect(arrays, "the store has no token arrays")
        for name, array in arrays.items():
            expect(isinstance(array, np.memmap), f"{name} is not memory-mapped")
        return f"{len(arrays)} token arrays mapped, {store.n_docs} documents"


@check("corpus", "stored article equals article from raw text")
def store_matches_raw(ctx: Context) -> str:
    from artigo_env.corpus.store import CorpusStore, StoreLexicon
    from artigo_env.engine import ArticleBuilder
    from artigo_env.engine.rulesets import get_ruleset

    ruleset = get_ruleset("canonical")
    raw = [
        json.loads(line)
        for line in demo_dir("articles.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    builder = ArticleBuilder(ruleset)
    with CorpusStore.open(demo_dir("corpus")) as store:
        lexicon = StoreLexicon(store, store.lemma_view(ruleset.lemmatizer))
        n = 3 if ctx.quick else store.n_docs
        for i in range(n):
            doc = raw[i]
            expect_equal(store.doc_id(i), doc["doc_id"], f"doc_id of document {i}")
            stored = store.article(i, ruleset)
            built, built_lex = builder.from_text(doc["title"], doc["body"])
            expect_equal(stored.surfaces, built.surfaces, f"surfaces of {doc['doc_id']}")
            for name in ("lengths", "flags", "stop"):
                expect(
                    np.array_equal(getattr(stored, name), getattr(built, name)),
                    f"{name} of {doc['doc_id']}",
                )
            expect(
                np.array_equal(stored.digit_positions, built.digit_positions),
                f"digit positions of {doc['doc_id']}",
            )
            stored_lemmas = [lexicon.lemma_text(int(x)) for x in stored.lemmas]
            built_lemmas = [built_lex.lemma_text(int(x)) for x in built.lemmas]
            expect_equal(stored_lemmas, built_lemmas, f"lemmas of {doc['doc_id']}")
    return f"{n} articles identical"
