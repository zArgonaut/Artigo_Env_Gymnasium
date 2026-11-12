"""Shared pytest configuration: every test must declare a suite and its requirements."""

from __future__ import annotations

from pathlib import Path

import pytest

SUITES = ("fast", "slow", "local", "perf")
REPO_ROOT = Path(__file__).resolve().parents[1]


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    problems: list[str] = []
    for item in items:
        suites = [name for name in SUITES if item.get_closest_marker(name) is not None]
        if len(suites) != 1:
            problems.append(f"{item.nodeid}: expected exactly one of {SUITES}, got {suites}")
        if item.get_closest_marker("req") is None:
            problems.append(f"{item.nodeid}: missing @pytest.mark.req(...)")
    if problems:
        raise pytest.UsageError("Unmarked tests (SDD 15 §1):\n" + "\n".join(problems))


@pytest.fixture
def repo_root() -> Path:
    """Root of the real repository checkout."""
    return REPO_ROOT


# --- shared engine fixtures ------------------------------------------------------------------

PROTOCOL_LEMMATIZER = "test-protocol"


def _register_protocol_lemmatizer() -> None:
    """Register a small rule table that covers only the words of the protocol text."""
    from artigo_env.text.lemmatizers import LEMMATIZERS
    from artigo_env.text.lemmatizers.base import RuleLemmatizer
    from artigo_env.text.lemmatizers.tables import load_tables

    if PROTOCOL_LEMMATIZER not in LEMMATIZERS:
        path = REPO_ROOT / "tests" / "fixtures" / "lemma_tables" / "protocol.json"
        LEMMATIZERS.register(
            PROTOCOL_LEMMATIZER, lambda: RuleLemmatizer(PROTOCOL_LEMMATIZER, load_tables(path))
        )


_register_protocol_lemmatizer()


@pytest.fixture
def protocol_ruleset():
    """Canonical levers with the synthetic protocol lemmatizer."""
    from artigo_env.engine.rulesets import Ruleset

    return Ruleset(name="test-protocol", lemmatizer=PROTOCOL_LEMMATIZER)


@pytest.fixture
def protocol_text() -> tuple[str, str]:
    """Title and body of the synthetic article written for the 15-guess protocol."""
    texts = REPO_ROOT / "tests" / "fixtures" / "texts"
    title = (texts / "protocol_title.txt").read_text(encoding="utf-8").strip()
    body = (texts / "protocol_body.txt").read_text(encoding="utf-8")
    return title, body


@pytest.fixture
def make_game():
    """Factory: ``make_game(title, body, ruleset)`` -> ``(engine, article, state)``."""
    from artigo_env.engine import ArticleBuilder, Engine

    def build(title: str, body: str, ruleset):
        article, lexicon = ArticleBuilder(ruleset).from_text(title, body)
        engine = Engine(ruleset, lexicon)
        return engine, article, engine.new_game(article)

    return build
