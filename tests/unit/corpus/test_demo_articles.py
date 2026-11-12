"""The bundled demonstration corpus is well formed and playable (REQ-REPO-005)."""

from __future__ import annotations

import json
from typing import Any

import pytest

from artigo_env.core.resources import resource_path
from artigo_env.text.stopwords import StopwordSpec, load_stopwords
from artigo_env.text.tokenizer import tokenize

pytestmark = pytest.mark.fast

EXPECTED_KEYS = {"doc_id", "title", "body", "license_id"}
FORBIDDEN = ("http", "{{", "[[")


def _lines() -> list[str]:
    text = resource_path("demo", "articles.jsonl").read_text(encoding="utf-8")
    return [line for line in text.splitlines() if line.strip()]


def _articles() -> list[dict[str, Any]]:
    return [json.loads(line) for line in _lines()]


def _content_words(text: str) -> list[str]:
    stopwords = load_stopwords(StopwordSpec())
    words = [token.text.lower() for token in tokenize(text)]
    return [word for word in words if word not in stopwords]


@pytest.mark.req("REQ-REPO-005")
def test_corpus_has_thirty_articles() -> None:
    assert len(_lines()) == 30


@pytest.mark.req("REQ-REPO-005")
def test_keys_are_exact_and_license_is_mit() -> None:
    for article in _articles():
        assert set(article) == EXPECTED_KEYS, article.get("doc_id")
        assert article["license_id"] == "MIT"
        assert all(isinstance(article[key], str) for key in EXPECTED_KEYS)


@pytest.mark.req("REQ-REPO-005")
def test_ids_and_titles_are_unique() -> None:
    articles = _articles()
    ids = [article["doc_id"] for article in articles]
    titles = [article["title"] for article in articles]
    assert ids == [f"demo-{index:03d}" for index in range(1, 31)]
    assert len(set(titles)) == len(titles)


@pytest.mark.req("REQ-REPO-005")
def test_titles_have_one_to_six_content_words() -> None:
    for article in _articles():
        count = len(_content_words(article["title"]))
        assert 1 <= count <= 6, (article["doc_id"], article["title"], count)


@pytest.mark.req("REQ-REPO-005")
def test_body_length_in_tokens() -> None:
    for article in _articles():
        count = len(tokenize(article["body"]))
        assert 150 <= count <= 2000, (article["doc_id"], count)


@pytest.mark.req("REQ-REPO-005")
def test_every_title_content_word_appears_in_body() -> None:
    for article in _articles():
        body_words = {token.text.lower() for token in tokenize(article["body"])}
        missing = [word for word in _content_words(article["title"]) if word not in body_words]
        assert not missing, (article["doc_id"], missing)


@pytest.mark.req("REQ-REPO-005")
def test_no_markup_or_links() -> None:
    for line in _lines():
        for marker in FORBIDDEN:
            assert marker not in line, marker


@pytest.mark.req("REQ-REPO-005")
def test_some_article_has_a_section_header() -> None:
    assert any(any(token.header for token in tokenize(a["body"])) for a in _articles())
