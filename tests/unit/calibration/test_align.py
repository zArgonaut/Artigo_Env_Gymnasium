from __future__ import annotations

import pytest
from synthetic import TEXT, TITLE, render_game

from artigo_env.text.tokenizer import tokenize
from tools.calibration.align import align, article_tokens, revealed_indices
from tools.calibration.dom import DomToken, dom_tokens, read_counter, read_message

pytestmark = [pytest.mark.fast, pytest.mark.req("REQ-CAL-002")]


def hidden(index: int, length: int, *, header: bool = False) -> DomToken:
    return DomToken(index, length, None, False, False, header)


def dom_for(text: str) -> list[DomToken]:
    return [hidden(i, len(t.text), header=t.header) for i, t in enumerate(tokenize(text))]


def test_coherent_text_and_dom_have_no_divergence() -> None:
    tokens = article_tokens(TITLE, TEXT)
    dom = dom_tokens(render_game(TITLE, TEXT, set(), 0, None))
    result = align(dom, tokens)
    assert result.divergences == ()
    assert result.pairs == tuple((i, i) for i in range(len(tokens)))
    assert len(dom) == len(tokens)


def test_hyphenated_compound_as_one_element_resynchronises() -> None:
    text = "um guarda-chuva azul cobre todos os dias do mes agora"
    tokens = tokenize(text)
    lengths = [len(t.text) for t in tokens]
    # The game treats "guarda-chuva" as ONE element of length 12, we produce two tokens.
    game_lengths = [lengths[0], 12, *lengths[3:]]
    dom = [hidden(i, n) for i, n in enumerate(game_lengths)]
    result = align(dom, tokens)
    assert [d.kind for d in result.divergences] == ["count"]
    assert result.divergences[0].dom_index == 1
    assert result.divergences[0].token_index == 1
    # Everything after the compound is paired again.
    assert result.pairs[-1] == (len(dom) - 1, len(tokens) - 1)
    assert (0, 0) in result.pairs
    assert len(result.pairs) == len(dom) - 1


def test_length_mismatch_is_recorded() -> None:
    tokens = tokenize("alpha beta gamma delta epsilon")
    dom = [hidden(i, len(t.text)) for i, t in enumerate(tokens)]
    dom[2] = hidden(2, 9)
    result = align(dom, tokens)
    assert [d.kind for d in result.divergences] == ["length"]
    assert result.divergences[0].dom_index == 2
    assert (4, 4) in result.pairs


def test_revealed_form_with_other_spelling_is_recorded() -> None:
    tokens = tokenize("alpha beta gamma")
    dom = dom_for("alpha beta gamma")
    dom[1] = DomToken(1, 4, "BETA", True, False, False)
    result = align(dom, tokens)
    assert [d.kind for d in result.divergences] == ["form"]
    assert result.pairs == ((0, 0), (1, 1), (2, 2))


def test_header_flag_mismatch_is_recorded() -> None:
    tokens = tokenize("alpha\n== beta ==\ngamma")
    dom = dom_for("alpha\n== beta ==\ngamma")
    dom[1] = hidden(1, 4, header=False)
    result = align(dom, tokens)
    assert [d.kind for d in result.divergences] == ["header"]


def test_dom_tokens_parse_state_title_and_header() -> None:
    page = render_game(TITLE, TEXT, {0, 15}, 1, None)
    dom = dom_tokens(page)
    assert dom[0].revealed and dom[0].text == "Rio" and dom[0].in_title and dom[0].length == 3
    assert not dom[1].revealed and dom[1].text is None and dom[1].in_title
    assert not dom[2].in_title and not dom[2].header and not dom[2].revealed
    assert dom[15].revealed and dom[15].text == "História" and dom[15].header
    assert dom[15].length == len("História")
    assert [d.index for d in dom] == list(range(len(dom)))


def test_revealed_indices() -> None:
    before = dom_tokens(render_game(TITLE, TEXT, {0}, 1, None))
    after = dom_tokens(render_game(TITLE, TEXT, {0, 3, 22}, 2, None))
    assert revealed_indices(before, after) == [3, 22]
    assert revealed_indices(after, before) == []
    assert revealed_indices([], []) == []


def test_counter_and_message_readers() -> None:
    page = render_game(TITLE, TEXT, set(), 7, "não conta")
    assert read_counter(page) == 7
    assert read_message(page) == "não conta"
    assert read_message(render_game(TITLE, TEXT, set(), 0, None)) is None
    assert read_counter("<div></div>") is None
    assert read_message("<div></div>") is None


def test_alignment_never_raises() -> None:
    tokens = tokenize("alpha beta gamma delta")
    assert align([], []).divergences == ()
    assert align([], tokens).divergences[0].kind == "count"
    assert align(dom_for("alpha beta"), []).divergences[0].kind == "count"
    wild = [hidden(i, 40 + i) for i in range(30)]
    result = align(wild, tokens)
    assert result.divergences
    result = align(dom_for("a b"), tokenize("aaaaaaaaaaaa " * 50))
    assert result.divergences
