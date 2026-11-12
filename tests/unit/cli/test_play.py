from __future__ import annotations

import io

import pytest

from artigo_env.cli.main import main
from artigo_env.cli.play import play_game
from artigo_env.cli.render import render_ansi, render_token
from artigo_env.engine.screen import ScreenToken

pytestmark = pytest.mark.fast

TITLE = "Gato preto"
BODY = "O gato preto dorme na casa azul.\n\n== Vida do gato ==\nEm 1990 o gato fugiu de casa."
HIDDEN_AT_START = ("gato", "preto", "dorme", "casa", "azul", "vida", "1990", "fugiu")


def run(ruleset, script: str) -> tuple[bool, str]:
    out = io.StringIO()
    won = play_game(TITLE, BODY, ruleset, io.StringIO(script), out)
    return won, out.getvalue()


@pytest.mark.req("DEC-071", "REQ-ENG-014")
def test_first_screen_hides_every_content_word(protocol_ruleset):
    won, text = run(protocol_ruleset, "")
    first = text.split(">")[0].lower()
    assert not won
    for word in HIDDEN_AT_START:
        assert word not in first
    assert "████ █████" in first  # the title, one block per hidden character
    assert "o ████ █████ █████ na ████ ████" in first
    assert "== ████ do ████ ==" in first
    assert "guesses: 0" in first
    assert text.rstrip().endswith("Game left unsolved.")


@pytest.mark.req("DEC-071")
def test_scripted_game_reaches_victory(protocol_ruleset):
    script = "casa\nde\ncasa\n!?\nzebra\n19\ngatos\n:help\npreto\nazul\n"
    won, text = run(protocol_ruleset, script)
    assert won
    assert '"casa": 2 occurrences' in text
    assert '"de" is a stopword: not counted' in text
    assert '"casa" was already guessed' in text
    assert "invalid guess" in text
    assert '"zebra": no occurrences' in text
    assert '"19": 1 occurrence' in text
    assert "19██" in text
    assert '"gatos": 4 occurrences' in text
    assert "Solved in 5 guesses." in text  # casa, zebra, 19, gatos, preto
    assert "(solved)" in text
    assert "azul" not in text  # input after the victory is never read, and stays hidden
    assert "History: casa: 2; zebra: 0; 19: 1; gatos: 4; preto: 2" in text


@pytest.mark.req("DEC-071")
def test_quit_command_stops_the_game(protocol_ruleset):
    won, text = run(protocol_ruleset, "casa\n:quit\ngato\npreto\n")
    assert not won
    assert "Game left unsolved." in text
    assert "Gato" not in text


@pytest.mark.req("DEC-071", "REQ-ENG-014")
def test_hidden_lengths_render_as_a_single_block(protocol_ruleset):
    from dataclasses import replace

    won, text = run(replace(protocol_ruleset, lengths_visible="none"), "19\n")
    first = text.split(">")[0]
    assert "█ █\n" in first
    assert "██" not in first
    assert "19█" in text and "19██" not in text


@pytest.mark.req("DEC-071")
def test_render_token():
    assert render_token(ScreenToken("Casa", None, 0, None)) == "Casa"
    assert render_token(ScreenToken(None, 4, 0, None)) == "████"
    assert render_token(ScreenToken(None, None, 0, None)) == "█"
    assert render_token(ScreenToken(None, None, 0, "1█")) == "1█"


@pytest.mark.req("DEC-071")
def test_render_ansi_layout(protocol_ruleset, make_game):
    engine, article, state = make_game(TITLE, BODY, protocol_ruleset)
    text = render_ansi(engine.screen(state, article), "hello")
    lines = text.split("\n")
    assert lines[0] == "████ █████"
    assert lines[1] == ""
    assert lines[2] == "O ████ █████ █████ na ████ ████"
    assert lines[3] == "== ████ do ████ =="
    assert lines[4] == "Em ████ o ████ █████ de ████"
    assert lines[-2] == "Guesses: 0"
    assert lines[-1] == "hello"


@pytest.mark.req("DEC-071")
def test_command_line_entry_point(tmp_path):
    body = tmp_path / "body.txt"
    body.write_text(BODY, encoding="utf-8")
    out = io.StringIO()
    code = main(
        ["play", "--title", TITLE, "--text-file", str(body), "--ruleset", "no-grouping"],
        stdin=io.StringIO("gato\npreto\n"),
        stdout=out,
    )
    assert code == 0
    assert "Solved in 2 guesses." in out.getvalue()


@pytest.mark.req("DEC-071")
def test_command_line_reports_errors(tmp_path):
    out = io.StringIO()
    missing = tmp_path / "missing.txt"
    code = main(["play", "--title", TITLE, "--text-file", str(missing)], io.StringIO(), out)
    assert code == 2 and "cannot read" in out.getvalue()

    body = tmp_path / "body.txt"
    body.write_text(BODY, encoding="utf-8")
    out = io.StringIO()
    code = main(
        ["play", "--title", "de", "--text-file", str(body), "--ruleset", "no-grouping"],
        io.StringIO(),
        out,
    )
    assert code == 2 and "error:" in out.getvalue()
