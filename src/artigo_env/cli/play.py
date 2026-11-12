"""``artigo-env play``: a human plays one article in the terminal (SDD 09 §7, DEC-071)."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TextIO

from artigo_env.cli.render import describe_outcome, render_ansi
from artigo_env.core.errors import ArtigoEnvError
from artigo_env.engine import ArticleBuilder, Engine, InvalidGuess, NumericGuess
from artigo_env.engine.rulesets import PRESETS, Ruleset, get_ruleset

QUIT = ":quit"
HELP = ":help"
_HELP_TEXT = f"Type a word or a number and press Enter. {QUIT} leaves the game; {HELP} shows this."


def play_game(
    title: str,
    body: str,
    ruleset: Ruleset,
    stdin: TextIO,
    stdout: TextIO,
) -> bool:
    """Run one interactive game; returns ``True`` when the title was solved."""
    article, lexicon = ArticleBuilder(ruleset).from_text(title, body)
    engine = Engine(ruleset, lexicon)
    state = engine.new_game(article)

    def show(message: str | None = None) -> None:
        stdout.write(render_ansi(engine.screen(state, article), message) + "\n")

    show(_HELP_TEXT)
    while state.status != "won":
        stdout.write("> ")
        stdout.flush()
        line = stdin.readline()
        if not line:  # end of input
            break
        raw = line.strip()
        if raw == QUIT:
            break
        if raw == HELP:
            stdout.write(_HELP_TEXT + "\n")
            continue
        guess = engine.parse_guess(raw)
        state, outcome = engine.apply(state, article, guess)
        if isinstance(guess, InvalidGuess):
            form = raw
        elif isinstance(guess, NumericGuess):
            form = guess.digits
        else:
            form = guess.form
        show(describe_outcome(outcome, form))
    won = state.status == "won"
    stdout.write(f"Solved in {state.n_counted} guesses.\n" if won else "Game left unsolved.\n")
    return won


def add_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("play", help="play one article in the terminal")
    parser.add_argument("--title", required=True, help="article title (the answer)")
    parser.add_argument(
        "--text-file", required=True, type=Path, help="UTF-8 text file with the article body"
    )
    parser.add_argument(
        "--ruleset", default="canonical", choices=PRESETS.names(), help="ruleset preset"
    )
    parser.set_defaults(handler=run)


def run(args: argparse.Namespace, stdin: TextIO, stdout: TextIO) -> int:
    try:
        body = args.text_file.read_text(encoding="utf-8")
    except OSError as error:
        stdout.write(f"error: cannot read {args.text_file}: {error.strerror}\n")
        return 2
    try:
        play_game(args.title, body, get_ruleset(args.ruleset), stdin, stdout)
    except ArtigoEnvError as error:
        stdout.write(f"error: {error}\n")
        return 2
    return 0
