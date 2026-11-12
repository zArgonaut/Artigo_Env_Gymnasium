"""ArtigoEnv ``engine`` layer (see docs/sdd/02, section B.2)."""

from artigo_env.engine.article import ArticleBuilder, assemble_article
from artigo_env.engine.engine import Engine
from artigo_env.engine.lexicon import InMemoryLexicon, Lexicon
from artigo_env.engine.screen import ScreenArrays, ScreenToken, ScreenView
from artigo_env.engine.types import (
    Article,
    GameState,
    Guess,
    GuessRecord,
    InvalidGuess,
    LexicalGuess,
    NumericGuess,
    Outcome,
    OutcomeKind,
    StopwordGuess,
    TitleProgress,
    TokenSeq,
)

__all__ = [
    "Article",
    "ArticleBuilder",
    "Engine",
    "GameState",
    "Guess",
    "GuessRecord",
    "InMemoryLexicon",
    "InvalidGuess",
    "LexicalGuess",
    "Lexicon",
    "NumericGuess",
    "Outcome",
    "OutcomeKind",
    "ScreenArrays",
    "ScreenToken",
    "ScreenView",
    "StopwordGuess",
    "TitleProgress",
    "TokenSeq",
    "assemble_article",
]
