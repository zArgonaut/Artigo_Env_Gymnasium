"""DOM selectors and class names of the game page (artigo.app).

Verified against the first real recording (2026-11-16, random mode, 9 guesses); the fake page
used by the tests follows exactly this structure. Both the recorder and the calibration tools
must take selectors from here, never hard-code them.

Structure of the real page (React + Material UI, so generated ``css-*`` / ``Mui*`` classes are
unstable and never used, except ``Mui-selected`` which marks the newest guess row):

- ``#root`` holds the whole app; ``#app-text-section`` holds ``h1#article-title`` and
  ``div#article-text`` (``p``, ``h2``, ``h3``, ``h4`` blocks); ``#app-controls-section`` holds
  the guess form and the guess list.
- Every token is ``span#token-title-N`` / ``span#token-text-N`` with ``data-token-length`` and
  class ``token``; ``N`` is the character offset of the token in the title / body text.
  Hidden tokens add ``obfuscated`` (shown as block characters); a token partly revealed by a
  numeric guess has no ``obfuscated`` class and shows a mix such as ``1██0``; tokens matching
  the latest counted guess add ``guess-match`` (and ``highlighted`` on one of them).
- The guess form is a plain ``<form>`` (no id) with ``input#guess-input`` and a
  ``button[title="Enviar"]`` of ``type="button"``.
- There is no counter element and no message element: the guess list is a virtualized list of
  ``li`` rows. The first row reads "N palpite(s)"; every other row holds
  ``div.list-item-left`` (the guess) and ``div.list-item-right`` (number of matching tokens, or
  a message for warnings), both carrying one status class: ``guess-found``,
  ``guess-not-found`` (a miss, shown with 0) or ``guess-warning`` (a stopword, shown with the
  text "não conta"; transient, not counted). The newest guess row comes first and its row has
  ``Mui-selected``.
"""

from __future__ import annotations

# --- containers ---------------------------------------------------------------------------
GAME_CONTAINER = "#root"
ARTICLE = "#app-text-section"
TITLE = "#article-title"
ARTICLE_TEXT = "#article-text"
CONTROLS = "#app-controls-section"

# --- guess form ---------------------------------------------------------------------------
GUESS_FORM = "#app-controls-section form"
GUESS_INPUT = "#guess-input"
SEND_BUTTON = "#app-controls-section form button[title='Enviar']"

# --- tokens -------------------------------------------------------------------------------
TOKEN_TAG = "span"
TOKEN_CLASS = "token"
TOKEN = f"{TOKEN_TAG}.{TOKEN_CLASS}"
TOKEN_ID_TITLE_PREFIX = "token-title-"
TOKEN_ID_TEXT_PREFIX = "token-text-"
TOKEN_HIDDEN_CLASS = "obfuscated"
TOKEN_MATCH_CLASS = "guess-match"
TOKEN_HIGHLIGHT_CLASS = "highlighted"
HIDDEN_CHAR = "█"  # full block, shown in place of every hidden character
TOKEN_LENGTH_ATTR = "data-token-length"
HEADING_TAGS = frozenset({"h2", "h3", "h4", "h5", "h6"})

# --- guess list ---------------------------------------------------------------------------
GUESS_ROW_TAG = "li"
GUESS_LEFT_CLASS = "list-item-left"
GUESS_RIGHT_CLASS = "list-item-right"
GUESS_FOUND_CLASS = "guess-found"
GUESS_NOT_FOUND_CLASS = "guess-not-found"
GUESS_WARNING_CLASS = "guess-warning"
GUESS_SELECTED_CLASS = "Mui-selected"
GUESS_HEADER_PATTERN = r"(\d+)\s+palpites?"

# The recorder waits for this element (the first token) before the "load" snapshot.
READY = f"{ARTICLE_TEXT} {TOKEN}"

# Response paths that carry the article served by the game: ``GET /articles/<mode>`` returns
# JSON ``{"title", "url", "text"}``.
ARTICLE_PATH_MARKER = "/articles/"
