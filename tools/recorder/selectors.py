"""DOM selectors of the game page.

All values are provisional until the rehearsal session against the real site (LOCAL-REC-1);
the fake page used by the tests follows exactly this structure. Both the recorder and the
calibration tools must take selectors from here, never hard-code them.
"""

from __future__ import annotations

GAME_CONTAINER = "#game"  # provisional: LOCAL-REC-1
ARTICLE = "#article"  # provisional: LOCAL-REC-1
TITLE = "#title"  # provisional: LOCAL-REC-1
GUESS_FORM = "#guess-form"  # provisional: LOCAL-REC-1
GUESS_INPUT = "#guess-input"  # provisional: LOCAL-REC-1
COUNTER = "#guess-count"  # provisional: LOCAL-REC-1
MESSAGE = "#message"  # provisional: LOCAL-REC-1

TOKEN_TAG = "span"  # provisional: LOCAL-REC-1
TOKEN_CLASS = "token"  # provisional: LOCAL-REC-1
TOKEN_HIDDEN_CLASS = "hidden"  # provisional: LOCAL-REC-1
TOKEN_LENGTH_ATTR = "data-token-length"  # read in the game's client code
TOKEN = f"{TOKEN_TAG}.{TOKEN_CLASS}"

# Response paths that carry the article served by the game (provisional format, see
# tools/calibration/session.py).
ARTICLE_PATH_MARKER = "/articles/"
