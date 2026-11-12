"""Text normalization shared by the tokenizer, the corpus pipeline and the engine."""

from __future__ import annotations

import unicodedata


def normalize(text: str) -> str:
    """Return ``text`` in Unicode NFC (REQ-TXT-001). Diacritics are never removed (REQ-TXT-002).

    The game's tokenizer does not recognise combining marks, so decomposed input must be
    composed before tokenization.
    """
    return unicodedata.normalize("NFC", text)


def lower(form: str) -> str:
    """Lowercase like JavaScript's ``toLowerCase`` (REQ-TXT-006).

    ``str.casefold`` is forbidden in this package: it diverges from the game (for example it
    turns "ß" into "ss").
    """
    return form.lower()
