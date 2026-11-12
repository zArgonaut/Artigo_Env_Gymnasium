"""S3 normalize: mojibake, NFC, quotes, whitespace, boilerplate and portal suffixes (SDD 06 §5).

REQ-COR-004. ``ftfy`` belongs to the optional ``data`` extra and is imported on first use.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from types import ModuleType

import pyarrow as pa
import regex

from artigo_env.core.config import ConfigBase
from artigo_env.core.errors import ArtigoEnvError
from artigo_env.corpus.stages.context import check_batch, finish
from artigo_env.corpus.types import StageContext, StageResult
from artigo_env.text.normalize import lower, normalize
from artigo_env.text.tokenizer import tokenize

__all__ = [
    "DEFAULT_BOILERPLATE",
    "PUBLIC_SUFFIXES",
    "Cleaner",
    "NormalizeStage",
    "clean_body",
    "clean_title",
    "strip_portal_suffix",
]

DEFAULT_BOILERPLATE: tuple[str, ...] = (
    "leia também",
    "leia mais",
    "veja também",
    "saiba mais",
    "continua depois da publicidade",
    "publicidade",
    "compartilhe",
)

# Simple public suffixes, longest first (a host ending in none of them loses its last label).
PUBLIC_SUFFIXES: tuple[str, ...] = (".com.br", ".gov.br", ".com", ".pt", ".org", ".net")

MAX_SUFFIX_TOKENS = 4

_QUOTES = str.maketrans(
    {
        "“": '"',
        "”": '"',
        "„": '"',
        "‟": '"',
        "«": '"',
        "»": '"',
        "″": '"',
        "‘": "'",
        "’": "'",
        "‚": "'",
        "‛": "'",
        "‹": "'",
        "›": "'",
        "′": "'",
    }
)
_SPACES = regex.compile(r"[^\S\n]+")
_ANY_SPACE = regex.compile(r"\s+")
_ALNUM = regex.compile(r"[\p{L}\p{N}]")
_BREAKS = regex.compile(r"\n{3,}")
_SEPARATOR = regex.compile(r" (?:-|\||–|—) ")
_NOT_ASCII_ALNUM = regex.compile(r"[^a-z0-9]")


@dataclass(frozen=True)
class Cleaner(ConfigBase):
    """Per-source cleaning rules (SDD 06 §5); comparisons ignore case."""

    boilerplate_prefixes: tuple[str, ...] = DEFAULT_BOILERPLATE
    title_suffixes: tuple[str, ...] = ()


_FTFY: ModuleType | None = None


def _ftfy() -> ModuleType:
    global _FTFY
    if _FTFY is None:
        try:
            import ftfy
        except ImportError as exc:
            raise ArtigoEnvError(
                "the normalize stage needs 'ftfy'; install the data extra: "
                "pip install 'artigo-env[data]'"
            ) from exc
        _FTFY = ftfy
    return _FTFY


def _fix(text: str) -> str:
    """``ftfy`` repair, NFC, ASCII quotes and Unix line breaks."""
    fixed: str = _ftfy().fix_text(text)
    fixed = normalize(fixed).translate(_QUOTES)
    return fixed.replace("\r\n", "\n").replace("\r", "\n")


def _is_boilerplate(line: str, prefixes: tuple[str, ...]) -> bool:
    low = lower(line)
    for prefix in prefixes:
        if low.startswith(prefix) and (
            len(low) == len(prefix) or _ALNUM.match(low, len(prefix)) is None
        ):
            return True
    return False


def clean_body(text: str, cleaner: Cleaner) -> str:
    """Body after S3; never truncated beyond the removed boilerplate and punctuation lines."""
    prefixes = tuple(lower(p.strip()) for p in cleaner.boilerplate_prefixes if p.strip())
    lines: list[str] = []
    for raw in _fix(text).split("\n"):
        line = _SPACES.sub(" ", raw).strip()
        if line and _ALNUM.search(line) is None:
            continue  # punctuation-only line
        if line and _is_boilerplate(line, prefixes):
            continue
        lines.append(line)
    return _BREAKS.sub("\n\n", "\n".join(lines)).strip("\n")


def _fold(text: str) -> str:
    """Lowercase ASCII letters and digits only, accents folded (for domain comparison)."""
    decomposed = unicodedata.normalize("NFD", lower(text))
    return _NOT_ASCII_ALNUM.sub("", decomposed)


def _domain_keys(host: str) -> tuple[str, ...]:
    """Registrable part of ``host`` (before its public suffix): joined, then label by label."""
    for suffix in PUBLIC_SUFFIXES:
        if host.endswith(suffix) and len(host) > len(suffix):
            part = host[: -len(suffix)]
            break
    else:
        part = host.rsplit(".", 1)[0] if "." in host else host
    labels = [_fold(label) for label in part.split(".")]
    keys = ["".join(labels), *labels]
    return tuple(dict.fromkeys(key for key in keys if len(key) >= 2))


def strip_portal_suffix(
    title: str, domain: str | None, cleaner: Cleaner, hyphen: str = "split"
) -> str:
    """Remove the last `` - ``/`` | ``/`` – ``/`` — `` segment when it names the portal.

    The segment goes when it has at most four tokens and either matches a configured
    ``title_suffixes`` entry or its folded form contains, or is contained in, the folded
    registrable part of ``domain`` (joined or any label of two or more characters). The
    remaining title must keep at least one token.
    """
    matches = list(_SEPARATOR.finditer(title))
    if not matches:
        return title
    last = matches[-1]
    head, segment = title[: last.start()].strip(), title[last.end() :].strip()
    if not tokenize(head, hyphen) or len(tokenize(segment, hyphen)) > MAX_SUFFIX_TOKENS:
        return title
    suffixes = {lower(s.strip()) for s in cleaner.title_suffixes}
    if lower(segment) in suffixes:
        return head
    key = _fold(segment)
    if domain and len(key) >= 2:
        for cand in _domain_keys(domain):
            if cand in key or key in cand:
                return head
    return title


def clean_title(title: str, domain: str | None, cleaner: Cleaner, hyphen: str = "split") -> str:
    """Title after S3: repaired, on one line, without a portal suffix."""
    one_line = _ANY_SPACE.sub(" ", _fix(title)).strip()
    return strip_portal_suffix(one_line, domain, cleaner, hyphen).strip()


class NormalizeStage:
    """Cleans ``title`` and ``body``; removes ``empty_body`` and ``empty_title`` documents."""

    name = "normalize"
    version = "1"

    def __init__(self, cleaner: Cleaner | None = None) -> None:
        self.cleaner = cleaner if cleaner is not None else Cleaner()

    def run(self, batch: pa.RecordBatch, ctx: StageContext) -> StageResult:
        check_batch(batch, self.name)
        if batch.num_rows:
            _ftfy()  # fail early with the install hint
        titles = batch.column("title").to_pylist()
        bodies = batch.column("body").to_pylist()
        domains = batch.column("url_domain").to_pylist()
        new_titles: list[str | None] = []
        new_bodies: list[str] = []
        reasons: list[str | None] = []
        for title, body, domain in zip(titles, bodies, domains, strict=True):
            clean_b = clean_body(body or "", self.cleaner)
            clean_t = (
                clean_title(title, domain, self.cleaner, ctx.tokenizer_hyphen) if title else ""
            )
            new_bodies.append(clean_b)
            new_titles.append(clean_t if title is not None else None)
            if not clean_b.strip():
                reasons.append("empty_body")
            elif not clean_t.strip():
                reasons.append("empty_title")
            else:
                reasons.append(None)
        return finish(self.name, batch, reasons, {"title": new_titles, "body": new_bodies})
