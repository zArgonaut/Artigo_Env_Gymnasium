"""Evaluation set W: title list, eligibility, stratified splits, L_max and the W index.

SDD 06 §9 and §12 (DEC-050), plan F3 WP3.5. W v1 is the TextExtracts snapshot written by
``tools/w/fetch_extracts.py`` (plan decision 5). The title list is the TSV of the Wikipédia
Offline project (columns ``titulo``, ``em_completa``, ``em_lista``, ``sublistas`` after ``#``
comment lines); the main universe is ``em_completa == 1``.

* :func:`measure_snapshot` reads a snapshot with the same normalization and token counts as the
  W build (S3 with an empty ``Cleaner``, S4);
* :func:`eligible` applies SDD 06 §12 items 1 and 5 (main list, no ``Anexo:``/``WP:``, a content
  word in the title, body not above the cap), merging titles that resolve to the same page;
* :func:`make_splits` draws W-test / W-val / W-pool stratified by title content words
  {1, 2, 3, >=4} x body-length tercile, with seed 20261109 and largest-remainder allocation;
* :class:`WEligibilityStage` gives each page of a W build its split (``w-extra`` when it is not
  an eligible page of the main list);
* :func:`build_windex` builds the decontamination index from a compiled W corpus plus every
  requested and resolved title of the snapshot.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pyarrow as pa

from artigo_env.core.errors import ArtigoEnvError
from artigo_env.core.resources import resource_path
from artigo_env.corpus.decontam.grams import gram_tokens
from artigo_env.corpus.decontam.titles import normalize_title
from artigo_env.corpus.decontam.windex import WIndex
from artigo_env.corpus.manifest import SourceManifestEntry
from artigo_env.corpus.sources.extracts import INDEX, ExtractsSource, read_index
from artigo_env.corpus.stages.context import BuildContext, check_batch, finish
from artigo_env.corpus.stages.normalize import Cleaner, NormalizeStage
from artigo_env.corpus.stages.playable import title_in_body
from artigo_env.corpus.stages.tokenize import TokenizeStage
from artigo_env.corpus.store.reader import CorpusStore
from artigo_env.corpus.types import (
    PlayabilityConfig,
    RawDocument,
    ShardRef,
    StageContext,
    StageResult,
    raw_to_row,
    to_batch,
)
from artigo_env.text.normalize import lower, normalize
from artigo_env.text.tokenizer import tokenize

__all__ = [
    "BODY_CAP",
    "EMPTY_CLEANER",
    "N_TEST",
    "N_VAL",
    "SEED",
    "SPLITS",
    "TitleEntry",
    "WEligibilityStage",
    "WPage",
    "WSplitRow",
    "build_windex",
    "eligible",
    "l_max",
    "load_title_list",
    "load_w_splits",
    "make_splits",
    "measure_snapshot",
    "snapshot_titles",
    "w_splits_path",
    "write_w_splits",
]

SEED = 20261109
N_TEST = 1_000
N_VAL = 200
BODY_CAP = 32_768
L_MAX_STEP = 1_024
W_TEST, W_VAL, W_POOL, W_EXTRA = "w-test", "w-val", "w-pool", "w-extra"
SPLITS = (W_TEST, W_VAL, W_POOL, W_EXTRA)
EXCLUDED_PREFIXES = ("anexo:", "wp:")
EMPTY_CLEANER = Cleaner(boilerplate_prefixes=(), title_suffixes=())
_BATCH = 500


class WSetError(ArtigoEnvError):
    """The W title list, snapshot or split file is unusable."""


# --- title list ------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TitleEntry:
    title: str
    in_complete: bool  # em_completa
    in_list: bool  # em_lista
    sublists: str


def load_title_list(path: Path) -> list[TitleEntry]:
    """Rows of the title list TSV (``#`` comments and the header row are skipped)."""
    entries: list[TitleEntry] = []
    header_seen = False
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip() or line.startswith("#"):
            continue
        cells = line.split("\t")
        if not header_seen:
            header_seen = True
            if cells[0].strip().lower() == "titulo":
                continue
        if len(cells) < 3:
            raise WSetError(f"{path.name}:{number}: expected at least 3 tab-separated columns")
        flags = []
        for cell in cells[1:3]:
            value = cell.strip()
            if value not in ("0", "1"):
                raise WSetError(f"{path.name}:{number}: flag must be 0 or 1, got {value!r}")
            flags.append(value == "1")
        entries.append(
            TitleEntry(
                cells[0].strip(), flags[0], flags[1], cells[3].strip() if len(cells) > 3 else ""
            )
        )
    return entries


# --- snapshot measurement --------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class WPage:
    """One page of a snapshot after S3/S4: resolved title, requested titles and token counts."""

    pageid: int
    title: str
    requested: tuple[str, ...]
    n_body_tokens: int
    n_title_content: int


@dataclass
class SnapshotMeasure:
    pages: list[WPage]
    missing: list[str] = field(default_factory=list)  # requested titles without a page
    empty: list[str] = field(default_factory=list)  # pages removed by S3 (empty text)
    n_index_rows: int = 0


def _snapshot_entry(snapshot_dir: Path) -> SourceManifestEntry:
    return SourceManifestEntry(
        id="w_snapshot",
        kind="wikipedia_extracts",
        state="evaluation",
        priority=0,
        license_id="CC-BY-SA-4.0",
        options={"subdir": snapshot_dir.name},
    )


def _batches(docs: Iterable[RawDocument], size: int) -> Iterator[list[RawDocument]]:
    chunk: list[RawDocument] = []
    for doc in docs:
        chunk.append(doc)
        if len(chunk) >= size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def measure_snapshot(snapshot_dir: Path, ctx: StageContext) -> SnapshotMeasure:
    """Pages of a snapshot with the W build's normalization and token counts."""
    if not (snapshot_dir / INDEX).is_file():
        raise WSetError(f"no snapshot index at {snapshot_dir / INDEX}")
    rows = read_index(snapshot_dir / INDEX)
    requested: dict[int, list[str]] = {}
    missing: list[str] = []
    for row in rows:
        pageid = row.get("pageid")
        if row.get("status") == "ok" and isinstance(pageid, int):
            requested.setdefault(pageid, []).append(str(row["requested"]))
        else:
            missing.append(str(row.get("requested")))
    source = ExtractsSource(_snapshot_entry(snapshot_dir))
    shard = ShardRef(source.source_id, "w", snapshot_dir)
    normalize_stage, tokenize_stage = NormalizeStage(EMPTY_CLEANER), TokenizeStage()
    pages: list[WPage] = []
    empty: list[str] = []
    for chunk in _batches(source.read_shard(shard), _BATCH):
        batch = to_batch(raw_to_row(doc) for doc in chunk)
        normalized = normalize_stage.run(batch, ctx).batch
        kept = set(normalized.column("doc_id").to_pylist())
        empty.extend(doc.title or doc.doc_id for doc in chunk if doc.doc_id not in kept)
        counted = tokenize_stage.run(normalized, ctx).batch
        for row in counted.to_pylist():
            pageid = int(row["doc_id"])
            pages.append(
                WPage(
                    pageid=pageid,
                    title=row["title"],
                    requested=tuple(sorted(set(requested.get(pageid, [])))),
                    n_body_tokens=int(row["n_body_tokens"]),
                    n_title_content=int(row["n_title_content"]),
                )
            )
    return SnapshotMeasure(pages, sorted(missing), sorted(empty), len(rows))


def snapshot_titles(snapshot_dir: Path) -> list[str]:
    """Every requested and resolved title of a snapshot index, missing pages included."""
    titles: set[str] = set()
    for row in read_index(snapshot_dir / INDEX):
        for key in ("requested", "resolved"):
            value = row.get(key)
            if isinstance(value, str) and value.strip():
                titles.add(value)
    return sorted(titles)


# --- eligibility -----------------------------------------------------------------------------


def _content_words(title: str, ctx: StageContext) -> int:
    return sum(
        1
        for tok in tokenize(normalize(title), ctx.tokenizer_hyphen)
        if not ctx.is_stopword(lower(tok.text))
    )


@dataclass
class Eligibility:
    pages: list[WPage]  # eligible pages, ordered by page id
    excluded: dict[str, list[str]]  # reason -> titles (sorted)


def eligible(
    measure: SnapshotMeasure,
    titles: Sequence[TitleEntry],
    ctx: StageContext,
    cap: int = BODY_CAP,
) -> Eligibility:
    """Eligible pages of the main list (SDD 06 §12 items 1 and 5).

    A page belongs to the main list when its resolved title or one of its requested titles is
    an ``em_completa`` title (compared by ``normalize_title``, so the first letter does not
    matter). It is excluded when its title starts with ``Anexo:`` or ``WP:``; when its resolved
    title has no content word, or none of its main-list requested titles has one (so ``@``,
    which resolves to "Arroba (símbolo)", stays out as SDD 06 §12 lists it); or when its body
    has more than ``cap`` tokens. Requested titles that resolve to the same page count once.
    """
    main = {normalize_title(entry.title) for entry in titles if entry.in_complete}
    excluded: dict[str, list[str]] = {
        "not_in_main_list": [],
        "annex_or_project": [],
        "no_content_word": [],
        "too_long": [],
        "missing": sorted(t for t in measure.missing if normalize_title(t) in main),
        "empty": sorted(t for t in measure.empty if normalize_title(t) in main),
    }
    kept: list[WPage] = []
    for page in sorted(measure.pages, key=lambda p: p.pageid):
        main_requested = [r for r in page.requested if normalize_title(r) in main]
        if not main_requested and normalize_title(page.title) not in main:
            excluded["not_in_main_list"].append(page.title)
            continue
        names = [page.title, *main_requested]
        if any(lower(name).startswith(EXCLUDED_PREFIXES) for name in names):
            excluded["annex_or_project"].append(page.title)
            continue
        requested_ok = not main_requested or any(
            _content_words(name, ctx) > 0 for name in main_requested
        )
        if page.n_title_content < 1 or not requested_ok:
            excluded["no_content_word"].append(page.title)
            continue
        if page.n_body_tokens > cap:
            excluded["too_long"].append(page.title)
            continue
        kept.append(page)
    return Eligibility(kept, {key: sorted(value) for key, value in excluded.items()})


# --- splits ----------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class WSplitRow:
    title: str
    split: str
    stratum: str


def _largest_remainder(total: int, sizes: Sequence[int], caps: Sequence[int]) -> list[int]:
    """Integer allocation of ``total`` proportional to ``sizes``, never above ``caps``."""
    population = sum(sizes)
    if total > sum(caps):
        raise WSetError(f"cannot draw {total} articles from {sum(caps)} available")
    if population == 0 or total == 0:
        return [0] * len(sizes)
    exact = [total * size / population for size in sizes]
    alloc = [min(math.floor(value), cap) for value, cap in zip(exact, caps, strict=True)]
    order = sorted(range(len(sizes)), key=lambda i: (-(exact[i] - math.floor(exact[i])), i))
    remaining = total - sum(alloc)
    while remaining > 0:
        progressed = False
        for i in order:
            if remaining == 0:
                break
            if alloc[i] < caps[i]:
                alloc[i] += 1
                remaining -= 1
                progressed = True
        if not progressed:  # pragma: no cover - excluded by the capacity check above
            raise WSetError("allocation did not converge")
    return alloc


def stratum_of(n_title_content: int, tercile: int) -> str:
    words = f"c{n_title_content}" if n_title_content < 4 else "c4+"
    return f"{words}-t{tercile + 1}"


def make_splits(
    pages: Sequence[WPage], seed: int = SEED, n_test: int = N_TEST, n_val: int = N_VAL
) -> list[WSplitRow]:
    """W-test, W-val and W-pool, stratified; rows sorted by title.

    Strata: title content words {1, 2, 3, >=4} x tercile of body tokens (by rank of
    ``(n_body_tokens, title)`` over the eligible pages). Each split's size is allocated over
    the strata by largest remainder; inside a stratum, pages sorted by title are permuted with
    one ``numpy`` generator seeded with ``seed``, visiting strata in label order.
    """
    n = len(pages)
    if n < n_test + n_val:
        raise WSetError(f"{n} eligible pages cannot hold {n_test} + {n_val} articles")
    ranked = sorted(pages, key=lambda p: (p.n_body_tokens, p.title))
    tercile = {page.pageid: 3 * rank // n for rank, page in enumerate(ranked)}
    strata: dict[str, list[WPage]] = {}
    for page in pages:
        label = stratum_of(page.n_title_content, tercile[page.pageid])
        strata.setdefault(label, []).append(page)
    labels = sorted(strata)
    sizes = [len(strata[label]) for label in labels]
    tests = _largest_remainder(n_test, sizes, sizes)
    vals = _largest_remainder(n_val, sizes, [s - t for s, t in zip(sizes, tests, strict=True)])
    rng = np.random.default_rng(seed)
    rows: list[WSplitRow] = []
    for label, n_t, n_v in zip(labels, tests, vals, strict=True):
        members = sorted(strata[label], key=lambda p: p.title)
        for position, index in enumerate(rng.permutation(len(members))):
            split = W_TEST if position < n_t else W_VAL if position < n_t + n_v else W_POOL
            rows.append(WSplitRow(members[int(index)].title, split, label))
    rows.sort(key=lambda row: row.title)
    return rows


def l_max(max_tokens: int) -> int:
    """Smallest multiple of 1,024 not below ``max_tokens`` (at least 1,024), capped at 32,768."""
    return min(BODY_CAP, max(L_MAX_STEP, math.ceil(max_tokens / L_MAX_STEP) * L_MAX_STEP))


def write_w_splits(
    path: Path, rows: Sequence[WSplitRow], *, max_tokens: int, seed: int = SEED
) -> None:
    lines = [
        "# W splits (SDD 06 §9): W-test, W-val, W-pool of the eligible /Completa pages",
        f"# seed {seed}; max body tokens in w-test+w-val {max_tokens}; L_max {l_max(max_tokens)}",
        "title\tsplit\tstratum",
        *(f"{row.title}\t{row.split}\t{row.stratum}" for row in rows),
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(("\n".join(lines) + "\n").encode("utf-8"))


def w_splits_path() -> Path:
    """The package's ``assets/eval/w_splits.tsv`` (error with instructions if it is missing)."""
    node = resource_path("eval", "w_splits.tsv")
    if not node.is_file() or not isinstance(node, Path):
        raise WSetError(
            "assets/eval/w_splits.tsv not found in the package; generate it with "
            "'artigo-env data w-splits --snapshot <dir> --titles <tsv> --out <tsv>'"
        )
    return node


def load_w_splits(path: Path) -> dict[str, str]:
    """``normalize_title(title) -> split`` from a split file."""
    splits: dict[str, str] = {}
    header_seen = False
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip() or line.startswith("#"):
            continue
        cells = line.split("\t")
        if not header_seen:
            header_seen = True
            if cells[0] == "title":
                continue
        if len(cells) < 2 or cells[1] not in (W_TEST, W_VAL, W_POOL):
            raise WSetError(f"{path.name}:{number}: expected title, split (w-test|w-val|w-pool)")
        key = normalize_title(cells[0])
        if key in splits:
            raise WSetError(f"{path.name}:{number}: duplicate title {cells[0]!r}")
        splits[key] = cells[1]
    return splits


# --- the W build stage -----------------------------------------------------------------------


class WEligibilityStage:
    """Fills ``title_in_body`` and ``split`` for a W build; removes nothing.

    A page gets its split from the split file when its title has a content word, its body is not
    above ``cap`` and its title is listed; every other page is ``w-extra`` (it still belongs to
    W for decontamination).
    """

    name = "w_eligibility"
    version = "1"

    def __init__(self, splits: Mapping[str, str], cap: int = BODY_CAP) -> None:
        self.splits = dict(splits)
        self.cap = cap

    def run(self, batch: pa.RecordBatch, ctx: StageContext) -> StageResult:
        check_batch(batch, self.name)
        found: list[bool] = []
        splits: list[str] = []
        titles = batch.column("title").to_pylist()
        bodies = batch.column("body").to_pylist()
        n_bodies = batch.column("n_body_tokens").to_pylist()
        n_contents = batch.column("n_title_content").to_pylist()
        for title, body, n_body, n_content in zip(
            titles, bodies, n_bodies, n_contents, strict=True
        ):
            found.append(title_in_body(title or "", body or "", ctx))
            ok = (n_content or 0) >= 1 and (n_body or 0) <= self.cap
            listed = self.splits.get(normalize_title(title or "")) if ok else None
            splits.append(listed or W_EXTRA)
        return finish(
            self.name, batch, [None] * batch.num_rows, {"title_in_body": found, "split": splits}
        )


def w_summary(docs: pa.Table, splits: Mapping[str, str], cap: int = BODY_CAP) -> dict[str, object]:
    """W-specific counts for the report of an evaluation build."""
    split_col = docs.column("split").to_pylist()
    in_body = docs.column("title_in_body").to_pylist()
    n_body = docs.column("n_body_tokens").to_pylist()
    n_content = docs.column("n_title_content").to_pylist()
    titles = {normalize_title(t or "") for t in docs.column("title").to_pylist()}
    per_split = {s: split_col.count(s) for s in SPLITS}
    false_per_split = {
        s: sum(1 for sp, ok in zip(split_col, in_body, strict=True) if sp == s and ok is False)
        for s in SPLITS
    }
    evaluated = [n for n, s in zip(n_body, split_col, strict=True) if s in (W_TEST, W_VAL)]
    max_eval = max(evaluated, default=0)
    return {
        "pages": len(split_col),
        "per_split": per_split,
        "title_in_body_false": false_per_split,
        "too_long": sum(1 for n in n_body if n > cap),
        "no_content_word": sum(1 for n in n_content if (n or 0) < 1),
        "split_titles_not_found": sum(1 for key in splits if key not in titles),
        "max_body_tokens_test_val": max_eval,
        "l_max": l_max(max_eval) if evaluated else None,
    }


# --- W index ---------------------------------------------------------------------------------


def build_windex(store: CorpusStore, extra_titles: Iterable[str] = ()) -> WIndex:
    """W index over every document of a compiled W corpus plus ``extra_titles``."""
    hyphen = str(store.manifest.get("tokenizer_hyphen", "split"))

    def docs() -> Iterator[tuple[str, list[str]]]:
        for i in range(store.n_docs):
            yield store.title_text(i), gram_tokens(store.body_text(i), hyphen)

    return WIndex.build(docs(), extra_titles)


def default_context(stopwords_ctx: BuildContext | None = None) -> BuildContext:
    """Context for measuring a snapshot: the default stopwords, identity lemmas (unused)."""
    if stopwords_ctx is not None:
        return stopwords_ctx
    return BuildContext(PlayabilityConfig(max_body_tokens=BODY_CAP), lemmatizer="identity")
