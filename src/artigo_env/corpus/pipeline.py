"""Corpus build orchestration: one YAML config to a compiled, reproducible corpus (SDD 06 §1).

Passes of a build (plan F3, WP3.5 and decision 3):

* P1, one task per shard, in parallel: S1 ingest -> S2 origin -> S3 normalize -> S4 tokenize ->
  S5 playable (or, for evaluation builds, the W eligibility stage), written to
  ``work/<build_id>/playable/<source>/<shard>/`` with a ``_DONE`` marker;
* P2, S6 decontamination against a W index in three passes (A interest, B document frequency,
  C judgement), each per shard and checkpointed;
* P3, S7 dedup over the whole build (``dedup.py``), then S8 annotate and S9 partition;
* S10 compile with ``CorpusWriter`` in ``(source_id, doc_id)`` order, then S11 report.

Shards whose ``_DONE`` marker exists are reused, so an interrupted build resumes where it
stopped. Every output depends only on the inputs, never on the number of workers or on the
interruption: the same config and raw data give the same ``build_id`` and byte-identical files
(``report.md`` alone carries a date).

Build kinds: ``train`` (every source ``accepted``; REQ-COR-002), ``evaluation`` (sources in state
``evaluation``; the set W), ``demo`` (``accepted``), ``measurement`` (any state, same processing as
``train``, ``trainable: false``; measures a pending source before the author accepts it).
"""

from __future__ import annotations

import concurrent.futures as cf
import json
import multiprocessing
import os
import shutil
import sys
import time
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypeVar

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import yaml

from artigo_env.core.config import ConfigBase, canonical_json, sha256_hex
from artigo_env.core.errors import ArtigoEnvError
from artigo_env.core.paths import data_root as configured_data_root
from artigo_env.corpus import sources as _sources  # noqa: F401  (registers the source kinds)
from artigo_env.corpus.decontam.coverage import (
    FREQ_LIMIT,
    DocFrequency,
    TrainDoc,
    WGramIndex,
    hashes_of_interest,
)
from artigo_env.corpus.decontam.grams import gram_tokens
from artigo_env.corpus.decontam.windex import WIndex, directory_hash
from artigo_env.corpus.dedup import dedup_batch
from artigo_env.corpus.fetch import sha256_file
from artigo_env.corpus.manifest import (
    SourceManifestEntry,
    get_source,
    load_sources,
    require_trainable,
)
from artigo_env.corpus.report import StageCounter, distributions, write_report
from artigo_env.corpus.sources.registry import open_source
from artigo_env.corpus.stages import (
    AnnotateStage,
    BuildContext,
    Cleaner,
    NormalizeStage,
    OriginStage,
    PartitionStage,
    PlayableStage,
    TokenizeStage,
)
from artigo_env.corpus.store.reader import CorpusStore
from artigo_env.corpus.store.writer import CorpusWriter, write_parquet
from artigo_env.corpus.types import (
    DOC_SCHEMA,
    REASONS,
    PipelineStage,
    PlayabilityConfig,
    RawDocument,
    ShardRef,
    raw_to_row,
    to_batch,
)
from artigo_env.corpus.wset import (
    BODY_CAP,
    EMPTY_CLEANER,
    WEligibilityStage,
    build_windex,
    load_w_splits,
    snapshot_titles,
    w_splits_path,
    w_summary,
)
from artigo_env.text import guess as _text_guess
from artigo_env.text import normalize as _text_normalize
from artigo_env.text import tokenizer as _text_tokenizer
from artigo_env.text.lemmatizers.registry import get_lemmatizer
from artigo_env.text.stopwords import StopwordSpec

__all__ = [
    "BUILD_KINDS",
    "BuildError",
    "BuildResult",
    "CorpusConfig",
    "build",
    "build_id",
    "build_w_index",
    "default_workers",
    "load_config",
    "peak_memory_bytes",
    "text_core_version",
]

PIPELINE_VERSION = 1
BUILD_KINDS = ("train", "evaluation", "demo", "measurement")
TRAINABLE_KINDS = ("train", "demo")
BATCH_ROWS = 2_000
DONE = "_DONE"
WINDEX_DIR = "windex"
POST_STAGES = ("decontam", "dedup", "annotate", "partition", "compile")
_HYPHEN_MODES = ("split", "join")

T = TypeVar("T")
R = TypeVar("R")
Log = Callable[[str], None]


class BuildError(ArtigoEnvError):
    """A corpus build cannot start or cannot finish."""


# --- configuration ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CorpusConfig(ConfigBase):
    """A corpus build (``configs/corpus/<name>.yaml``).

    ``workers`` is an execution setting: it never changes the output and is left out of the
    ``build_id`` (so is the location of ``w_splits``, which enters by content). ``w_splits``
    (evaluation builds) is a split file path, absolute or relative to ``ARTIGO_DATA_ROOT``;
    ``None`` means the package's ``assets/eval/w_splits.tsv``.
    """

    name: str
    kind: str
    sources: tuple[str, ...]
    shards: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    playability: PlayabilityConfig = field(default_factory=PlayabilityConfig)
    tokenizer_hyphen: str = "split"
    stopwords: StopwordSpec = field(default_factory=StopwordSpec)
    lemmatizer: str = "open-tables"
    lemma_views: tuple[str, ...] = ("open-tables",)
    optional_lemma_views: tuple[str, ...] = ()
    w_index: str | None = None
    w_splits: str | None = None
    workers: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("name must be a non-empty string")
        if self.kind not in BUILD_KINDS:
            raise ValueError(f"kind must be one of {BUILD_KINDS}, got {self.kind!r}")
        sources = tuple(self.sources)
        if not sources or len(set(sources)) != len(sources):
            raise ValueError("sources must be a non-empty list of distinct source ids")
        object.__setattr__(self, "sources", sources)
        if not isinstance(self.shards, Mapping):
            raise ValueError("shards must map source ids to lists of shard ids")
        shards = {str(k): tuple(str(s) for s in v) for k, v in sorted(self.shards.items())}
        unknown = sorted(set(shards) - set(sources))
        if unknown:
            raise ValueError(f"shards names sources that are not in the build: {unknown}")
        object.__setattr__(self, "shards", shards)
        if self.tokenizer_hyphen not in _HYPHEN_MODES:
            raise ValueError(f"tokenizer_hyphen must be one of {_HYPHEN_MODES}")
        views = tuple(self.lemma_views)
        if not views or len(set(views)) != len(views):
            raise ValueError("lemma_views must be a non-empty list of distinct lemmatizer names")
        object.__setattr__(self, "lemma_views", views)
        optional = tuple(v for v in self.optional_lemma_views if v not in views)
        object.__setattr__(self, "optional_lemma_views", optional)
        if self.workers is not None and (
            isinstance(self.workers, bool) or not isinstance(self.workers, int) or self.workers < 1
        ):
            raise ValueError("workers must be a positive integer or null")
        if self.kind == "evaluation" and self.w_index is not None:
            raise ValueError("an evaluation build is W itself: w_index must be null")
        if self.kind != "evaluation" and self.w_splits is not None:
            raise ValueError("w_splits applies only to evaluation builds")

    def identity(self) -> dict[str, Any]:
        """The configuration as it enters the ``build_id``.

        ``workers`` never changes the output and the ``w_splits`` location does not matter:
        the split file enters the ``build_id`` by its SHA-256.
        """
        data = self.to_dict()
        data.pop("workers")
        data.pop("w_splits")
        return data


def load_config(path: Path) -> CorpusConfig:
    """Read and validate a corpus build config (YAML)."""
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise BuildError(f"cannot read corpus config {path}: {error}") from None
    if not isinstance(data, Mapping):
        raise BuildError(f"corpus config {path} must be a mapping")
    try:
        return CorpusConfig.from_dict(data)
    except (TypeError, ValueError) as error:
        raise BuildError(f"corpus config {path}: {error}") from None


def default_workers() -> int:
    """CPU count minus two, at least one."""
    return max(1, (os.cpu_count() or 1) - 2)


# --- identity --------------------------------------------------------------------------------


def text_core_version() -> str:
    """SHA-256 of the sources of the ``text`` core modules the pipeline depends on.

    The ``text`` core has no version number; its normalizer, tokenizer and guess cleaner are
    hashed instead (line endings normalized, so a Windows checkout gives the same value).
    """
    digest_parts = []
    for module in (_text_normalize, _text_tokenizer, _text_guess):
        source = Path(str(module.__file__)).read_bytes().replace(b"\r\n", b"\n")
        digest_parts.append(f"{module.__name__}:{sha256_hex(source.decode('utf-8'))}")
    return sha256_hex("\n".join(digest_parts))


def build_id(
    config: CorpusConfig,
    sources: Sequence[Mapping[str, Any]],
    w_index: str | None,
    components: Mapping[str, Any] | None = None,
) -> str:
    """First 16 hex of the SHA-256 of everything that determines the output (REQ-COR-016).

    ``sources`` describe the raw data (revision and SHA-256 per shard), ``w_index`` is the
    ``WIndex`` content hash (``None`` without decontamination) and ``components`` carries the
    text core version, the context hash (stopwords, playability lemmatizer, playability), the
    stage versions, the lemma views and the split file hash.
    """
    payload = {
        "pipeline_version": PIPELINE_VERSION,
        "config": sha256_hex(canonical_json(config.identity())),
        "sources": list(sources),
        "w_index": w_index,
        "components": dict(components or {}),
    }
    return sha256_hex(canonical_json(payload))[:16]


# --- results ---------------------------------------------------------------------------------


@dataclass(frozen=True)
class BuildResult:
    build_id: str
    out_dir: Path
    n_docs: int
    reused: bool  # the compiled corpus already existed; nothing was done
    shards_processed: int = 0
    shards_reused: int = 0
    report: Mapping[str, Any] = field(default_factory=dict)


# --- tasks (top-level so that worker processes can import them) ------------------------------


@dataclass(frozen=True)
class ShardTask:
    kind: str
    entry: SourceManifestEntry
    shard: ShardRef
    ctx: BuildContext
    out_dir: Path
    splits: Mapping[str, str] | None = None


@dataclass(frozen=True)
class ShardOutcome:
    source_id: str
    shard_id: str
    counts: Mapping[str, Any]
    peak_memory: int | None
    seconds: float


def peak_memory_bytes() -> int | None:
    """Peak resident memory of this process, or ``None`` when it cannot be measured."""
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class _Counters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = _Counters()
        counters.cb = ctypes.sizeof(_Counters)
        try:
            kernel32 = ctypes.WinDLL("kernel32")
            psapi = ctypes.WinDLL("psapi")
            kernel32.GetCurrentProcess.restype = wintypes.HANDLE
            psapi.GetProcessMemoryInfo.argtypes = [
                wintypes.HANDLE,
                ctypes.POINTER(_Counters),
                wintypes.DWORD,
            ]
            handle = kernel32.GetCurrentProcess()
            ok = psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb)
        except (OSError, ctypes.ArgumentError):
            return None
        return int(counters.PeakWorkingSetSize) if ok else None
    else:
        import resource

        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(peak) if sys.platform == "darwin" else int(peak) * 1024


def shard_stages(kind: str, splits: Mapping[str, str] | None) -> list[PipelineStage]:
    """The per-shard stages of a build kind, in order."""
    stages: list[PipelineStage] = [OriginStage()]
    if kind == "evaluation":
        stages += [NormalizeStage(EMPTY_CLEANER), TokenizeStage()]
        stages.append(WEligibilityStage(splits or {}, BODY_CAP))
    else:
        stages += [NormalizeStage(Cleaner()), TokenizeStage(), PlayableStage()]
    return stages


def _batched(docs: Iterable[RawDocument], size: int) -> Iterator[list[RawDocument]]:
    chunk: list[RawDocument] = []
    for doc in docs:
        chunk.append(doc)
        if len(chunk) >= size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def _replace_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)


def _write_json(path: Path, data: Any) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes((canonical_json(data) + "\n").encode("utf-8"))
    tmp.replace(path)


def _parquet_writer(path: Path) -> pq.ParquetWriter:
    return pq.ParquetWriter(path, DOC_SCHEMA, compression="zstd", compression_level=3)


def run_playable_shard(task: ShardTask) -> ShardOutcome:
    """P1 for one shard: read in batches, run the per-shard stages, keep the survivors."""
    start = time.perf_counter()
    source = open_source(task.entry)
    stages = shard_stages(task.kind, task.splits)
    counter = StageCounter()
    out = task.out_dir
    _replace_dir(out)
    tmp = out / "docs.parquet.tmp"
    writer = _parquet_writer(tmp)
    try:
        for chunk in _batched(source.read_shard(task.shard), BATCH_ROWS):
            batch = to_batch(raw_to_row(doc) for doc in chunk)
            counter.add("ingest", batch.num_rows, batch.num_rows)
            for stage in stages:
                n_in = batch.num_rows
                result = stage.run(batch, task.ctx)
                batch = result.batch
                counter.add(stage.name, n_in, batch.num_rows, result.removed)
            if batch.num_rows:
                writer.write_batch(batch)
    finally:
        writer.close()
    tmp.replace(out / "docs.parquet")
    if not counter.stages:  # an empty shard still reports every stage
        counter.add("ingest", 0, 0)
        for stage in stages:
            counter.add(stage.name, 0, 0, dict.fromkeys(REASONS.get(stage.name, ()), 0))
    counts = counter.to_dict()
    _write_json(out / "counts.json", counts)
    (out / DONE).write_bytes(b"")
    return ShardOutcome(
        task.shard.source_id,
        task.shard.shard_id,
        counts,
        peak_memory_bytes(),
        time.perf_counter() - start,
    )


@dataclass(frozen=True)
class DecontamTask:
    source_id: str
    shard_id: str
    docs: Path  # playable docs.parquet of the shard
    out_dir: Path
    windex_dir: Path
    hyphen: str
    interest: Path | None = None  # pass B input
    frequent: Path | None = None  # pass C input


_GRAMS_CACHE: dict[str, WGramIndex] = {}
_WINDEX_CACHE: dict[str, WIndex] = {}


def _grams(directory: Path) -> WGramIndex:
    key = str(directory)
    if key not in _GRAMS_CACHE:
        _GRAMS_CACHE[key] = WGramIndex.load(directory / "grams.npy")
    return _GRAMS_CACHE[key]


def _windex(directory: Path) -> WIndex:
    key = str(directory)
    if key not in _WINDEX_CACHE:
        _WINDEX_CACHE[key] = WIndex.load(directory)
    return _WINDEX_CACHE[key]


def _iter_doc_batches(path: Path, columns: Sequence[str] | None = None) -> Iterator[pa.RecordBatch]:
    parquet = pq.ParquetFile(path)
    yield from parquet.iter_batches(batch_size=BATCH_ROWS, columns=columns)


def _save_npy(path: Path, array: np.ndarray[Any, Any]) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as handle:
        np.save(handle, array, allow_pickle=False)
    tmp.replace(path)


def run_interest(task: DecontamTask) -> ShardOutcome:
    """Pass A for one shard: 13-gram hashes of the documents that share a 13-gram with W."""
    start = time.perf_counter()
    grams = _grams(task.windex_dir)
    parts: list[np.ndarray[Any, Any]] = []
    for batch in _iter_doc_batches(task.docs, ["body"]):
        for body in batch.column("body").to_pylist():
            hashes = hashes_of_interest(gram_tokens(body or "", task.hyphen), grams)
            if hashes is not None:
                parts.append(hashes)
    task.out_dir.mkdir(parents=True, exist_ok=True)
    union = np.unique(np.concatenate(parts)) if parts else np.empty(0, dtype=np.uint64)
    _save_npy(task.out_dir / "interest.npy", union.astype(np.uint64))
    (task.out_dir / DONE).write_bytes(b"")
    return ShardOutcome(
        task.source_id, task.shard_id, {}, peak_memory_bytes(), time.perf_counter() - start
    )


def run_frequency(task: DecontamTask) -> ShardOutcome:
    """Pass B for one shard: document frequency of the hashes of interest."""
    start = time.perf_counter()
    assert task.interest is not None
    frequency = DocFrequency(np.load(task.interest, allow_pickle=False))
    for batch in _iter_doc_batches(task.docs, ["body"]):
        for body in batch.column("body").to_pylist():
            frequency.add(gram_tokens(body or "", task.hyphen))
    task.out_dir.mkdir(parents=True, exist_ok=True)
    _save_npy(task.out_dir / "counts.npy", frequency.counts)
    (task.out_dir / DONE).write_bytes(b"")
    return ShardOutcome(
        task.source_id, task.shard_id, {}, peak_memory_bytes(), time.perf_counter() - start
    )


def run_judge(task: DecontamTask) -> ShardOutcome:
    """Pass C for one shard: title, 13-gram and near-duplicate barriers; keep the clean ones."""
    start = time.perf_counter()
    assert task.frequent is not None
    windex = _windex(task.windex_dir)
    frequent = np.load(task.frequent, allow_pickle=False)
    reasons_allowed = REASONS["decontam"]
    removed = dict.fromkeys(reasons_allowed, 0)
    n_in = n_out = 0
    _replace_dir(task.out_dir)
    tmp = task.out_dir / "docs.parquet.tmp"
    writer = _parquet_writer(tmp)
    try:
        for batch in _iter_doc_batches(task.docs):
            keep: list[bool] = []
            ids = batch.column("doc_id").to_pylist()
            titles = batch.column("title").to_pylist()
            bodies = batch.column("body").to_pylist()
            for doc_id, title, body in zip(ids, titles, bodies, strict=True):
                doc = TrainDoc(doc_id, title or "", gram_tokens(body or "", task.hyphen))
                verdict = windex.judge(doc, frequent)
                if verdict.reason is not None:
                    removed[verdict.reason] += 1
                keep.append(verdict.reason is None)
            n_in += batch.num_rows
            kept = batch.filter(pa.array(keep, type=pa.bool_()))
            n_out += kept.num_rows
            if kept.num_rows:
                writer.write_batch(kept)
    finally:
        writer.close()
    tmp.replace(task.out_dir / "docs.parquet")
    counts = {"decontam": {"in": n_in, "out": n_out, "removed": removed}}
    _write_json(task.out_dir / "counts.json", counts)
    (task.out_dir / DONE).write_bytes(b"")
    return ShardOutcome(
        task.source_id, task.shard_id, counts, peak_memory_bytes(), time.perf_counter() - start
    )


# --- the build -------------------------------------------------------------------------------


@dataclass
class _Plan:
    config: CorpusConfig
    entries: list[SourceManifestEntry]
    shards: list[tuple[SourceManifestEntry, ShardRef]]
    source_infos: list[dict[str, Any]]
    ctx: BuildContext
    views: dict[str, str]  # lemmatizer name -> content hash
    windex_dir: Path | None
    windex_hash: str | None
    splits: dict[str, str] | None
    splits_hash: str | None
    components: dict[str, Any]
    build_id: str


def _check_states(config: CorpusConfig, entries: Sequence[SourceManifestEntry]) -> None:
    for entry in entries:
        if config.kind in ("train", "demo"):
            require_trainable(entry)  # REQ-COR-002
        elif config.kind == "evaluation" and entry.state != "evaluation":
            raise BuildError(
                f"source {entry.id!r} is {entry.state!r}: an evaluation build only reads "
                "sources in state 'evaluation'"
            )


def _shard_sha256(entry: SourceManifestEntry, shard: ShardRef, raw_root: Path) -> str:
    """SHA-256 of a shard's raw data: the pinned value when the manifest pins the file."""
    path = shard.path
    if path.is_dir():  # a TextExtracts snapshot: its index pins every page by lastrevid
        return sha256_file(path / "index.jsonl")
    try:
        relative = path.relative_to(entry.source_dir(raw_root)).as_posix()
    except ValueError:
        relative = None
    for pinned in entry.files:
        if pinned.path == relative:
            size = path.stat().st_size
            if size != pinned.size:
                raise BuildError(
                    f"{entry.id}: {relative} has {size} bytes, the manifest pins {pinned.size}; "
                    f"run: artigo-env data fetch {entry.id}"
                )
            return pinned.sha256
    return sha256_file(path)


def _resolve_views(config: CorpusConfig) -> dict[str, str]:
    views: dict[str, str] = {}
    for name in config.lemma_views:
        views[name] = get_lemmatizer(name).content_hash()
    for name in config.optional_lemma_views:
        try:
            views[name] = get_lemmatizer(name).content_hash()
        except ArtigoEnvError:
            continue  # optional and not available in this installation
    return views


def _resolve_splits_path(config: CorpusConfig, root: Path) -> Path:
    if config.w_splits is None:
        return w_splits_path()
    path = Path(config.w_splits)
    return path if path.is_absolute() else root / path


def _plan(config: CorpusConfig, sources: Mapping[str, SourceManifestEntry], root: Path) -> _Plan:
    raw_root = root / "raw"
    entries = [get_source(source_id, sources) for source_id in config.sources]
    _check_states(config, entries)
    if config.kind == "train" and config.w_index is None:
        raise BuildError(
            "a train build needs a W index for decontamination (set w_index to the build id of "
            "the W corpus, built with configs/corpus/w-v1.yaml)"
        )
    shards: list[tuple[SourceManifestEntry, ShardRef]] = []
    infos: list[dict[str, Any]] = []
    for entry in entries:
        available = list(open_source(entry).iter_shards(raw_root))
        wanted = config.shards.get(entry.id)
        if wanted:
            by_id = {shard.shard_id: shard for shard in available}
            missing = [shard_id for shard_id in wanted if shard_id not in by_id]
            if missing:
                raise BuildError(
                    f"source {entry.id!r} has no shard(s) {missing}; available: {sorted(by_id)}"
                )
            chosen = [by_id[shard_id] for shard_id in sorted(set(wanted))]
        else:
            chosen = available
        infos.append(
            {
                "id": entry.id,
                "kind": entry.kind,
                "revision": entry.revision,
                "priority": entry.priority,
                "license_id": entry.license_id,
                "redistributable": entry.redistributable,
                "variant": entry.variant,
                "options": dict(entry.options),
                "shards": [
                    {"shard_id": s.shard_id, "sha256": _shard_sha256(entry, s, raw_root)}
                    for s in chosen
                ],
            }
        )
        shards.extend((entry, shard) for shard in chosen)
    ctx = BuildContext(
        config.playability, config.tokenizer_hyphen, config.stopwords, config.lemmatizer
    )
    try:
        context_hash = ctx.content_hash()
        views = _resolve_views(config)
    except ArtigoEnvError as error:
        raise BuildError(f"cannot load the text components of the build: {error}") from None
    windex_dir: Path | None = None
    windex_hash: str | None = None
    if config.w_index is not None:
        windex_dir = root / "corpus" / config.w_index / WINDEX_DIR
        if not (windex_dir / "meta.json").is_file():
            raise BuildError(
                f"no W index at corpus/{config.w_index}/{WINDEX_DIR}: build W first "
                "(artigo-env data build configs/corpus/w-v1.yaml) or run "
                f"'artigo-env data windex {config.w_index}'"
            )
        windex_hash = directory_hash(windex_dir)
    splits: dict[str, str] | None = None
    splits_hash: str | None = None
    if config.kind == "evaluation":
        path = _resolve_splits_path(config, root)
        if not path.is_file():
            raise BuildError(f"W split file not found: {path}")
        splits = load_w_splits(path)
        splits_hash = sha256_file(path)
    stage_names = [s.name for s in shard_stages(config.kind, splits)]
    stage_versions = {s.name: s.version for s in shard_stages(config.kind, splits)}
    stage_versions.update({"annotate": AnnotateStage.version, "partition": PartitionStage.version})
    components = {
        "text_core": text_core_version(),
        "context": context_hash,
        "stages": stage_versions,
        "stage_order": stage_names,
        "lemma_views": views,
        "w_splits": splits_hash,
    }
    identity = build_id(config, infos, windex_hash, components)
    return _Plan(
        config,
        entries,
        shards,
        infos,
        ctx,
        views,
        windex_dir,
        windex_hash,
        splits,
        splits_hash,
        components,
        identity,
    )


class _Runner:
    """Runs tasks inline (``workers == 1``) or in one shared process pool."""

    def __init__(self, workers: int) -> None:
        self.workers = workers
        self._pool: cf.ProcessPoolExecutor | None = None

    def map(self, fn: Callable[[T], R], tasks: Sequence[T]) -> Iterator[R]:
        if not tasks:
            return
        if self.workers <= 1 or len(tasks) == 1:
            for task in tasks:
                yield fn(task)
            return
        if self._pool is None:
            self._pool = cf.ProcessPoolExecutor(
                max_workers=self.workers, mp_context=multiprocessing.get_context("spawn")
            )
        futures = [self._pool.submit(fn, task) for task in tasks]
        try:
            for future in cf.as_completed(futures):
                yield future.result()
        except BaseException:
            for future in futures:
                future.cancel()
            raise

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=True, cancel_futures=True)
            self._pool = None


def _shard_dir(work: Path, stage: str, source_id: str, shard_id: str) -> Path:
    return work / stage / source_id / shard_id


def _read_counts(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return result


def _concat(paths: Sequence[Path]) -> pa.Table:
    tables = [pq.read_table(path, schema=DOC_SCHEMA) for path in paths]
    if not tables:
        return DOC_SCHEMA.empty_table()
    return pa.concat_tables(tables).combine_chunks()


def _empty_batch() -> pa.RecordBatch:
    return pa.RecordBatch.from_pylist([], schema=DOC_SCHEMA)


def _one_batch(table: pa.Table) -> pa.RecordBatch:
    """The whole table as a single record batch."""
    if table.num_rows == 0:
        return _empty_batch()
    batches = table.combine_chunks().to_batches()
    if len(batches) == 1:
        return batches[0]
    return pa.Table.from_batches(batches, schema=DOC_SCHEMA).combine_chunks().to_batches()[0]


def _run_stage_on_table(
    stage: PipelineStage, table: pa.Table, ctx: BuildContext
) -> tuple[pa.Table, dict[str, int]]:
    batches = table.to_batches(max_chunksize=BATCH_ROWS) or [_empty_batch()]
    outputs: list[pa.RecordBatch] = []
    removed: dict[str, int] = {}
    for batch in batches:
        result = stage.run(batch, ctx)
        outputs.append(result.batch)
        for reason, count in result.removed.items():
            removed[reason] = removed.get(reason, 0) + count
    return pa.Table.from_batches(outputs, schema=DOC_SCHEMA), removed


def _dir_size(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def build(
    config: CorpusConfig,
    *,
    data_root: Path | None = None,
    sources: Mapping[str, SourceManifestEntry] | None = None,
    resume: bool = True,
    recompile: bool = False,
    log: Log | None = None,
) -> BuildResult:
    """Build the corpus described by ``config`` under ``<data_root>/corpus/<build_id>/``.

    ``sources`` defaults to the package manifest. With ``resume`` (default) finished shards are
    reused; ``recompile`` deletes an existing compiled corpus of the same ``build_id`` and
    compiles it again from the work files.
    """
    say = log or (lambda message: None)
    started = time.perf_counter()
    root = configured_data_root() if data_root is None else Path(data_root)
    manifest = load_sources() if sources is None else sources
    plan = _plan(config, manifest, root)
    out_dir = root / "corpus" / plan.build_id
    work = root / "corpus" / "work" / plan.build_id
    say(f"build {plan.build_id} ({config.name}, {config.kind}): {len(plan.shards)} shard(s)")
    if out_dir.exists() and (out_dir / "report.json").is_file() and not recompile:
        report = json.loads((out_dir / "report.json").read_text(encoding="utf-8"))
        say(f"corpus already compiled at corpus/{plan.build_id}; nothing to do")
        return BuildResult(
            plan.build_id, out_dir, int(report["n_docs"]), True, 0, len(plan.shards), report
        )
    if out_dir.exists():
        shutil.rmtree(out_dir)
    if not resume and work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True, exist_ok=True)
    workers = config.workers or default_workers()
    runner = _Runner(workers)
    peaks: list[int] = []
    processed = reused = 0
    try:
        # P1: per-shard stages
        tasks: list[ShardTask] = []
        for entry, shard in plan.shards:
            directory = _shard_dir(work, "playable", entry.id, shard.shard_id)
            if (directory / DONE).is_file():
                reused += 1
                continue
            tasks.append(ShardTask(config.kind, entry, shard, plan.ctx, directory, plan.splits))
        if reused:
            say(f"P1: reusing {reused} finished shard(s)")
        for outcome in runner.map(run_playable_shard, tasks):
            processed += 1
            if outcome.peak_memory:
                peaks.append(outcome.peak_memory)
            out = outcome.counts.get("ingest", {}).get("in", 0)
            last = list(outcome.counts)[-1]
            say(
                f"P1: {outcome.source_id}/{outcome.shard_id} done in {outcome.seconds:.1f} s: "
                f"{out:,} read, {outcome.counts[last]['out']:,} kept"
            )
        playable = {
            (e.id, s.shard_id): _shard_dir(work, "playable", e.id, s.shard_id)
            for e, s in plan.shards
        }
        totals = StageCounter()
        by_source: dict[str, StageCounter] = {e.id: StageCounter() for e in plan.entries}
        for (source_id, _), directory in playable.items():
            counts = _read_counts(directory / "counts.json")
            totals.merge(counts)
            by_source[source_id].merge(counts)
        shard_order = ["ingest", *plan.components["stage_order"]]
        totals.reorder(shard_order)
        for counter in by_source.values():
            counter.reorder(shard_order)
        current = {key: directory / "docs.parquet" for key, directory in playable.items()}

        # P2: decontamination (S6)
        decontam: dict[str, Any]
        notes: list[str] = []
        if config.kind == "evaluation":
            decontam = {"status": "not-applicable"}
        elif plan.windex_dir is None:
            decontam = {"status": "skipped"}
            notes.append("decontam: skipped (no W index configured; not for training)")
            n = totals.stages[list(totals.stages)[-1]]["out"]
            totals.add("decontam", n, n, dict.fromkeys(REASONS["decontam"], 0), skipped=True)
            for counter in by_source.values():
                n = counter.stages[list(counter.stages)[-1]]["out"]
                counter.add("decontam", n, n, dict.fromkeys(REASONS["decontam"], 0), skipped=True)
        else:
            current = _decontaminate(plan, work, current, runner, totals, by_source, peaks, say)
            decontam = {
                "status": "applied",
                "w_index": config.w_index,
                "w_index_hash": plan.windex_hash,
            }
        if peak := peak_memory_bytes():
            peaks.append(peak)

        # P3: dedup (S7), annotate (S8), partition (S9)
        ordered = [current[key] for key in sorted(current)]
        table = _concat(ordered)
        priorities = {entry.id: entry.priority for entry in plan.entries}
        if config.kind != "evaluation":
            table = _dedup(table, priorities, plan, work, totals, by_source, say)
        stage_list: list[PipelineStage] = [AnnotateStage()]
        if config.kind != "evaluation":
            stage_list.append(PartitionStage())
        for stage in stage_list:
            n_in = table.num_rows
            table, removed = _run_stage_on_table(stage, table, plan.ctx)
            totals.add(stage.name, n_in, table.num_rows, removed)
            for source_id, counter in by_source.items():
                n = _count_source(table, source_id)
                counter.add(stage.name, n, n, removed={})
        table = table.sort_by([("source_id", "ascending"), ("doc_id", "ascending")])

        # S10 compile
        say(f"S10: compiling {table.num_rows:,} documents")
        report_extra: dict[str, Any] = {}
        n_docs = _compile(plan, table, out_dir, decontam)
        totals.add("compile", n_docs, n_docs)
        for source_id, counter in by_source.items():
            n = _count_source(table, source_id)
            counter.add("compile", n, n)
        if config.kind == "evaluation":
            assert plan.splits is not None
            report_extra["w"] = w_summary(table, plan.splits)
            windex_hash = _write_windex(plan, out_dir, root)
            report_extra["w"]["windex_hash"] = windex_hash
            say(f"W index written to corpus/{plan.build_id}/{WINDEX_DIR}")

        # S11 report
        order = ["ingest", *plan.components["stage_order"], *POST_STAGES]
        totals.reorder(order)
        for counter in by_source.values():
            counter.reorder(order)
        report = {
            "report_version": 1,
            "build_id": plan.build_id,
            "name": config.name,
            "kind": config.kind,
            "trainable": config.kind in TRAINABLE_KINDS,
            "config": config.identity(),
            "sources": plan.source_infos,
            "decontam": decontam,
            "stage_order": totals.order(),
            "stages": totals.to_dict(),
            "by_source": {sid: counter.to_dict() for sid, counter in sorted(by_source.items())},
            "n_docs": n_docs,
            "distributions": distributions(table),
            "notes": notes,
            **report_extra,
        }
        if peak := peak_memory_bytes():
            peaks.append(peak)
        run: dict[str, Any] = {
            "generated": datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC"),
            "wall_seconds": time.perf_counter() - started,
            "peak_memory_bytes": max(peaks) if peaks else None,
            "shards_processed": processed,
            "shards_reused": reused,
            "size_bytes": _dir_size(out_dir),
            "workers": workers,
        }
        write_report(out_dir, report, run)
        say(
            f"done: {n_docs:,} documents in corpus/{plan.build_id} "
            f"({run['size_bytes'] / 2**20:,.1f} MiB, {run['wall_seconds']:.1f} s)"
        )
        return BuildResult(plan.build_id, out_dir, n_docs, False, processed, reused, report)
    finally:
        runner.close()


def _count_source(table: pa.Table, source_id: str) -> int:
    if table.num_rows == 0:
        return 0
    return int(pc.sum(pc.equal(table.column("source_id"), source_id)).as_py() or 0)


def _decontaminate(
    plan: _Plan,
    work: Path,
    current: dict[tuple[str, str], Path],
    runner: _Runner,
    totals: StageCounter,
    by_source: dict[str, StageCounter],
    peaks: list[int],
    say: Log,
) -> dict[tuple[str, str], Path]:
    windex_dir = plan.windex_dir
    assert windex_dir is not None
    hyphen = plan.config.tokenizer_hyphen
    keys = sorted(current)

    def task(stage: str, key: tuple[str, str], **extra: Any) -> DecontamTask:
        return DecontamTask(
            key[0],
            key[1],
            current[key],
            _shard_dir(work, stage, *key),
            windex_dir,
            hyphen,
            **extra,
        )

    def pending(stage: str, **extra: Any) -> list[DecontamTask]:
        return [
            task(stage, key, **extra)
            for key in keys
            if not (_shard_dir(work, stage, *key) / DONE).is_file()
        ]

    def run(fn: Callable[[DecontamTask], ShardOutcome], tasks: list[DecontamTask]) -> None:
        for outcome in runner.map(fn, tasks):
            if outcome.peak_memory:
                peaks.append(outcome.peak_memory)

    say("S6: pass A (13-grams shared with W)")
    run(run_interest, pending("decontam_a"))
    union_path = work / "decontam_interest.npy"
    if not union_path.is_file():
        parts = [np.load(_shard_dir(work, "decontam_a", *k) / "interest.npy") for k in keys]
        union = np.unique(np.concatenate(parts)) if parts else np.empty(0, dtype=np.uint64)
        _save_npy(union_path, union.astype(np.uint64))
    say("S6: pass B (document frequency)")
    run(run_frequency, pending("decontam_b", interest=union_path))
    frequent_path = work / "decontam_frequent.npy"
    if not frequent_path.is_file():
        union = np.load(union_path)
        doc_freq = np.zeros(union.size, dtype=np.int64)
        for k in keys:
            doc_freq += np.load(_shard_dir(work, "decontam_b", *k) / "counts.npy")
        _save_npy(frequent_path, union[doc_freq > FREQ_LIMIT].astype(np.uint64))
    say("S6: pass C (title, 13-gram coverage, near duplicates)")
    run(run_judge, pending("decontam_c", frequent=frequent_path))
    result: dict[tuple[str, str], Path] = {}
    for key in keys:
        directory = _shard_dir(work, "decontam_c", *key)
        counts = _read_counts(directory / "counts.json")
        totals.merge(counts)
        by_source[key[0]].merge(counts)
        result[key] = directory / "docs.parquet"
    return result


def _dedup(
    table: pa.Table,
    priorities: Mapping[str, int],
    plan: _Plan,
    work: Path,
    totals: StageCounter,
    by_source: dict[str, StageCounter],
    say: Log,
) -> pa.Table:
    directory = work / "dedup"
    if (directory / DONE).is_file():
        say("S7: reusing finished dedup")
        counts = _read_counts(directory / "counts.json")
    else:
        say(f"S7: dedup over {table.num_rows:,} documents")
        batch = _one_batch(table)
        result, reasons = dedup_batch(batch, priorities, plan.config.tokenizer_hyphen)
        counts = {
            "total": {
                "in": table.num_rows,
                "out": result.batch.num_rows,
                "removed": dict(result.removed),
            },
            "by_source": {},
        }
        sources = table.column("source_id").to_pylist()
        for source_id in priorities:
            removed = dict.fromkeys(REASONS["dedup"], 0)
            n_in = 0
            for sid, reason in zip(sources, reasons, strict=True):
                if sid != source_id:
                    continue
                n_in += 1
                if reason is not None:
                    removed[reason] += 1
            counts["by_source"][source_id] = {
                "in": n_in,
                "out": n_in - sum(removed.values()),
                "removed": removed,
            }
        _replace_dir(directory)
        tmp = directory / "docs.parquet.tmp"
        write_parquet(pa.Table.from_batches([result.batch], schema=DOC_SCHEMA), tmp)
        tmp.replace(directory / "docs.parquet")
        _write_json(directory / "counts.json", counts)
        (directory / DONE).write_bytes(b"")
    total = counts["total"]
    totals.add("dedup", total["in"], total["out"], total["removed"])
    for source_id, entry in counts["by_source"].items():
        by_source[source_id].add("dedup", entry["in"], entry["out"], entry["removed"])
    return pq.read_table(directory / "docs.parquet", schema=DOC_SCHEMA)


def _compile(plan: _Plan, table: pa.Table, out_dir: Path, decontam: Mapping[str, Any]) -> int:
    config = plan.config
    writer = CorpusWriter(out_dir, config.tokenizer_hyphen)
    with writer:
        for batch in table.to_batches(max_chunksize=BATCH_ROWS):
            for row in batch.to_pylist():
                writer.add(row)
        for name in plan.views:
            writer.add_lemma_view(get_lemmatizer(name))
        writer.finish(
            {
                "build_id": plan.build_id,
                "name": config.name,
                "kind": config.kind,
                "trainable": config.kind in TRAINABLE_KINDS,
                "config": config.identity(),
                "config_hash": sha256_hex(canonical_json(config.identity())),
                "playability": config.playability.to_dict(),
                "lemmatizer": {"name": config.lemmatizer},
                "sources": plan.source_infos,
                "components": plan.components,
                "decontam": dict(decontam),
                "pipeline_version": PIPELINE_VERSION,
            }
        )
    return writer.n_docs


def _write_windex(plan: _Plan, out_dir: Path, root: Path) -> str:
    extra: list[str] = []
    for _entry, shard in plan.shards:
        if shard.path.is_dir() and (shard.path / "index.jsonl").is_file():
            extra.extend(snapshot_titles(shard.path))
    with CorpusStore.open(out_dir) as store:
        windex = build_windex(store, extra)
    windex.save(out_dir / WINDEX_DIR)
    return windex.content_hash()


def build_w_index(
    w_build_id: str,
    *,
    data_root: Path | None = None,
    sources: Mapping[str, SourceManifestEntry] | None = None,
) -> Path:
    """(Re)build ``corpus/<w_build_id>/windex/`` from a compiled W corpus and its snapshot."""
    root = configured_data_root() if data_root is None else Path(data_root)
    out_dir = root / "corpus" / w_build_id
    if not (out_dir / "manifest.json").is_file():
        raise BuildError(f"no compiled corpus at corpus/{w_build_id}")
    with CorpusStore.open(out_dir) as store:
        manifest = dict(store.manifest)
        if manifest.get("kind") != "evaluation":
            raise BuildError(f"corpus {w_build_id} is not an evaluation (W) build")
        table = load_sources() if sources is None else sources
        extra: list[str] = []
        for info in manifest.get("sources", []):
            entry = get_source(info["id"], table)
            for shard in open_source(entry).iter_shards(root / "raw"):
                if shard.path.is_dir():
                    extra.extend(snapshot_titles(shard.path))
        windex = build_windex(store, extra)
    target = out_dir / WINDEX_DIR
    if target.exists():
        shutil.rmtree(target)
    windex.save(target)
    return target


def with_overrides(config: CorpusConfig, **changes: Any) -> CorpusConfig:
    """``config`` with the given fields replaced (``None`` values are ignored)."""
    clean = {key: value for key, value in changes.items() if value is not None}
    return replace(config, **clean) if clean else config
