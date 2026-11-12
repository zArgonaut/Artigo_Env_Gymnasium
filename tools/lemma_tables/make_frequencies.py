"""Build a form-frequency list from the local CC-News Portuguese sample.

Each article (title and text) is tokenized with ``artigo_env.text.tokenizer.tokenize`` and
lower-cased with the package's ``lower``; the output has one ``form<TAB>count`` line per form,
sorted by decreasing count and then by form.

Usage (needs ``pyarrow``, which is not a project dependency)::

    uv run --env-file .env --with pyarrow python -m tools.lemma_tables.make_frequencies \
        [--input DIR] [--out FILE] [--max-docs N] [--workers N]
"""

from __future__ import annotations

import argparse
import importlib
import sys
import time
from collections import Counter
from collections.abc import Iterator, Sequence
from multiprocessing import Pool
from pathlib import Path

from artigo_env.core.paths import data_path
from artigo_env.text.normalize import lower
from artigo_env.text.tokenizer import tokenize

BATCH_ROWS = 2_000
COLUMNS = ("title", "text")


def count_batch(texts: list[str]) -> tuple[int, Counter[str]]:
    counts: Counter[str] = Counter()
    for text in texts:
        counts.update(lower(token.text) for token in tokenize(text))
    return len(texts), counts


def iter_batches(files: Sequence[Path], max_docs: int) -> Iterator[list[str]]:
    parquet = importlib.import_module("pyarrow.parquet")
    remaining = max_docs
    for path in files:
        reader = parquet.ParquetFile(str(path))
        for batch in reader.iter_batches(batch_size=BATCH_ROWS, columns=list(COLUMNS)):
            rows = batch.to_pydict()
            texts = [
                "\n".join(part for part in (title, text) if part)
                for title, text in zip(rows["title"], rows["text"], strict=True)
            ]
            if remaining < len(texts):
                texts = texts[:remaining]
            remaining -= len(texts)
            yield texts
            if remaining <= 0:
                return


def write_frequencies(counts: Counter[str], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    with out.open("w", encoding="utf-8", newline="\n") as handle:
        for form, count in ordered:
            handle.write(f"{form}\t{count}\n")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--max-docs", type=int, default=300_000)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args(argv)
    source = args.input or data_path("raw", "cc_news_pt", "cc_news_pt", "train")
    out = args.out or data_path("measurements", "cc-news-form-freq.tsv")
    files = sorted(source.glob("*.parquet"))
    if not files:
        print(f"no parquet files in {source}", file=sys.stderr)
        return 1
    started = time.perf_counter()
    total: Counter[str] = Counter()
    documents = 0
    with Pool(args.workers) as pool:
        for size, counts in pool.imap(count_batch, iter_batches(files, args.max_docs)):
            total.update(counts)
            documents += size
    write_frequencies(total, out)
    elapsed = time.perf_counter() - started
    print(
        f"{documents} documents, {sum(total.values())} tokens, "
        f"{len(total)} forms in {elapsed:.0f}s -> {out}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
