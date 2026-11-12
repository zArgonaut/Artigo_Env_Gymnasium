"""Measure the agreement of a rule lemmatizer with a reference ``form<TAB>lemma`` list.

The reference comes from running the game's lemma function as a black box
(``run_reference.mjs``); this tool never reads the game's tables. Stopwords (``game227``) are
excluded from every measurement, because the game never lemmatizes them.

Usage::

    # evaluation forms: the N most frequent non-stopword letter-only forms
    uv run --env-file .env python -m tools.lemma_tables.compare forms --top 50000 --out FORMS

    # agreement report
    uv run --env-file .env python -m tools.lemma_tables.compare \
        --reference REF.tsv --forms FORMS [--freq FREQ.tsv] [--tables tables.json] \
        [--out REPORT.json] [--journal AGGREGATE.json]
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path

import regex

from artigo_env.core.paths import data_path
from artigo_env.text.lemmatizers.base import RuleLemmatizer
from artigo_env.text.lemmatizers.tables import LemmaTables, load_tables
from artigo_env.text.stopwords import StopwordSpec, load_stopwords

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TABLES = (
    REPO_ROOT / "src" / "artigo_env" / "assets" / "lemmatizers" / "open-tables" / "tables.json"
)
TOP_N = (1_000, 5_000, 20_000)
_LETTERS = regex.compile(r"\p{L}+")


def read_frequencies(path: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            form, _, count = line.rstrip("\n").partition("\t")
            if form and count:
                counts[form] = int(count)
    return counts


def read_reference(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            form, sep, lemma = line.rstrip("\n").partition("\t")
            if sep:
                out[form] = lemma
    return out


def evaluation_forms(freq: Mapping[str, int], top: int, stopwords: frozenset[str]) -> list[str]:
    """The ``top`` most frequent letter-only non-stopword forms (ties broken by form)."""
    ordered = sorted(freq.items(), key=lambda item: (-item[1], item[0]))
    out: list[str] = []
    for form, _ in ordered:
        if form in stopwords or not _LETTERS.fullmatch(form):
            continue
        out.append(form)
        if len(out) >= top:
            break
    return out


def stage_of(tables: LemmaTables, form: str) -> str:
    """Which stage of ``RuleLemmatizer`` decides the lemma of ``form``."""
    word = form.lower()
    if tables.irregular.apply(word) is not None:
        return "irregular"
    if tables.regular.apply(word) is not None:
        return "regular"
    plural = tables.plural.apply(word)
    gender = tables.gender.apply(word if plural is None else plural)
    if plural is not None and gender is not None:
        return "plural+gender"
    if plural is not None:
        return "plural"
    if gender is not None:
        return "gender"
    return "none"


def compare(
    tables: LemmaTables,
    reference: Mapping[str, str],
    forms: Sequence[str],
    freq: Mapping[str, int],
    stopwords: frozenset[str],
    top_disagreements: int = 200,
) -> dict[str, object]:
    lemmatizer = RuleLemmatizer("candidate", tables)
    kept = [f for f in forms if f in reference and f not in stopwords]
    if freq:
        kept.sort(key=lambda f: (-freq.get(f, 0), f))
    ours = {f: lemmatizer.lemma(f) for f in kept}
    agree = {f: ours[f] == reference[f] for f in kept}

    def share(items: Sequence[str]) -> float:
        return sum(agree[f] for f in items) / len(items) if items else 0.0

    report: dict[str, object] = {
        "forms": len(kept),
        "agreement_distinct": share(kept),
        "agreement_top": {str(n): share(kept[:n]) for n in TOP_N if n <= len(kept)},
    }
    if freq:
        total = sum(freq.get(f, 0) for f in kept)
        good = sum(freq.get(f, 0) for f in kept if agree[f])
        report["agreement_weighted"] = good / total if total else 0.0
    wrong = [f for f in kept if not agree[f]]
    categories: Counter[str] = Counter()
    weighted: Counter[str] = Counter()
    for f in wrong:
        changed = "same" if reference[f] == f else "changed"
        category = f"ours:{stage_of(tables, f)}/reference:{changed}"
        categories[category] += 1
        weighted[category] += freq.get(f, 0)
    report["disagreement_categories"] = {
        k: {"forms": categories[k], "tokens": weighted[k]} for k in sorted(categories)
    }
    report["disagreements_top5000"] = sum(1 for f in kept[:5_000] if not agree[f])
    report["top_disagreements"] = [
        {
            "form": f,
            "ours": ours[f],
            "reference": reference[f],
            "count": freq.get(f, 0),
            "stage": stage_of(tables, f),
        }
        for f in wrong[:top_disagreements]
    ]
    return report


def aggregate(report: Mapping[str, object]) -> dict[str, object]:
    """Numbers only (no word lists), for the versioned journal."""
    keys = (
        "forms",
        "agreement_distinct",
        "agreement_top",
        "agreement_weighted",
        "disagreements_top5000",
    )
    out = {k: report[k] for k in keys if k in report}
    categories = report.get("disagreement_categories", {})
    assert isinstance(categories, dict)
    out["disagreement_categories"] = categories
    return out


def _forms_main(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(description="write the evaluation form list")
    parser.add_argument("--freq", type=Path, default=None)
    parser.add_argument("--top", type=int, default=50_000)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    freq = read_frequencies(args.freq or data_path("measurements", "cc-news-form-freq.tsv"))
    forms = evaluation_forms(freq, args.top, load_stopwords(StopwordSpec()).words)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(forms) + "\n", encoding="utf-8", newline="\n")
    print(f"{len(forms)} forms -> {args.out}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["forms"]:
        return _forms_main(argv[1:])
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--forms", type=Path, required=True)
    parser.add_argument("--freq", type=Path, default=None)
    parser.add_argument("--tables", type=Path, default=DEFAULT_TABLES)
    parser.add_argument("--out", type=Path, default=None, help="full report (local only)")
    parser.add_argument("--journal", type=Path, default=None, help="aggregated numbers only")
    parser.add_argument("--top-disagreements", type=int, default=200)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    freq = read_frequencies(args.freq) if args.freq else {}
    forms = [
        line.strip() for line in args.forms.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    report = compare(
        load_tables(args.tables),
        read_reference(args.reference),
        forms,
        freq,
        load_stopwords(StopwordSpec()).words,
        args.top_disagreements,
    )
    report = {"date": dt.date.today().isoformat(), "tables": args.tables.name, **report}
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    if args.journal:
        args.journal.parent.mkdir(parents=True, exist_ok=True)
        payload = {"date": report["date"], **aggregate(report)}
        args.journal.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    summary = {k: report[k] for k in report if k not in ("top_disagreements",)}
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    if not args.quiet:
        for row in report["top_disagreements"][:60]:  # type: ignore[index]
            print(row)
    return 0


if __name__ == "__main__":
    sys.exit(main())
