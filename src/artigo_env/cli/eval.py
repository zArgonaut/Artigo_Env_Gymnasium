"""``artigo-env eval``: evaluate a reference agent on a fixed evaluation set (SDD 14 §3)."""

from __future__ import annotations

import argparse
import importlib
from collections.abc import Callable
from pathlib import Path
from typing import Any, TextIO

import yaml

from artigo_env.core.errors import ArtigoEnvError
from artigo_env.core.vocab import Vocabulary
from artigo_env.engine.rulesets import PRESETS


def add_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser(
        "eval",
        help="evaluate an agent on a fixed evaluation set",
        description="Play one episode per article of the scenario's fixed list, in order, for "
        "each seed; write episodes.parquet, steps.parquet and summary.json to --out or "
        "ARTIGO_DATA_ROOT/runs/<run_id>/eval/<scenario>/ and print the summary.",
    )
    parser.add_argument("--agent", required=True, help="agent name in the agent registry")
    parser.add_argument("--corpus", required=True, help="'demo', a build id or a build path")
    parser.add_argument("--ruleset", default="canonical", choices=PRESETS.names())
    parser.add_argument("--scenario", required=True, help="evaluation scenario, e.g. eval-w-val")
    parser.add_argument("--vocab", default=None, help="vocabulary id or path (default: matched)")
    parser.add_argument(
        "--knowledge",
        default=None,
        help="knowledge store id or path (default: the one built for V)",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[0], help="evaluation seeds")
    parser.add_argument("--limit", type=int, default=None, help="evaluate the first N articles")
    parser.add_argument("--out", type=Path, default=None, help="output directory")
    parser.add_argument(
        "--final", action="store_true", help="mark the run final (required for eval-w-test)"
    )
    parser.add_argument(
        "--param",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="agent hyperparameter (VALUE parsed as YAML); repeatable",
    )
    parser.set_defaults(handler=run_eval)


def _params(items: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for item in items:
        key, sep, value = item.partition("=")
        if not sep or not key:
            raise ValueError(f"--param expects KEY=VALUE, got {item!r}")
        out[key.strip()] = yaml.safe_load(value)
    return out


def agent_factory_function() -> Callable[..., Any]:
    """``artigo_env.agents.make_agent``, imported at run time."""
    try:
        module = importlib.import_module("artigo_env.agents")
    except ImportError as exc:  # pragma: no cover - the package always ships the module
        raise ArtigoEnvError(f"the agent registry cannot be imported: {exc}") from exc
    make_agent = getattr(module, "make_agent", None)
    if not callable(make_agent):
        raise ArtigoEnvError(
            "the agent registry (artigo_env.agents.make_agent) is not available in this "
            "version of the package"
        )
    return make_agent  # type: ignore[no-any-return]


def open_knowledge(knowledge: str | None, vocabulary: Vocabulary) -> Any:
    """Knowledge store named by ``--knowledge`` (path or id), else the one matching V."""
    from artigo_env.core.paths import data_path
    from artigo_env.services.knowledge import MANIFEST_FILE, KnowledgeStore, find_knowledge

    if knowledge is None:
        path = find_knowledge(vocabulary.vocab_id)
    else:
        path = Path(knowledge)
        if not (path / MANIFEST_FILE).is_file():
            path = data_path("knowledge", knowledge)
    return KnowledgeStore.open(path, vocab=vocabulary)


def run_eval(args: argparse.Namespace, stdin: TextIO, stdout: TextIO) -> int:
    from artigo_env.envs.core_env import (
        ArtigoCoreEnv,
        resolve_corpus,
        resolve_ruleset,
        resolve_scenario,
        resolve_vocab,
    )
    from artigo_env.evaluation import FinalRunRequiredError, evaluate, summary_table
    from artigo_env.evaluation.harness import FINAL_ONLY_SPLITS

    try:
        scenario = resolve_scenario(args.scenario)
        if scenario.split in FINAL_ONLY_SPLITS and not args.final:
            stdout.write(
                f"error: scenario {scenario.name!r} uses the held-out split {scenario.split!r}; "
                "pass --final to mark this run final (REQ-EVL-003)\n"
            )
            return 2
        if not scenario.is_evaluation and scenario.name != "demo":
            stdout.write(
                f"error: scenario {scenario.name!r} is not an evaluation scenario "
                "(use eval-w-val, eval-ext-val, eval-w-test or demo)\n"
            )
            return 2
        params = _params(args.param)
        make_agent = agent_factory_function()
        ruleset = resolve_ruleset(args.ruleset)
        from artigo_env.corpus.store import CorpusStore

        with CorpusStore.open(resolve_corpus(args.corpus)) as store:
            build_id = store.build_id
        vocab_path = resolve_vocab(args.vocab, args.corpus, build_id, ruleset)
        vocabulary = Vocabulary.open(vocab_path)
        try:
            knowledge = open_knowledge(args.knowledge, vocabulary)

            def build_agent(seed: int) -> Any:
                return make_agent(
                    args.agent,
                    vocab=vocabulary,
                    knowledge=knowledge,
                    ruleset=ruleset,
                    seed=seed,
                    **params,
                )

            def build_env() -> ArtigoCoreEnv:
                return ArtigoCoreEnv(
                    corpus=args.corpus, ruleset=ruleset, scenario=scenario, vocab=vocab_path
                )

            result = evaluate(
                build_agent,
                build_env,
                scenario.name,
                args.seeds,
                limit=args.limit,
                out_dir=args.out,
                final=args.final,
            )
        finally:
            vocabulary.close()
    except FinalRunRequiredError as exc:
        stdout.write(f"error: {exc}\n")
        return 2
    except (ArtigoEnvError, ValueError, TypeError, KeyError) as exc:
        stdout.write(f"error: {exc}\n")
        return 1
    stdout.write(summary_table(result.summary) + "\n")
    stdout.write(f"written to {result.out_dir}\n")
    return 0
