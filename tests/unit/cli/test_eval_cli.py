"""``artigo-env eval`` with a stand-in agent registry."""

from __future__ import annotations

import io
import json
import sys
import types

import numpy as np
import pytest

from artigo_env.cli import eval as eval_cli
from artigo_env.cli.main import main
from artigo_env.core.vocab import Vocabulary

pytestmark = pytest.mark.fast


class FirstAllowed:
    def __init__(self, seed: int) -> None:
        self.seed = seed

    def reset(self, obs, info):
        pass

    def act(self, obs, info, action_mask):
        return int(np.argmax(action_mask))


@pytest.fixture
def registry(monkeypatch):
    calls: list[dict] = []

    def make_agent(name, *, vocab, knowledge, ruleset, seed, **hyperparams):
        calls.append(
            {"name": name, "vocab": vocab, "knowledge": knowledge, "ruleset": ruleset, "seed": seed}
            | hyperparams
        )
        return FirstAllowed(seed)

    monkeypatch.setattr(eval_cli, "agent_factory_function", lambda: make_agent)
    monkeypatch.setattr(eval_cli, "open_knowledge", lambda knowledge, vocab: "KNOWLEDGE")
    return calls


def run(*argv: str) -> tuple[int, str]:
    out = io.StringIO()
    code = main(list(argv), stdin=io.StringIO(), stdout=out)
    return code, out.getvalue()


@pytest.mark.req("REQ-EVL-001", "REQ-EVL-005")
def test_eval_on_the_demo_corpus(tmp_path, registry):
    code, text = run(
        "eval",
        "--agent",
        "first",
        "--corpus",
        "demo",
        "--scenario",
        "demo",
        "--seeds",
        "0",
        "1",
        "--limit",
        "2",
        "--out",
        str(tmp_path),
        "--param",
        "temperature=0.5",
    )
    assert code == 0, text
    assert "N mean (main)" in text and "win rate" in text
    assert [c["seed"] for c in registry] == [0, 1]
    assert all(isinstance(c["vocab"], Vocabulary) for c in registry)
    assert registry[0]["knowledge"] == "KNOWLEDGE" and registry[0]["temperature"] == 0.5
    assert registry[0]["ruleset"].name == "canonical"
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["n_episodes"] == 4 and summary["seeds"] == [0, 1]
    assert (tmp_path / "episodes.parquet").is_file() and (tmp_path / "steps.parquet").is_file()


@pytest.mark.req("REQ-EVL-003")
def test_w_test_is_refused_without_final(tmp_path, registry):
    code, text = run("eval", "--agent", "a", "--corpus", "demo", "--scenario", "eval-w-test")
    assert code == 2
    assert "--final" in text and "REQ-EVL-003" in text
    assert registry == []


@pytest.mark.req("REQ-EVL-001")
def test_training_scenarios_are_refused(registry):
    code, text = run("eval", "--agent", "a", "--corpus", "demo", "--scenario", "train-default")
    assert code == 2 and "not an evaluation scenario" in text


@pytest.mark.req("REQ-EVL-001")
def test_missing_agent_registry_is_reported(monkeypatch):
    monkeypatch.setitem(sys.modules, "artigo_env.agents", types.ModuleType("artigo_env.agents"))
    code, text = run("eval", "--agent", "a2", "--corpus", "demo", "--scenario", "demo")
    assert code == 1
    assert "agent registry" in text


@pytest.mark.req("REQ-EVL-001")
def test_malformed_param_is_reported(registry):
    code, text = run(
        "eval", "--agent", "a", "--corpus", "demo", "--scenario", "demo", "--param", "oops"
    )
    assert code == 1 and "KEY=VALUE" in text
