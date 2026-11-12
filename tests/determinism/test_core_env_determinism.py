"""Reproducibility of ArtigoCoreEnv (SDD 09 §5, REQ-ENV-002, DEC-062)."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys

import numpy as np
import pytest

from artigo_env.envs import ArtigoCoreEnv

SCRIPT = """
import hashlib, json, sys
import numpy as np
from artigo_env.envs import ArtigoCoreEnv
env = ArtigoCoreEnv(ruleset=sys.argv[1], strict_actions=False)
digest = hashlib.sha256()
for seed in (0, 7):
    obs, info = env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    for _ in range(40):
        obs, reward, term, trunc, info = env.step(int(rng.integers(env.action_space.n)))
        for key in sorted(obs):
            digest.update(obs[key].tobytes())
        digest.update(json.dumps([reward, term, trunc, info], sort_keys=True, default=str).encode())
        if term or trunc:
            break
print(digest.hexdigest())
"""


def trajectory_digest(ruleset: str) -> str:
    env = ArtigoCoreEnv(ruleset=ruleset, strict_actions=False)
    digest = hashlib.sha256()
    for seed in (0, 7):
        obs, info = env.reset(seed=seed)
        rng = np.random.default_rng(seed)
        for _ in range(40):
            obs, reward, term, trunc, info = env.step(int(rng.integers(env.action_space.n)))
            for key in sorted(obs):
                digest.update(obs[key].tobytes())
            payload = json.dumps([reward, term, trunc, info], sort_keys=True, default=str)
            digest.update(payload.encode())
            if term or trunc:
                break
    env.close()
    return digest.hexdigest()


@pytest.mark.fast
@pytest.mark.req("REQ-ENV-002")
@pytest.mark.parametrize("ruleset", ["canonical", "hard"])
def test_same_seed_same_trajectory(ruleset):
    assert trajectory_digest(ruleset) == trajectory_digest(ruleset)


@pytest.mark.slow
@pytest.mark.req("REQ-ENV-002")
def test_same_trajectory_in_another_process(repo_root):
    result = subprocess.run(
        [sys.executable, "-c", SCRIPT, "canonical"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == trajectory_digest("canonical")


@pytest.mark.fast
@pytest.mark.req("REQ-ENV-002", "REQ-SCN-002")
def test_reseeding_restarts_and_unseeded_reset_continues():
    env = ArtigoCoreEnv()

    def refs(first_seed, n):
        out = [env.reset(seed=first_seed) and env.unwrapped._article.article_ref]
        for _ in range(n - 1):
            env.reset()
            out.append(env._article.article_ref)
        return out

    a = refs(5, 12)
    b = refs(5, 12)
    assert a == b
    assert len(set(a[:12])) == 12  # demo has 30 documents: one epoch has no repeats
    env.close()


@pytest.mark.fast
@pytest.mark.req("REQ-SCN-003", "REQ-ENV-002")
def test_workers_draw_disjoint_documents():
    seen = []
    for worker in range(3):
        env = ArtigoCoreEnv(worker_index=worker, num_workers=3)
        env.reset(seed=100 + worker)  # vectorized convention: seed + worker_index
        docs = [env._doc_idx]
        for _ in range(9):
            env.reset()
            docs.append(env._doc_idx)
        seen.append(set(docs))
        env.close()
    assert sum(len(s) for s in seen) == 30
    assert set.union(*seen) == set(range(30))
    with pytest.raises(ValueError, match="seed - worker_index"):
        ArtigoCoreEnv(worker_index=2, num_workers=3).reset(seed=1)
