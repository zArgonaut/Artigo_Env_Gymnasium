import dataclasses
import os
import subprocess
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import pytest

from artigo_env.core.config import ConfigBase, canonical_json, sha256_hex

pytestmark = [pytest.mark.fast, pytest.mark.req("REQ-REPO-002")]


class Mode(Enum):
    A = "a"
    B = "b"


@dataclass(frozen=True)
class Inner(ConfigBase):
    k: int = 1


@dataclass(frozen=True)
class Outer(ConfigBase):
    name: str
    items: tuple[str, ...] = ()
    inner: Inner = Inner()
    ratio: float | None = None


def test_roundtrip():
    original = Outer(name="x", items=("a", "b"), inner=Inner(k=7), ratio=0.5)
    rebuilt = Outer.from_dict(original.to_dict())
    assert rebuilt == original
    assert isinstance(rebuilt.items, tuple)
    assert isinstance(rebuilt.inner, Inner)


def test_hash_ignores_key_order():
    first = Outer.from_dict({"name": "x", "items": ["a"], "inner": {"k": 2}})
    second = Outer.from_dict({"inner": {"k": 2}, "items": ["a"], "name": "x"})
    assert first.content_hash() == second.content_hash()


def test_hash_changes_with_content():
    assert Outer(name="x").content_hash() != Outer(name="y").content_hash()
    assert Outer(name="x").content_hash() != Outer(name="x", inner=Inner(k=2)).content_hash()


def test_hash_is_64_hex():
    digest = Outer(name="x").content_hash()
    assert len(digest) == 64
    assert set(digest) <= set("0123456789abcdef")
    assert digest == sha256_hex(canonical_json(Outer(name="x").to_dict()))


def test_canonical_json_shape():
    assert canonical_json({"b": 1, "a": ("x", "y")}) == '{"a":["x","y"],"b":1}'
    assert canonical_json({"s": "ação"}) == '{"s":"ação"}'


def test_tuple_list_and_set_normalization():
    assert canonical_json(("a", "b")) == canonical_json(["a", "b"])
    assert canonical_json(frozenset({"b", "a"})) == '["a","b"]'
    assert canonical_json(Mode.A) == '"a"'


@pytest.mark.parametrize("value", [float("nan"), float("inf"), {"x": float("-inf")}])
def test_non_finite_rejected(value):
    with pytest.raises(ValueError):
        canonical_json(value)


@pytest.mark.parametrize("value", [Path("x"), {1: "a"}, b"bytes", object()])
def test_unsupported_type_rejected(value):
    with pytest.raises(TypeError):
        canonical_json(value)


def test_unknown_key_rejected():
    with pytest.raises(ValueError, match="unknown keys"):
        Outer.from_dict({"name": "a", "zzz": 1})


def test_missing_required_key_rejected():
    with pytest.raises(ValueError, match="missing key"):
        Outer.from_dict({"items": []})


def test_frozen():
    with pytest.raises(dataclasses.FrozenInstanceError):
        Outer(name="x").name = "y"  # type: ignore[misc]


def test_hash_stable_across_processes():
    code = (
        "from dataclasses import dataclass\n"
        "from artigo_env.core.config import ConfigBase\n"
        "@dataclass(frozen=True)\n"
        "class Inner(ConfigBase):\n"
        "    k: int = 1\n"
        "@dataclass(frozen=True)\n"
        "class Outer(ConfigBase):\n"
        "    name: str\n"
        "    items: tuple[str, ...] = ()\n"
        "    inner: Inner = Inner()\n"
        "    ratio: float | None = None\n"
        "print(Outer(name='x', items=('a', 'b'), inner=Inner(k=7)).content_hash())\n"
    )
    expected = Outer(name="x", items=("a", "b"), inner=Inner(k=7)).content_hash()
    for seed in ("1", "2"):
        env = {**os.environ, "PYTHONHASHSEED": seed}
        result = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, env=env, check=True
        )
        assert result.stdout.strip() == expected
