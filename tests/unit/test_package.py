import importlib

import pytest

pytestmark = [pytest.mark.fast, pytest.mark.req("REQ-PLT-001")]

LAYERS = [
    "core",
    "text",
    "engine",
    "corpus",
    "scenarios",
    "synthetic",
    "services",
    "envs",
    "wrappers",
    "agents",
    "evaluation",
    "analysis",
    "cli",
]


def test_version():
    import artigo_env

    assert artigo_env.__version__ == "0.1.0"


@pytest.mark.parametrize("layer", LAYERS)
def test_layer_packages_exist(layer):
    importlib.import_module(f"artigo_env.{layer}")
