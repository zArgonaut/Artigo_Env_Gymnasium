import pytest

from artigo_env.core.errors import DataRootNotConfiguredError
from artigo_env.core.paths import ENV_VAR, data_path, data_root

pytestmark = [pytest.mark.fast, pytest.mark.req("REQ-PLT-003")]


def test_data_root_from_env(monkeypatch, tmp_path):
    monkeypatch.setenv(ENV_VAR, str(tmp_path))
    assert data_root() == tmp_path
    assert data_path("runs", "x") == tmp_path / "runs" / "x"


@pytest.mark.parametrize("value", [None, "", "   ", "relative/dir"])
def test_unset_empty_or_relative_raises(monkeypatch, value):
    if value is None:
        monkeypatch.delenv(ENV_VAR, raising=False)
    else:
        monkeypatch.setenv(ENV_VAR, value)
    with pytest.raises(DataRootNotConfiguredError, match=ENV_VAR):
        data_root()


def test_env_is_read_on_every_call(monkeypatch, tmp_path):
    first, second = tmp_path / "a", tmp_path / "b"
    monkeypatch.setenv(ENV_VAR, str(first))
    assert data_root() == first
    monkeypatch.setenv(ENV_VAR, str(second))
    assert data_root() == second


def test_root_does_not_need_to_exist(monkeypatch, tmp_path):
    monkeypatch.setenv(ENV_VAR, str(tmp_path / "missing"))
    assert not data_root().exists()


@pytest.mark.parametrize("part", ["..", "a/../b", "a\\..\\b", "/abs", "C:\\abs", "C:rel"])
def test_data_path_rejects_escape(monkeypatch, tmp_path, part):
    monkeypatch.setenv(ENV_VAR, str(tmp_path))
    with pytest.raises(ValueError):
        data_path("runs", part)
