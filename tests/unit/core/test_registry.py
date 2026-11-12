import pytest

from artigo_env.core.errors import DuplicateNameError, UnknownNameError
from artigo_env.core.registry import Registry

pytestmark = [pytest.mark.fast, pytest.mark.req("DEC-008", "DEC-012")]


def test_register_and_get():
    registry: Registry[int] = Registry("number")
    assert registry.register("one", 1) == 1
    assert registry.get("one") == 1
    assert "one" in registry
    assert len(registry) == 1


def test_decorator_returns_the_object():
    registry: Registry[type] = Registry("lemmatizer")

    @registry.register("identity")
    class Identity:
        pass

    assert registry.get("identity") is Identity


def test_duplicate_rejected():
    registry: Registry[int] = Registry("number")
    registry.register("one", 1)
    with pytest.raises(DuplicateNameError, match="one"):
        registry.register("one", 2)
    assert registry.get("one") == 1


def test_unknown_lists_kind_and_available_names():
    registry: Registry[int] = Registry("reward term")
    registry.register("b", 2)
    registry.register("a", 1)
    with pytest.raises(UnknownNameError) as excinfo:
        registry.get("zzz")
    message = str(excinfo.value)
    assert "reward term" in message and "'zzz'" in message and "a, b" in message
    assert isinstance(excinfo.value, KeyError)


@pytest.mark.parametrize("name", ["Bad Name", "", "UPPER", "-lead", "a b"])
def test_invalid_name_rejected(name):
    with pytest.raises(ValueError):
        Registry[int]("number").register(name, 1)


@pytest.mark.parametrize(
    "name", ["canonical", "first-letter@50", "custom:file.txt", "closed_class"]
)
def test_valid_names(name):
    Registry[int]("number").register(name, 1)


def test_names_sorted():
    registry: Registry[int] = Registry("number")
    for name in ("c", "a", "b"):
        registry.register(name, 0)
    assert registry.names() == ("a", "b", "c")
