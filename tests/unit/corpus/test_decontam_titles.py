"""Title barrier (SDD 06 §7.1)."""

from __future__ import annotations

import unicodedata

import pytest

from artigo_env.corpus.decontam.titles import TitleBarrier, normalize_title

pytestmark = pytest.mark.fast


@pytest.mark.req("REQ-COR-007")
def test_first_letter_is_indifferent():
    assert normalize_title("al-Khwarizmi") == normalize_title("Al-Khwarizmi")
    assert "al-Khwarizmi" in TitleBarrier(["Al-Khwarizmi"])


@pytest.mark.req("REQ-COR-007")
def test_normalization_rules():
    decomposed = unicodedata.normalize("NFD", "São_Paulo")
    assert normalize_title(decomposed) == "são paulo"
    assert normalize_title("  Rio   de\tJaneiro \n") == "rio de janeiro"
    assert normalize_title("Guerra_dos_Cem_Anos") == "guerra dos cem anos"


@pytest.mark.req("REQ-COR-007")
def test_normalization_is_idempotent():
    for title in ("İstanbul", "São Paulo", "ΣΟΦΙΑ", "a_b  c"):
        key = normalize_title(title)
        assert normalize_title(key) == key


@pytest.mark.req("REQ-COR-007")
def test_barrier_membership_and_size(tmp_path):
    barrier = TitleBarrier(["Lisboa", "lisboa", "Porto", "", "  "])
    assert len(barrier) == 2
    assert "LISBOA" in barrier
    assert "Coimbra" not in barrier
    assert 3 not in barrier
    assert list(barrier) == ["lisboa", "porto"]
    path = tmp_path / "titles.txt"
    barrier.save(path)
    assert path.read_bytes() == b"lisboa\nporto\n"
    loaded = TitleBarrier.load(path)
    assert list(loaded) == list(barrier)
