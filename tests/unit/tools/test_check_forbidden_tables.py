from pathlib import Path

import pytest

from tools.repo import check_forbidden_tables as mod
from tools.repo.check_forbidden_tables import check, main

pytestmark = [pytest.mark.fast, pytest.mark.req("REQ-TXT-012")]

SYNTHETIC = "zzalpha|zzbeta|zzgamma|zzdelta|zzepsilon|zzzeta|zzeta2|zztheta"


def _write(tmp_path: Path, name: str, text: str) -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def test_generic_rule_flags_long_alternation_with_marker(tmp_path):
    alternatives = "|".join(f"w{i}" for i in range(45))
    path = _write(tmp_path, "a.txt", f'x = ["{alternatives}|connosco"]\n')
    result = check([path], [])
    assert len(result) == 1 and str(path) in result[0]


def test_generic_rule_needs_both_conditions(tmp_path):
    many = "|".join(f"w{i}" for i in range(60))
    few = "|".join(f"w{i}" for i in range(10))
    assert check([_write(tmp_path, "no_marker.txt", many + "\n")], []) == []
    assert check([_write(tmp_path, "few.txt", few + "|connosco\n")], []) == []


def test_exactly_forty_alternatives_is_allowed(tmp_path):
    line = "|".join(["connosco"] + [f"w{i}" for i in range(39)])
    assert line.count("|") + 1 == 40
    assert check([_write(tmp_path, "edge.txt", line)], []) == []
    assert check([_write(tmp_path, "over.txt", line + "|extra")], []) != []


def test_stopword_list_one_word_per_line_is_fine(tmp_path):
    words = "\n".join(["o", "a", "connosco", "conosco"] * 20)
    assert check([_write(tmp_path, "stop.txt", words)], []) == []


def test_signature_substring_is_flagged(tmp_path):
    bad = _write(tmp_path, "bad.py", f"data = '{SYNTHETIC}'\n")
    good = _write(tmp_path, "good.py", "data = 'nothing here'\n")
    result = check([bad, good], [SYNTHETIC])
    assert len(result) == 1 and str(bad) in result[0]


def test_blank_signatures_never_match(tmp_path):
    assert check([_write(tmp_path, "a.txt", "anything\n")], ["", "   "]) == []


def test_binary_and_undecodable_files_are_skipped(tmp_path):
    binary = tmp_path / "b.bin"
    binary.write_bytes(b"\x00\x01" + SYNTHETIC.encode() + b"\x00")
    latin = tmp_path / "l.txt"
    latin.write_bytes(b"\xff\xfe" + SYNTHETIC.encode())
    assert check([binary, latin, tmp_path / "missing.txt"], [SYNTHETIC]) == []


def test_cli_uses_signatures_from_data_root(tmp_path, monkeypatch, capsys):
    lemma_dir = tmp_path / "root" / "reference" / "lemmatizer"
    lemma_dir.mkdir(parents=True)
    (lemma_dir / "signatures.txt").write_text(SYNTHETIC + "\n", encoding="utf-8")
    monkeypatch.setenv("ARTIGO_DATA_ROOT", str(tmp_path / "root"))
    bad = _write(tmp_path, "bad.txt", SYNTHETIC)
    good = _write(tmp_path, "good.txt", "fine\n")
    assert main([str(good)]) == 0
    assert main([str(bad)]) == 1
    assert "bad.txt" in capsys.readouterr().out


def test_cli_without_data_root_still_runs_generic_rule(tmp_path, monkeypatch):
    monkeypatch.delenv("ARTIGO_DATA_ROOT", raising=False)
    monkeypatch.setattr(mod, "REPO_ROOT", tmp_path)
    assert mod.load_signatures() == []
    assert main([str(_write(tmp_path, "ok.txt", "fine\n"))]) == 0


def test_load_signatures_reads_dotenv(tmp_path, monkeypatch):
    root = tmp_path / "data"
    (root / "reference" / "lemmatizer").mkdir(parents=True)
    (root / "reference" / "lemmatizer" / "signatures.txt").write_text(
        "s1\n\ns2\n", encoding="utf-8"
    )
    (tmp_path / ".env").write_text(f"ARTIGO_DATA_ROOT={root}\n", encoding="utf-8")
    monkeypatch.delenv("ARTIGO_DATA_ROOT", raising=False)
    monkeypatch.setattr(mod, "REPO_ROOT", tmp_path)
    assert mod.load_signatures() == ["s1", "s2"]


def test_tooling_files_have_no_very_long_lines(repo_root):
    files = [
        *sorted((repo_root / "tools" / "lemma_tables").glob("*.mjs")),
        repo_root / "tools" / "repo" / "check_forbidden_tables.py",
    ]
    assert len(files) >= 4
    for file in files:
        for number, line in enumerate(file.read_text(encoding="utf-8").splitlines(), start=1):
            assert len(line) <= 400, f"{file.name}:{number} is {len(line)} chars (pasted table?)"
