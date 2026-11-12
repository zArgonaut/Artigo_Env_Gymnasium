from __future__ import annotations

import json

import pytest

from tools.w.fetch_extracts import api_url, file_name, load_index, read_titles, snapshot, summarize

pytestmark = pytest.mark.fast


def answer(title: str, extract: str = "Texto.\n\n== Seção ==\nMais texto.", **extra):
    page = {"pageid": 7, "title": title, "lastrevid": 99, "touched": "2026-11-01T00:00:00Z"}
    page["extract"] = extract
    page.update(extra)
    return {"batchcomplete": True, "query": {"pages": [page]}}


@pytest.mark.req("DEC-050")
def test_read_titles_skips_comments_header_and_duplicates(tmp_path):
    path = tmp_path / "list.tsv"
    path.write_text(
        "# comment\n# another\ntitulo\tem_completa\n2009\t1\nA Arte da Guerra\t1\n2009\t0\n\n",
        encoding="utf-8",
    )
    assert read_titles(path) == ["2009", "A Arte da Guerra"]


@pytest.mark.req("DEC-050")
def test_request_asks_for_plain_text_with_wiki_headers_and_redirects():
    url = api_url("São Paulo")
    assert url.startswith("https://pt.wikipedia.org/w/api.php?")
    for part in ("explaintext=1", "exsectionformat=wiki", "redirects=1", "titles=S%C3%A3o+Paulo"):
        assert part in url


@pytest.mark.req("DEC-050")
def test_summarize_statuses():
    assert summarize("A", answer("A"))["status"] == "ok"
    assert summarize("A", answer("A", extract=""))["status"] == "empty"
    missing = {"query": {"pages": [{"title": "A", "missing": True}]}}
    assert summarize("A", missing)["status"] == "missing"
    redirected = answer("Novo título")
    redirected["query"]["redirects"] = [{"from": "Velho", "to": "Novo título"}]
    row = summarize("Velho", redirected)
    assert (row["resolved"], row["redirected"], row["lastrevid"]) == ("Novo título", True, 99)


@pytest.mark.req("DEC-050")
def test_snapshot_is_resumable_and_keeps_raw_answers(tmp_path):
    calls: list[str] = []
    pauses: list[float] = []

    def fetch(title):
        calls.append(title)
        return answer(title)

    def run(titles):
        return list(
            snapshot(
                titles, tmp_path, fetch=fetch, interval=2.0, sleep=pauses.append, now=lambda: "t"
            )
        )

    first = run(["A", "B"])
    assert [row["requested"] for row in first] == ["A", "B"]
    assert pauses == [2.0]  # no pause before the first request
    second = run(["A", "B", "C"])
    assert [row["requested"] for row in second] == ["C"]
    assert calls == ["A", "B", "C"]

    index = load_index(tmp_path / "index.jsonl")
    assert set(index) == {"A", "B", "C"}
    stored = json.loads((tmp_path / "pages" / file_name("B")).read_text(encoding="utf-8"))
    assert stored["requested"] == "B"
    assert stored["response"]["query"]["pages"][0]["extract"].startswith("Texto.")
    assert index["B"]["n_chars"] == len(stored["response"]["query"]["pages"][0]["extract"])
