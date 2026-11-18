"""Outcome reading under the failure modes of the real recordings (REQ-CAL-001)."""

from __future__ import annotations

from pathlib import Path

import pytest
from synthetic import TEXT, TITLE, Guess, build_session, render_game

from tools.calibration.dom import read_guess_list
from tools.calibration.fixtures import build_fixture
from tools.calibration.outcomes import clean_label
from tools.calibration.session import load_session

GUESSES = [
    Guess("rio", True, (0, 3, 22)),
    Guess("de", False, (), "não conta"),
    Guess("rio", False, (), "já foi"),
    Guess("vale", True, (7, 19)),
    Guess("zzz", True, ()),
]


def _expected(path: Path) -> list[dict[str, object]]:
    return [g["expected"] for g in build_fixture(load_session(path), "cal-v1")["guesses"]]


@pytest.mark.fast
@pytest.mark.req("REQ-CAL-001")
def test_scrolled_list_has_no_newest_row_and_keeps_positions() -> None:
    rows = [("c", 2), ("b", 0), ("a", 1)]
    page = render_game(TITLE, TEXT, set(), rows, positioned=True, scroll=2)
    scrolled = read_guess_list(page)
    assert scrolled.total is None
    assert scrolled.newest is None  # the first rendered row is an older guess
    assert [(e.label, e.index) for e in scrolled.entries] == [("b", 2), ("a", 3)]
    assert scrolled.container_rows == 4  # header + 3 rows


@pytest.mark.fast
@pytest.mark.req("REQ-CAL-001")
def test_warning_row_takes_position_one_and_is_flagged() -> None:
    page = render_game(TITLE, TEXT, set(), [("a", 1)], ("de", "não conta"), positioned=True)
    guess_list = read_guess_list(page)
    assert guess_list.total == 1
    assert guess_list.has_warning
    assert [(e.label, e.status, e.index) for e in guess_list.entries] == [
        ("de", "warning", 1),
        ("a", "found", 2),
    ]
    assert guess_list.container_rows == 3  # header + warning + one row


@pytest.mark.fast
@pytest.mark.req("REQ-CAL-001")
def test_rows_without_a_position_keep_their_dom_order() -> None:
    guess_list = read_guess_list(render_game(TITLE, TEXT, set(), [("b", 0), ("a", 1)]))
    assert [(e.label, e.index) for e in guess_list.entries] == [("b", 1), ("a", 2)]
    assert guess_list.newest is not None
    assert guess_list.newest.label == "b"


@pytest.mark.fast
@pytest.mark.req("REQ-CAL-001")
def test_clean_label_follows_what_the_game_shows() -> None:
    assert clean_label("Sendo]") == "sendo"
    assert clean_label("  Água ") == "água"
    assert clean_label("3") == "3"
    assert clean_label("guarda-chuva") == "guarda-chuva"


@pytest.mark.fast
@pytest.mark.req("REQ-CAL-001")
def test_late_snapshots_still_give_every_outcome(tmp_path: Path) -> None:
    """No snapshot per guess (the recorder lagged): the one final list and the mutation stream
    are enough; a repeated label owns a single row, the first guess's."""
    reference = _expected(
        build_session(tmp_path / "a", GUESSES, positioned=True, token_events=True)
    )
    late = _expected(
        build_session(tmp_path / "b", GUESSES, positioned=True, token_events=True, snapshots="last")
    )
    assert not any(item.get("unreliable") for item in late)
    assert [(g["counted"], g["count"], g["status"]) for g in late] == [
        (True, 3, "found"),
        (False, 0, "warning"),
        (False, 0, "warning"),
        (True, 2, "found"),
        (True, 0, "not-found"),
    ]
    assert late[0]["revealed_token_indices"] == [0, 3, 22]
    assert late[3]["revealed_token_indices"] == [7, 19]
    assert late[3]["matched_token_indices"] == [7, 19]  # the previous guess lost its match class
    # Only the warning messages are lost (the transient rows were never captured).
    for old, new in zip(reference, late, strict=True):
        assert {k: v for k, v in old.items() if k != "message"} == {
            k: v for k, v in new.items() if k != "message"
        }
    assert [g["message"] for g in late] == [None, None, None, None, None]
    assert [g["message"] for g in reference][1:3] == ["não conta", "já foi"]


@pytest.mark.fast
@pytest.mark.req("REQ-CAL-001")
def test_scrolled_snapshots_are_not_misread(tmp_path: Path) -> None:
    plain = _expected(build_session(tmp_path / "a", GUESSES, positioned=True, token_events=True))
    scrolled = _expected(
        build_session(
            tmp_path / "b",
            GUESSES,
            positioned=True,
            token_events=True,
            scroll_at={1: 2, 2: 2, 3: 1},
        )
    )
    assert not any(item.get("unreliable") for item in scrolled)
    # A guess whose only snapshot lacks its row cannot be a warning: the header total (or the
    # container height) proves the counted ones, and the transient messages are simply unseen.
    for old, new in zip(plain, scrolled, strict=True):
        assert (old["counted"], old["count"], old["status"]) == (
            new["counted"],
            new["count"],
            new["status"],
        )


@pytest.mark.fast
@pytest.mark.req("REQ-CAL-001")
def test_guess_without_any_observed_row_is_unreliable(tmp_path: Path) -> None:
    """The last guess is counted but only a scrolled list (no newest row) was recorded."""
    path = build_session(tmp_path, GUESSES, positioned=True, snapshots="last", scroll_at={4: 2})
    last = _expected(path)[-1]
    assert last["unreliable"] is True
    assert last["reason"] == "its row was never observed"


@pytest.mark.fast
@pytest.mark.req("REQ-CAL-001")
def test_prior_history_is_recorded_and_explains_repeats(tmp_path: Path) -> None:
    """The game holds guesses of an earlier game: the header counts them and a guess equal to
    one of them is a repeat ("já foi") although the player never made it in this session."""
    guesses = [Guess("ano", False, (), "já foi"), Guess("rio", True, (0, 3, 22))]
    path = build_session(
        tmp_path, guesses, positioned=True, mode="random", prior=(("fazer", 11), ("ano", 4))
    )
    fixture = build_fixture(load_session(path), "cal-v1")
    assert fixture["prior_guesses"] == ["ano", "fazer"]  # oldest first, as the game stores them
    assert fixture["prior_total"] == 2
    first, second = (g["expected"] for g in fixture["guesses"])
    assert (first["counted"], first["message"], first["prior_history"]) == (False, "já foi", True)
    assert (second["counted"], second["count"], second["status"]) == (True, 3, "found")
    assert "prior_history" not in second
    assert not first.get("unreliable") and not second.get("unreliable")
