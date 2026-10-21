"""Tests for grid override files: their shape, reading them, and applying one to a race.

The race here is five cars in qualifying order. An override must move the named cars and
let everyone else close the gaps in their old order, put pit-lane starters behind the
grid, and refuse anything it can't apply exactly: a typo, a slot that doesn't exist, a
driver who isn't entered, or a file written for another race.
"""

from __future__ import annotations

import codecs
import json

import pandas as pd
import pytest

from f1predict.override import (
    GridOverride,
    GridOverrideError,
    apply_grid_override,
    load_grid_override,
    parse_grid_override,
)

DRIVERS = ["norris", "piastri", "hamilton", "leclerc", "russell"]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def race_rows(
    grid: tuple[int, ...] = (1, 2, 3, 4, 5), pitlane: tuple[int, ...] = (0, 0, 0, 0, 0)
) -> pd.DataFrame:
    """One race's rows as the predictor sees them: ids, codes, grid, pit-lane flag, a gap."""
    return pd.DataFrame(
        {
            "season": 2026,
            "round": 18,
            "driver_id": pd.array(DRIVERS, dtype="string"),
            "driver_code": pd.array([driver[:3].upper() for driver in DRIVERS], dtype="string"),
            "grid": [float(slot) for slot in grid],
            "pitlane_start": [float(flag) for flag in pitlane],
            "quali_gap_pole_pct": [0.0, 0.1, 0.2, 0.3, 0.4],
        }
    )


def starts(race: pd.DataFrame) -> dict[str, tuple[float, float]]:
    """Each driver's (grid, pitlane_start)."""
    return {
        str(driver): (grid, pitlane)
        for driver, grid, pitlane in zip(
            race["driver_id"], race["grid"], race["pitlane_start"], strict=True
        )
    }


def apply(data: dict, race: pd.DataFrame | None = None) -> pd.DataFrame:
    """Check ``data`` as an override and apply it to ``race`` (the default five cars)."""
    return apply_grid_override(race_rows() if race is None else race, parse_grid_override(data))


# ---------------------------------------------------------------------------
# Applying an override
# ---------------------------------------------------------------------------


def test_a_penalty_moves_a_driver_back_and_the_cars_behind_move_up() -> None:
    """Piastri from 2nd to 4th: Hamilton and Leclerc move up one place each."""
    assert starts(apply({"grid": {"piastri": 4}})) == {
        "norris": (1.0, 0.0),
        "piastri": (4.0, 0.0),
        "hamilton": (2.0, 0.0),
        "leclerc": (3.0, 0.0),
        "russell": (5.0, 0.0),
    }


def test_pit_lane_starters_go_behind_the_whole_grid() -> None:
    """Norris from the pit lane: four cars on the grid, Norris flagged one slot behind them."""
    assert starts(apply({"pitlane": ["norris"]})) == {
        "norris": (5.0, 1.0),
        "piastri": (1.0, 0.0),
        "hamilton": (2.0, 0.0),
        "leclerc": (3.0, 0.0),
        "russell": (4.0, 0.0),
    }


def test_grid_slots_and_pit_lane_starts_together() -> None:
    """Russell to pole and Hamilton to the pit lane; the rest keep their order."""
    assert starts(apply({"grid": {"russell": 1}, "pitlane": ["hamilton"]})) == {
        "norris": (2.0, 0.0),
        "piastri": (3.0, 0.0),
        "hamilton": (5.0, 1.0),
        "leclerc": (4.0, 0.0),
        "russell": (1.0, 0.0),
    }


def test_a_pit_lane_start_in_the_data_stays_unless_given_a_slot() -> None:
    """For a race already run, its own pit-lane starter stays there unless the override moves it."""
    race = race_rows(grid=(1, 2, 3, 5, 4), pitlane=(0, 0, 0, 1, 0))
    assert starts(apply({"grid": {"russell": 1}}, race))["leclerc"] == (5.0, 1.0)
    moved = starts(apply({"grid": {"leclerc": 2}}, race))
    assert moved == {
        "norris": (1.0, 0.0),
        "piastri": (3.0, 0.0),
        "hamilton": (4.0, 0.0),
        "leclerc": (2.0, 0.0),
        "russell": (5.0, 0.0),
    }


def test_only_the_grid_columns_change_and_the_input_is_left_alone() -> None:
    """Qualifying gaps stay with their driver; the original rows keep their old grid."""
    race = race_rows()
    moved = apply({"grid": {"norris": 5}}, race)
    gaps = dict(zip(moved["driver_id"], moved["quali_gap_pole_pct"], strict=True))
    assert gaps == dict(zip(race["driver_id"], race["quali_gap_pole_pct"], strict=True))
    assert race["grid"].tolist() == [1.0, 2.0, 3.0, 4.0, 5.0]
    assert moved["grid"].dtype == "float64"
    assert moved["pitlane_start"].dtype == "float64"


def test_rows_come_back_in_the_new_starting_order() -> None:
    """Like the features table: sorted by grid, so the simulation sees the cars in that order."""
    moved = apply({"grid": {"russell": 1}, "pitlane": ["norris"]})
    assert moved["driver_id"].tolist() == ["russell", "piastri", "hamilton", "leclerc", "norris"]
    assert moved["grid"].tolist() == [1.0, 2.0, 3.0, 4.0, 5.0]


# ---------------------------------------------------------------------------
# Overrides that don't fit the race
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "hint"),
    [
        ("NOR", "did you mean 'norris'?"),
        ("hamiltn", "did you mean 'hamilton'?"),
    ],
)
def test_an_unknown_driver_gets_a_suggestion(name, hint) -> None:
    """A driver code or a typo points at the right driver id."""
    with pytest.raises(GridOverrideError, match=hint):
        apply({"grid": {name: 1}})


def test_a_driver_not_in_the_race_lists_who_is() -> None:
    """With no close match, the message lists every entered driver."""
    with pytest.raises(GridOverrideError, match="Entered: hamilton, leclerc") as error:
        apply({"pitlane": ["senna"]})
    assert "did you mean" not in str(error.value)


def test_a_slot_beyond_the_grid_is_refused() -> None:
    """Five cars means five slots, and four once a car starts from the pit lane."""
    with pytest.raises(GridOverrideError, match="5 grid slots"):
        apply({"grid": {"norris": 6}})
    with pytest.raises(GridOverrideError, match=r"4 grid slots \(with 1 in the pit lane\)"):
        apply({"grid": {"norris": 5}, "pitlane": ["piastri"]})


@pytest.mark.parametrize(
    ("data", "message"),
    [
        ({"season": 2025, "grid": {"norris": 2}}, "for season 2025, but the race is 2026 round 18"),
        ({"round": 17, "grid": {"norris": 2}}, "for round 17, but the race is 2026 round 18"),
    ],
)
def test_an_override_for_another_race_is_refused(data, message) -> None:
    """A file that names its race can't be applied to a different one."""
    with pytest.raises(GridOverrideError, match=message):
        apply(data)


# ---------------------------------------------------------------------------
# The file's shape
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("data", "message"),
    [
        ({}, "the override is empty"),
        ({"grid": {"norris": 0}}, "grid.norris: Input should be greater than or equal to 1"),
        ({"grid": {"norris": 2.5}}, "grid.norris: Input should be a valid integer"),
        ({"grid": {"norris": "3"}}, "grid.norris: Input should be a valid integer"),
        ({"grid": {"norris": True}}, "grid.norris: Input should be a valid integer"),
        ({"grid": {"norris": 1, "piastri": 1}}, "more than one driver is given slot 1"),
        ({"pitlane": ["norris", "norris"]}, "'pitlane' names 'norris' more than once"),
        ({"grid": {"norris": 1}, "pitlane": ["norris"]}, "can't have a grid slot and start"),
        ({"pitlnae": ["norris"]}, "pitlnae: Extra inputs are not permitted"),
        ({"season": "2026", "grid": {"norris": 1}}, "season: Input should be a valid integer"),
        ({"pitlane": "norris"}, "pitlane: Input should be a valid list"),
    ],
)
def test_malformed_overrides_are_refused_with_the_reason(data, message) -> None:
    """Each mistake gets a message naming where it is and what's wrong."""
    with pytest.raises(GridOverrideError, match="Invalid grid override") as error:
        parse_grid_override(data)
    assert message in str(error.value)


def test_a_checked_override_passes_through_unchanged() -> None:
    """A GridOverride (e.g. from a web request model) isn't checked twice."""
    override = GridOverride(grid={"norris": 2})
    assert parse_grid_override(override) is override


# ---------------------------------------------------------------------------
# Reading files
# ---------------------------------------------------------------------------

FILE_TEXT = json.dumps({"season": 2026, "round": 18, "grid": {"hamilton": 5}})


@pytest.mark.parametrize(
    "raw",
    [
        FILE_TEXT.encode("utf-8"),
        codecs.BOM_UTF8 + FILE_TEXT.encode("utf-8"),
        codecs.BOM_UTF16_LE + FILE_TEXT.encode("utf-16-le"),
    ],
    ids=["utf-8", "utf-8 with BOM", "utf-16 (PowerShell 5.1)"],
)
def test_override_files_load_in_every_windows_encoding(tmp_path, raw) -> None:
    """VS Code, Notepad and PowerShell's `>` all write files that load."""
    path = tmp_path / "penalties.json"
    path.write_bytes(raw)
    assert load_grid_override(path) == GridOverride(season=2026, round=18, grid={"hamilton": 5})


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        (b'{"grid": {"hamilton": 5}', "Invalid JSON"),
        (b"\x80\x81{}", "isn't UTF-8 or UTF-16"),
        (b'{"grid": {"hamilton": 0}}', "greater than or equal to 1"),
    ],
)
def test_a_bad_file_names_itself_and_the_problem(tmp_path, raw, message) -> None:
    """Broken JSON, an odd encoding or a bad value: the message names the file."""
    path = tmp_path / "penalties.json"
    path.write_bytes(raw)
    with pytest.raises(GridOverrideError, match=message) as error:
        load_grid_override(path)
    assert "penalties.json" in str(error.value)


def test_a_missing_file_is_explained(tmp_path) -> None:
    """No such file: a clear message instead of a traceback."""
    with pytest.raises(GridOverrideError, match="Can't read"):
        load_grid_override(tmp_path / "missing.json")
