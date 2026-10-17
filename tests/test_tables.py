"""Tests for tidy tables built from real Jolpica responses, their checks, and parquet storage."""

from __future__ import annotations

import math
from datetime import UTC, datetime
from typing import Any

import pandas as pd
import pytest

from f1predict.data.tables import (
    DataValidationError,
    build_tables,
    load_table,
    parse_lap_time,
    parse_utc,
    qualifying_table,
    results_table,
    save_season_tables,
    schedule_table,
    validate_season,
)

NOW = datetime(2026, 10, 17, 12, 0, tzinfo=UTC)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def utc(*parts: int) -> pd.Timestamp:
    """Shorthand for a UTC timestamp."""
    return pd.Timestamp(datetime(*parts, tzinfo=UTC))


def tables_2018(jolpica_races, **overrides: Any) -> dict[str, pd.DataFrame]:
    """Tables for the 2018 season opener (a finished season), with optional replacements."""
    parts = {
        "schedule": jolpica_races("schedule_2018.json"),
        "results": jolpica_races("results_2018_1.json"),
        "qualifying": jolpica_races("qualifying_2018_1.json"),
        "sprint": [],
    }
    parts.update(overrides)
    return build_tables(**parts)


# ---------------------------------------------------------------------------
# Field parsing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "seconds"), [("1:23.456", 83.456), ("59.123", 59.123), ("2:00.000", 120.0)]
)
def test_lap_times_become_seconds(text: str, seconds: float) -> None:
    """Minute:second lap times turn into plain seconds."""
    assert parse_lap_time(text) == pytest.approx(seconds)


@pytest.mark.parametrize("text", ["", None, "DNF", "1:2x.3"])
def test_missing_or_odd_lap_times_become_nan(text: str | None) -> None:
    """Blank, missing or malformed times become NaN instead of crashing."""
    assert math.isnan(parse_lap_time(text))


def test_parse_utc_uses_noon_when_only_a_date_is_known() -> None:
    """Old practice sessions have no time of day; noon UTC stands in for it."""
    assert parse_utc("2018-03-23", None) == datetime(2018, 3, 23, 12, 0, tzinfo=UTC)
    assert parse_utc("2026-10-11", "12:00:00Z") == datetime(2026, 10, 11, 12, 0, tzinfo=UTC)
    assert parse_utc(None, "12:00:00Z") is None


# ---------------------------------------------------------------------------
# Schedule
# ---------------------------------------------------------------------------


def test_schedule_has_one_row_per_weekend_with_utc_session_times(jolpica_races) -> None:
    """Session times are UTC timestamps; a sprint weekend has no FP2."""
    schedule = schedule_table(jolpica_races("schedule_2026.json"))
    assert schedule["round"].tolist() == [1, 2, 16, 17, 18]
    assert str(schedule["race_start"].dtype) == "datetime64[ns, UTC]"
    singapore = schedule.set_index("round").loc[17]
    assert singapore["race_start"] == utc(2026, 10, 11, 12, 0)
    assert singapore["sprint_start"] == utc(2026, 10, 10, 9, 0)
    assert singapore["sprint_quali_start"] < singapore["sprint_start"]
    assert pd.isna(singapore["fp2_start"])


def test_sprint_weekends_and_the_new_venue(jolpica_races) -> None:
    """Sprint weekends are flagged; round 16 runs at Sepang."""
    schedule = schedule_table(jolpica_races("schedule_2026.json")).set_index("round")
    assert schedule["is_sprint_weekend"].to_dict() == {
        1: False,
        2: True,
        16: False,
        17: True,
        18: False,
    }
    assert schedule.loc[16, "circuit_id"] == "sepang"


def test_old_schedules_without_practice_times_still_parse(jolpica_races) -> None:
    """In 2018 practice has a date but no time; the race has both."""
    row = schedule_table(jolpica_races("schedule_2018.json")).iloc[0]
    assert row["fp1_start"] == utc(2018, 3, 23, 12, 0)
    assert row["race_start"] == utc(2018, 3, 25, 5, 10)


# ---------------------------------------------------------------------------
# Results and sprint
# ---------------------------------------------------------------------------


def test_singapore_2026_winner(jolpica_races) -> None:
    """The winner row carries the race-weekend car number (3), grid, points and time."""
    results = results_table(jolpica_races("results_2026_17.json")).set_index("position")
    assert len(results) == 22
    winner = results.loc[1]
    assert winner["driver_code"] == "VER"
    assert winner["car_number"] == 3
    assert winner["grid"] == 1
    assert winner["points"] == 25.0
    assert winner["time_ms"] == 6120785
    assert winner["status_class"] == "finished"


def test_classified_retiree_is_a_dnf_but_keeps_its_position(jolpica_races) -> None:
    """P18 retired late but was classified; P19 behind it finished a lap down."""
    results = results_table(jolpica_races("results_2026_17.json")).set_index("position")
    assert (results.loc[18, "position_text"], results.loc[18, "status_class"]) == ("18", "dnf")
    assert (results.loc[19, "status"], results.loc[19, "status_class"]) == ("Lapped", "finished")


def test_unclassified_retirements(jolpica_races) -> None:
    """Cars that retired early are marked "R" and count as DNFs."""
    results = results_table(jolpica_races("results_2026_17.json"))
    retired = results[results["position_text"] == "R"]
    assert len(retired) == 3
    assert set(retired["status_class"]) == {"dnf"}


def test_old_wording_from_2018(jolpica_races) -> None:
    """2018's "+1 Lap" counts as finished and named causes ("Engine") as DNFs."""
    results = results_table(jolpica_races("results_2018_1.json"))
    assert results["status_class"].value_counts().to_dict() == {"finished": 15, "dnf": 5}
    assert results.loc[results["status"] == "+1 Lap", "status_class"].item() == "finished"


def test_sprint_results_share_the_results_columns(jolpica_races) -> None:
    """Sprint rows parse with the same columns; a last-lap retiree keeps P14."""
    sprint = results_table(jolpica_races("sprint_2026_17.json"), "SprintResults")
    assert len(sprint) == 22
    assert set(sprint["status_class"]) == {"finished", "dnf"}
    p14 = sprint.set_index("position").loc[14]
    assert (p14["driver_code"], p14["status_class"]) == ("PIA", "dnf")


def test_integer_columns_allow_missing_values(jolpica_races) -> None:
    """Grid and position use pandas' nullable Int64, so a missing value stays missing."""
    results = results_table(jolpica_races("results_2026_17.json"))
    assert str(results["grid"].dtype) == "Int64"
    assert str(results["position"].dtype) == "Int64"


# ---------------------------------------------------------------------------
# Qualifying
# ---------------------------------------------------------------------------


def test_qualifying_times_are_seconds_and_gaps_are_nan(jolpica_races) -> None:
    """Pole's Q3 time parses; the driver with no Q1 time has NaN for all three."""
    quali = qualifying_table(jolpica_races("qualifying_2026_17.json")).set_index("position")
    assert len(quali) == 22
    assert quali.loc[1, "q3_s"] == pytest.approx(91.373)
    last = quali.loc[22]
    assert math.isnan(last["q1_s"])
    assert math.isnan(last["q2_s"])
    assert math.isnan(last["q3_s"])


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_complete_finished_season_passes(jolpica_races) -> None:
    """A finished season with results and qualifying for every round has no problems."""
    assert validate_season(tables_2018(jolpica_races), 2018, now=NOW) == []


def test_finished_season_with_a_missing_round_fails_loudly(jolpica_races) -> None:
    """A finished season must have results for every scheduled round."""
    with pytest.raises(DataValidationError, match=r"results: no data for rounds \[1\]"):
        validate_season(tables_2018(jolpica_races, results=[]), 2018, now=NOW)


def test_current_season_gaps_are_warnings_not_errors(jolpica_races) -> None:
    """In the season in progress, missing rounds are reported but do not stop the fetch."""
    tables = build_tables(
        schedule=jolpica_races("schedule_2026.json"),
        results=jolpica_races("results_2026_17.json"),
        qualifying=jolpica_races("qualifying_2026_17.json"),
        sprint=jolpica_races("sprint_2026_17.json"),
    )
    warnings = validate_season(tables, 2026, now=NOW)
    assert warnings == [
        "2026 results: no data for rounds [1, 2, 16]",
        "2026 qualifying: no data for rounds [1, 2, 16]",
        "2026 sprint: no data for rounds [2]",
    ]


def test_duplicate_rows_are_always_an_error(jolpica_races) -> None:
    """The same driver twice in one race means pages overlapped; that must never pass."""
    races = jolpica_races("results_2018_1.json")
    with pytest.raises(DataValidationError, match="duplicate"):
        validate_season(tables_2018(jolpica_races, results=races + races), 2018, now=NOW)


def test_empty_schedule_is_an_error() -> None:
    """A season without a schedule means the download went wrong."""
    tables = build_tables(schedule=[], results=[], qualifying=[], sprint=[])
    with pytest.raises(DataValidationError, match="schedule is empty"):
        validate_season(tables, 2018, now=NOW)


# ---------------------------------------------------------------------------
# Saving and loading
# ---------------------------------------------------------------------------


def test_tables_round_trip_through_parquet(tmp_path, jolpica_races) -> None:
    """What is saved loads back identically, types included; empty tables write no file."""
    tables = tables_2018(jolpica_races)
    counts = save_season_tables(tables, 2018, tmp_path)
    assert counts == {"schedule": 1, "results": 20, "qualifying": 20, "sprint": 0}
    pd.testing.assert_frame_equal(load_table("results", tmp_path), tables["results"])
    pd.testing.assert_frame_equal(load_table("schedule", tmp_path), tables["schedule"])
    assert not (tmp_path / "sprint" / "2018.parquet").exists()
    assert list(tmp_path.rglob("*.tmp")) == []


def test_empty_table_removes_an_old_file(tmp_path, jolpica_races) -> None:
    """Saving an empty table for a season deletes that season's stale file."""
    sprint = results_table(jolpica_races("sprint_2026_17.json"), "SprintResults")
    save_season_tables({"sprint": sprint}, 2026, tmp_path)
    assert (tmp_path / "sprint" / "2026.parquet").exists()
    save_season_tables({"sprint": sprint.iloc[0:0]}, 2026, tmp_path)
    assert not (tmp_path / "sprint" / "2026.parquet").exists()


def test_loading_before_fetching_says_what_to_run(tmp_path) -> None:
    """With no saved data, the error tells you to run the fetch command."""
    with pytest.raises(FileNotFoundError, match="f1predict fetch"):
        load_table("results", tmp_path)


def test_unknown_table_name_is_rejected(tmp_path) -> None:
    """Only the four known tables can be loaded."""
    with pytest.raises(ValueError, match="Unknown table"):
        load_table("laps", tmp_path)
