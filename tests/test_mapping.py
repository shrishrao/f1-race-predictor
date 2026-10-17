"""Tests for matching practice drivers to Jolpica driver ids, round by round."""

from __future__ import annotations

import pandas as pd
import pytest

from f1predict.data.mapping import (
    DriverMappingError,
    UnknownDriverError,
    map_driver,
    map_session_drivers,
    round_entries,
)
from f1predict.data.tables import results_table

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def entries_for(rows: list[tuple[str, str | None, int | None]], round_number: int = 1):
    """Build a one-round entry list from (driver_id, code, car number) tuples."""
    return pd.DataFrame(
        [
            {
                "season": 2026,
                "round": round_number,
                "driver_id": d,
                "driver_code": c,
                "car_number": n,
            }
            for d, c, n in rows
        ]
    ).astype({"driver_code": "string", "car_number": "Int64", "driver_id": "string"})


ENTRIES = entries_for(
    [("max_verstappen", "VER", 3), ("norris", "NOR", 1), ("hulkenberg", "HUL", 27)]
)


def practice_drivers(rows: list[tuple[int, str | None]]) -> pd.DataFrame:
    """Build a practice-session driver list from (car number, code) tuples."""
    return pd.DataFrame(rows, columns=["driver_number", "driver_code"])


# ---------------------------------------------------------------------------
# One driver
# ---------------------------------------------------------------------------


def test_code_and_number_agree() -> None:
    """The normal case: code and car number both point to the same entrant."""
    assert map_driver("VER", 3, ENTRIES) == "max_verstappen"


def test_number_is_the_fallback_when_the_code_is_missing() -> None:
    """With no code, the car number used that weekend identifies the driver."""
    assert map_driver(None, 27, ENTRIES) == "hulkenberg"


def test_code_alone_is_enough_when_the_number_is_unknown() -> None:
    """A driver whose number isn't in the entry list still maps by code."""
    assert map_driver("NOR", 4, ENTRIES) == "norris"


def test_unknown_driver_raises() -> None:
    """A driver who matches no entrant raises UnknownDriverError (a likely FP1 stand-in)."""
    with pytest.raises(UnknownDriverError, match="not in this round's entry list"):
        map_driver("HER", 45, ENTRIES)


def test_code_and_number_pointing_at_different_drivers_raises() -> None:
    """A contradiction is a real data problem, not something to guess about."""
    with pytest.raises(DriverMappingError, match="points to"):
        map_driver("VER", 1, ENTRIES)


def test_ambiguous_entry_list_raises() -> None:
    """Two entrants sharing a code in one round can't be told apart."""
    entries = entries_for([("driver_a", "ABC", 10), ("driver_b", "ABC", 11)])
    with pytest.raises(DriverMappingError, match="several entrants"):
        map_driver("ABC", None, entries)


def test_round_specific_numbers_are_used() -> None:
    """Verstappen raced #33 in 2021: the round's own entry list maps 33, not today's 3."""
    entries_2021 = entries_for([("max_verstappen", "VER", 33)])
    assert map_driver("VER", 33, entries_2021) == "max_verstappen"


# ---------------------------------------------------------------------------
# A whole session
# ---------------------------------------------------------------------------


def test_every_real_singapore_driver_maps(jolpica_races, openf1_fixture) -> None:
    """All 22 OpenF1 drivers of Singapore 2026 FP1 map to the Jolpica race entrants."""
    entries = round_entries(results_table(jolpica_races("results_2026_17.json")))
    drivers = practice_drivers(
        [(d["driver_number"], d["name_acronym"]) for d in openf1_fixture("drivers_11378.json")]
    )
    mapping, stand_ins = map_session_drivers(drivers, entries, context="2026 round 17 FP1")
    assert len(mapping) == 22
    assert stand_ins == []
    assert mapping[3] == "max_verstappen"
    assert mapping[1] == "norris"


def test_stand_ins_are_reported_not_dropped_or_fatal() -> None:
    """An FP1 rookie is returned in the stand-in list; the race drivers still map."""
    drivers = practice_drivers([(3, "VER"), (1, "NOR"), (45, "HER")])
    mapping, stand_ins = map_session_drivers(drivers, ENTRIES, context="test")
    assert mapping == {3: "max_verstappen", 1: "norris"}
    assert stand_ins == [{"driver_number": 45, "driver_code": "HER"}]


def test_two_practice_drivers_claiming_one_entrant_raises() -> None:
    """Two cars mapping to the same driver means something is wrong with the data."""
    drivers = practice_drivers([(3, "VER"), (33, "VER")])
    with pytest.raises(DriverMappingError, match="several practice drivers"):
        map_session_drivers(drivers, ENTRIES, context="2021 round 1 FP1")


def test_contradictions_name_the_session() -> None:
    """Errors say which session they came from, so they're easy to track down."""
    with pytest.raises(DriverMappingError, match="2024 round 5 FP2"):
        map_session_drivers(practice_drivers([(1, "VER")]), ENTRIES, context="2024 round 5 FP2")


# ---------------------------------------------------------------------------
# Entry lists
# ---------------------------------------------------------------------------


def test_entry_list_is_the_union_of_results_qualifying_and_sprint() -> None:
    """Someone only in qualifying (e.g. a non-starter) still counts as entered."""
    race = entries_for([("norris", "NOR", 1)])
    quali = entries_for([("norris", "NOR", 1), ("hulkenberg", "HUL", 27)])
    combined = round_entries(race, quali, pd.DataFrame())
    assert sorted(combined["driver_id"]) == ["hulkenberg", "norris"]
