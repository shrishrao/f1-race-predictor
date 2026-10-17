"""Tests for the practice laps table: stints, lap flags, deleted laps, green-flag windows."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pandas as pd
import pytest

from f1predict.data.mapping import round_entries
from f1predict.data.openf1 import PracticeData, PracticeSession
from f1predict.data.practice import (
    build_practice_season,
    deleted_laps,
    lap_table,
    non_green_windows,
)
from f1predict.data.tables import results_table

SINGAPORE_FP1 = PracticeSession(
    season=2026,
    round=17,
    session="FP1",
    session_key=11378,
    start=datetime(2026, 10, 9, 8, 30, tzinfo=UTC),
    end=datetime(2026, 10, 9, 9, 30, tzinfo=UTC),
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def utc(hour: int, minute: int, second: int = 0) -> pd.Timestamp:
    """A timestamp on Singapore FP1's day (9 October 2026, UTC)."""
    return pd.Timestamp(datetime(2026, 10, 9, hour, minute, second, tzinfo=UTC))


def message(at: str, category: str, text: str, **extra: Any) -> dict[str, Any]:
    """Build one race-control message at ``HH:MM:SS`` on Singapore FP1's day."""
    return {"date": f"2026-10-09T{at}+00:00", "category": category, "message": text, **extra}


@pytest.fixture
def singapore(openf1_fixture) -> PracticeData:
    """Singapore 2026 FP1 from saved OpenF1 data (laps for NOR, STR and HUL only)."""
    return PracticeData(
        session=SINGAPORE_FP1,
        drivers=openf1_fixture("drivers_11378.json"),
        laps=openf1_fixture("laps_11378.json"),
        stints=openf1_fixture("stints_11378.json"),
        race_control=openf1_fixture("race_control_11378.json"),
    )


@pytest.fixture
def laps(singapore) -> pd.DataFrame:
    """The laps table for Singapore FP1, indexed by (driver code, lap number)."""
    return lap_table(singapore).set_index(["driver_code", "lap_number"])


# ---------------------------------------------------------------------------
# The laps table on real data
# ---------------------------------------------------------------------------


def test_only_timed_laps_are_kept(laps) -> None:
    """66 laps were recorded for the three drivers; the 4 without a time are left out."""
    assert len(laps) == 62
    assert laps["lap_time_s"].notna().all()


def test_stints_give_compound_and_tyre_age(laps) -> None:
    """Norris's third stint is on softs, starting from a fresh set."""
    assert laps.loc[("NOR", 18), "compound"] == "SOFT"
    assert laps.loc[("NOR", 18), "stint"] == 3
    assert laps.loc[("NOR", 18), "tyre_age"] == 0
    assert laps.loc[("NOR", 21), "tyre_age"] == 3


def test_out_laps_and_in_laps(laps) -> None:
    """A stint starts with an out-lap and its last lap is the in-lap."""
    assert laps.loc[("NOR", 8), "is_out_lap"]
    assert laps.loc[("NOR", 7), "is_in_lap"]
    assert laps.loc[("NOR", 17), "is_in_lap"]
    assert not laps.loc[("NOR", 10), "is_in_lap"]


def test_deleted_laps_are_found_by_lap_time(laps) -> None:
    """Race control deleted STR 1:44.142 and NOR 1:38.239 and 1:52.494 (track limits)."""
    deleted = sorted(laps.index[laps["is_deleted"]])
    assert deleted == [("NOR", 2), ("NOR", 9), ("STR", 2)]


def test_laps_under_yellow_or_vsc_are_not_green(laps) -> None:
    """Norris's lap 14 ran into the VSC; lap 13 just before it and lap 15 after were green."""
    assert not laps.loc[("NOR", 14), "is_green"]
    assert laps.loc[("NOR", 13), "is_green"]
    assert laps.loc[("NOR", 15), "is_green"]


def test_accurate_means_sectors_add_up(laps) -> None:
    """Out-laps miss a sector time, so they are not "accurate"; normal laps are."""
    assert not laps.loc[("NOR", 8), "is_accurate"]
    assert laps.loc[("NOR", 10), "is_accurate"]


# ---------------------------------------------------------------------------
# Race-control windows and deletions, on hand-built messages
# ---------------------------------------------------------------------------


def test_sector_yellow_lasts_until_that_sector_clears() -> None:
    """Two sectors yellow at once make one window, closed by their CLEARs."""
    windows = non_green_windows(
        [
            message(
                "08:40:00",
                "Flag",
                "YELLOW IN TRACK SECTOR 3",
                flag="YELLOW",
                scope="Sector",
                sector=3,
            ),
            message(
                "08:40:00",
                "Flag",
                "YELLOW IN TRACK SECTOR 4",
                flag="YELLOW",
                scope="Sector",
                sector=4,
            ),
            message(
                "08:40:05",
                "Flag",
                "CLEAR IN TRACK SECTOR 3",
                flag="CLEAR",
                scope="Sector",
                sector=3,
            ),
            message(
                "08:40:07",
                "Flag",
                "CLEAR IN TRACK SECTOR 4",
                flag="CLEAR",
                scope="Sector",
                sector=4,
            ),
        ],
        utc(9, 30),
    )
    assert windows == [(utc(8, 40, 0), utc(8, 40, 7))]


def test_red_flag_lasts_until_the_track_goes_green() -> None:
    """A red flag stays open through other messages until the next track-wide GREEN."""
    windows = non_green_windows(
        [
            message("08:50:00", "Flag", "RED FLAG", flag="RED", scope="Track"),
            message("08:55:00", "Other", "CAR 1 (NOR) NOTED"),
            message("09:05:00", "Flag", "GREEN LIGHT - PIT EXIT OPEN", flag="GREEN", scope="Track"),
        ],
        utc(9, 30),
    )
    assert windows == [(utc(8, 50), utc(9, 5))]


def test_unclosed_window_ends_with_the_session_or_its_last_message() -> None:
    """A yellow never cleared closes at the scheduled end, or later if the session overran."""
    windows = non_green_windows(
        [
            message("09:25:00", "Flag", "YELLOW", flag="YELLOW", scope="Sector", sector=2),
            message("09:34:00", "Other", "CHECK TRACK LIMITS"),
        ],
        utc(9, 30),
    )
    assert windows == [(utc(9, 25), utc(9, 34))]


def test_vsc_window_from_deployed_to_ending() -> None:
    """A virtual safety car counts as non-green from DEPLOYED to ENDING."""
    windows = non_green_windows(
        [
            message("09:09:38", "SafetyCar", "VSC DEPLOYED"),
            message("09:10:47", "SafetyCar", "VSC ENDING"),
        ],
        utc(9, 30),
    )
    assert windows == [(utc(9, 9, 38), utc(9, 10, 47))]


def test_reinstated_lap_is_no_longer_deleted() -> None:
    """A later REINSTATED message cancels the deletion of the same lap."""
    laps = [
        {
            "driver_number": 4,
            "lap_number": 5,
            "lap_duration": 90.5,
            "date_start": "2026-10-09T08:40:00+00:00",
        },
        {
            "driver_number": 4,
            "lap_number": 7,
            "lap_duration": 91.25,
            "date_start": "2026-10-09T08:45:00+00:00",
        },
    ]
    messages = [
        message("08:42:00", "Other", "CAR 4 (NOR) TIME 1:30.500 DELETED - TRACK LIMITS AT TURN 5"),
        message("08:47:00", "Other", "CAR 4 (NOR) TIME 1:31.250 DELETED - TRACK LIMITS AT TURN 9"),
        message("08:50:00", "Other", "CAR 4 (NOR) TIME 1:31.250 REINSTATED"),
    ]
    assert deleted_laps(messages, laps) == {(4, 5)}


# ---------------------------------------------------------------------------
# A season: mapping drivers and auditing stand-ins
# ---------------------------------------------------------------------------


def test_drivers_are_mapped_to_jolpica_ids(singapore, jolpica_races) -> None:
    """Every lap gets the driver id of a race entrant; no stand-ins in this session."""
    entries = round_entries(results_table(jolpica_races("results_2026_17.json")))
    laps, stand_ins, warnings = build_practice_season([singapore], entries)
    assert set(laps["driver_id"]) == {"norris", "stroll", "hulkenberg"}
    assert stand_ins.empty
    assert warnings == ["2026 round 17 FP1: only 3 drivers set a lap time"]


def test_stand_in_laps_are_kept_and_audited(singapore, jolpica_races) -> None:
    """A rookie in Hulkenberg's car keeps their laps (no driver id) and lands in the audit."""
    results = results_table(jolpica_races("results_2026_17.json"))
    entries = round_entries(results[results["driver_code"] != "HUL"])
    laps, stand_ins, _ = build_practice_season([singapore], entries)
    assert laps.loc[laps["driver_code"] == "HUL", "driver_id"].isna().all()
    assert stand_ins[["round", "session", "driver_number", "driver_code"]].values.tolist() == [
        [17, "FP1", 27, "HUL"]
    ]


def test_round_without_an_entry_list_waits_for_qualifying(singapore) -> None:
    """Before qualifying there is no entry list, so laps stay unmapped and a warning says so."""
    empty = round_entries()
    laps, stand_ins, warnings = build_practice_season([singapore], empty)
    assert laps["driver_id"].isna().all()
    assert stand_ins.empty
    assert "no Jolpica entry list yet" in warnings[0]
