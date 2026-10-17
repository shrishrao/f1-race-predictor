"""Tests for the feature builders: grid, qualifying gaps, rolling form, and practice pace."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from f1predict.data.lineage import LINEAGE, UnknownConstructorError, team_lineage
from f1predict.data.practice import PRACTICE_DTYPES
from f1predict.data.tables import typed_frame
from f1predict.features.form import fixed_points, form_features
from f1predict.features.practice import best_lap_features, long_run_features, long_runs
from f1predict.features.quali import grid_features, quali_features

# ---------------------------------------------------------------------------
# Lineage
# ---------------------------------------------------------------------------


def test_rebrands_share_one_lineage() -> None:
    """Force India, Racing Point and Aston Martin are one team; so are Sauber, Alfa and Audi."""
    assert {team_lineage(c) for c in ("force_india", "racing_point", "aston_martin")} == {
        "aston_martin"
    }
    assert {team_lineage(c) for c in ("sauber", "alfa", "audi")} == {"audi"}
    assert team_lineage("cadillac") == "cadillac"


def test_unknown_constructor_fails_loudly() -> None:
    """A team that isn't in the map must be added on purpose, not guessed."""
    with pytest.raises(UnknownConstructorError, match="lineage map"):
        team_lineage("andretti")


def test_every_constructor_in_the_fixtures_is_mapped(jolpica_races) -> None:
    """All constructor ids in the saved Jolpica responses are in the lineage map."""
    ids = {
        entry["Constructor"]["constructorId"]
        for name in ("results_2018_1.json", "results_2026_17.json")
        for race in jolpica_races(name)
        for entry in race["Results"]
    }
    assert ids <= set(LINEAGE)


# ---------------------------------------------------------------------------
# Grid and qualifying
# ---------------------------------------------------------------------------


def test_pit_lane_start_goes_behind_the_last_grid_slot() -> None:
    """Grid 0 means pit lane: the car is flagged and placed one behind the last slot."""
    entries = pd.DataFrame(
        {"season": 2024, "round": 1, "driver_id": ["a", "b", "c"], "grid": [1, 0, 2]}
    ).astype({"grid": "Int64"})
    out = grid_features(entries).set_index("driver_id")
    assert out.loc["b", "grid"] == 3
    assert out.loc["b", "pitlane_start"] == 1
    assert out.loc["a", "pitlane_start"] == 0


def test_quali_gaps_to_pole_and_teammate() -> None:
    """Teammates are compared in the latest session both reached (Q2 here, not Q1)."""
    quali = pd.DataFrame(
        {
            "season": 2024,
            "round": 1,
            "driver_id": ["pole", "mate", "solo"],
            "constructor_id": ["team_a", "team_a", "team_b"],
            "q1_s": [91.0, 90.5, 92.0],
            "q2_s": [90.0, 90.9, np.nan],
            "q3_s": [89.0, np.nan, np.nan],
        }
    )
    out = quali_features(quali).set_index("driver_id")
    assert out.loc["pole", "quali_gap_pole_pct"] == 0
    assert out.loc["solo", "quali_gap_pole_pct"] == pytest.approx((92.0 / 89.0 - 1) * 100)
    assert out.loc["pole", "quali_gap_teammate_pct"] == pytest.approx((90.0 / 90.9 - 1) * 100)
    assert out.loc["mate", "quali_gap_teammate_pct"] == pytest.approx((90.9 / 90.0 - 1) * 100)
    assert np.isnan(out.loc["solo", "quali_gap_teammate_pct"])


# ---------------------------------------------------------------------------
# Rolling form
# ---------------------------------------------------------------------------


def history_row(race_seq: int, driver: str, team: str, position: int, status: str) -> dict:
    """One results row for the hand-built form history."""
    classified = status == "finished" or position <= 10
    return {
        "season": 2024,
        "round": race_seq + 1,
        "race_seq": race_seq,
        "driver_id": driver,
        "team_id": team,
        "circuit_id": "monaco" if race_seq % 2 else "bahrain",
        "position": position,
        "position_text": str(position) if classified else "R",
        "status_class": status,
    }


@pytest.fixture
def history() -> pd.DataFrame:
    """Four races for two drivers of one team; driver b retires in race 2."""
    rows = []
    for seq, (pos_a, pos_b, status_b) in enumerate(
        [(1, 3, "finished"), (2, 9, "dnf"), (4, 5, "finished"), (6, 2, "finished")]
    ):
        rows.append(history_row(seq, "a", "team", pos_a, "finished"))
        rows.append(history_row(seq, "b", "team", pos_b, status_b))
    return pd.DataFrame(rows).astype({"position": "Int64", "position_text": "string"})


def test_fixed_points_ignore_status_but_need_a_classified_position(history) -> None:
    """A classified retiree in P9 still gets 2 points; an unclassified car gets none."""
    frame = pd.DataFrame({"position": [1, 9, 15], "position_text": ["1", "9", "R"]}).astype(
        {"position": "Int64", "position_text": "string"}
    )
    assert fixed_points(frame).tolist() == [25.0, 2.0, 0.0]


def test_form_only_uses_earlier_races(history) -> None:
    """Before race 4 (seq 3), b's last finishes are 3 and 5 (the DNF is skipped)."""
    entries = history[history["race_seq"] == 3]
    out = form_features(entries, history).set_index("driver_id")
    assert out.loc["b", "drv_avg_finish_5"] == pytest.approx(4.0)
    assert out.loc["a", "drv_avg_finish_5"] == pytest.approx((1 + 2 + 4) / 3)


def test_team_points_use_the_last_three_races(history) -> None:
    """Team points before race 4: races 1-3 scored 25+15, 18+2, 12+10 -> mean 82/3."""
    entries = history[history["race_seq"] == 3]
    out = form_features(entries, history)
    assert out["team_pts_3"].iloc[0] == pytest.approx((40 + 20 + 22) / 3)


def test_dnf_rate_is_shrunk_toward_the_overall_rate(history) -> None:
    """b: 1 DNF in 3 starts, overall 1 in 6, prior strength 5 -> (1 + 5/6) / (3 + 5)."""
    entries = history[history["race_seq"] == 3]
    out = form_features(entries, history).set_index("driver_id")
    assert out.loc["b", "drv_dnf_rate"] == pytest.approx((1 + 5 * (1 / 6)) / (3 + 5))


def test_first_race_has_no_form(history) -> None:
    """With nothing before race 1, form features are NaN rather than invented."""
    out = form_features(history[history["race_seq"] == 0], history)
    assert out[["drv_avg_finish_5", "team_pts_3", "drv_dnf_rate"]].isna().all().all()


# ---------------------------------------------------------------------------
# Practice: a hand-built long run with a known answer
# ---------------------------------------------------------------------------


def lap(driver: str | None, number: int, lap_no: int, seconds: float, **flags) -> dict:
    """One FP2 lap; flags default to a clean, green, slick push lap in stint 1."""
    return {
        "season": 2024,
        "round": 1,
        "session": "FP2",
        "session_key": 1,
        "driver_number": number,
        "driver_code": (driver or "ROO")[:3].upper(),
        "driver_id": driver,
        "lap_number": lap_no,
        "lap_start": datetime(2024, 3, 1, tzinfo=UTC) + timedelta(minutes=lap_no),
        "lap_time_s": seconds,
        "stint": flags.get("stint", 1),
        "compound": flags.get("compound", "MEDIUM"),
        "tyre_age": lap_no,
        "is_out_lap": flags.get("out", False),
        "is_in_lap": flags.get("in_", False),
        "is_deleted": flags.get("deleted", False),
        "is_green": flags.get("green", True),
        "is_accurate": True,
    }


@pytest.fixture
def practice() -> pd.DataFrame:
    """Two drivers and a stand-in in one FP2 session, with known long-run medians.

    Driver "a", stint 1: out-lap, six push laps (95.0-96.0), a 120 s cool-down lap and an
    in-lap. Only the six push laps survive, median 95.5. Stint 2 has too few clean laps.
    Driver "b": five push laps around 96.0 (median 96.0) plus a faster but deleted lap.
    The stand-in (no driver id) runs a 95.0 median, which is the session's best.
    """
    rows = [lap("a", 1, 1, 130.0, out=True)]
    rows += [lap("a", 1, 2 + i, t) for i, t in enumerate([95.0, 95.2, 95.4, 95.6, 95.8, 96.0])]
    rows += [lap("a", 1, 8, 120.0), lap("a", 1, 9, 125.0, in_=True)]
    rows += [lap("a", 1, 10 + i, 95.3, stint=2, green=i > 2) for i in range(6)]
    rows += [lap("b", 2, 1 + i, t) for i, t in enumerate([95.9, 96.0, 96.1, 95.8, 96.2])]
    rows += [lap("b", 2, 6, 93.0, deleted=True), lap("b", 2, 7, 92.5, compound="INTERMEDIATE")]
    rows += [lap(None, 99, 1 + i, 95.0) for i in range(5)]
    return typed_frame(rows, PRACTICE_DTYPES)


def test_long_run_detection_matches_the_hand_count(practice) -> None:
    """Exactly three long runs survive: a's first stint, b's stint, and the stand-in's."""
    runs = long_runs(practice).set_index(["driver_number", "stint"])
    assert sorted(runs.index) == [(1, 1), (2, 1), (99, 1)]
    assert runs.loc[(1, 1), "laps"] == 6
    assert runs.loc[(1, 1), "median_s"] == pytest.approx(95.5)
    assert runs.loc[(2, 1), "median_s"] == pytest.approx(96.0)


def test_long_run_gap_is_measured_against_the_fastest_car_even_a_stand_in(practice) -> None:
    """The stand-in's 95.0 median is the reference; stand-ins get no feature row."""
    out = long_run_features(practice).set_index("driver_id")
    assert sorted(out.index) == ["a", "b"]
    assert out.loc["a", "fp2_longrun_gap_pct"] == pytest.approx((95.5 / 95.0 - 1) * 100)
    assert out.loc["b", "longrun_best_gap_pct"] == pytest.approx((96.0 / 95.0 - 1) * 100)
    assert np.isnan(out.loc["a", "fp1_longrun_gap_pct"])


def test_best_lap_ignores_deleted_and_wet_tyre_laps(practice) -> None:
    """b's deleted 93.0 and intermediate 92.5 don't count; the best clean lap is 95.0 (a)."""
    out = best_lap_features(practice).set_index("driver_id")
    assert out.loc["a", "fp2_best_gap_pct"] == pytest.approx(0.0)
    assert out.loc["b", "fp2_best_gap_pct"] == pytest.approx((95.8 / 95.0 - 1) * 100)
