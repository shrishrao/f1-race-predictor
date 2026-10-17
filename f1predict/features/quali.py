"""Grid and qualifying features: starting slot, gap to pole, gap to teammate.

All of these are known on Saturday night, before the race, so they are leakage-safe as
long as they only use the race's own grid and qualifying session.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

KEYS = ["season", "round", "driver_id"]
Q_SESSIONS = ("q3_s", "q2_s", "q1_s")

# ---------------------------------------------------------------------------
# Grid
# ---------------------------------------------------------------------------


def grid_features(entries: pd.DataFrame) -> pd.DataFrame:
    """Starting slot and pit-lane flag per driver per race.

    Jolpica marks a pit-lane start as grid 0. Those cars get ``pitlane_start = 1`` and a
    ``grid`` one slot behind the last car on the grid, so "higher number = further back"
    holds for every driver.
    """
    frame = entries[[*KEYS, "grid"]].copy()
    raw = frame["grid"].astype("Float64")
    pitlane = raw.eq(0).fillna(False)
    last_slot = raw.where(~pitlane).groupby([frame["season"], frame["round"]]).transform("max")
    frame["grid"] = raw.where(~pitlane, last_slot + 1).astype("float64")
    frame["pitlane_start"] = pitlane.astype("int64")
    return frame


# ---------------------------------------------------------------------------
# Qualifying gaps
# ---------------------------------------------------------------------------


def quali_features(qualifying: pd.DataFrame) -> pd.DataFrame:
    """Gap to the fastest qualifying lap and gap to the teammate, both in percent.

    ``quali_gap_pole_pct`` compares a driver's best time from Q1-Q3 with the fastest time
    anyone set in the session. ``quali_gap_teammate_pct`` compares the two cars of a team in
    the latest part of qualifying where both set a time (Q3, else Q2, else Q1), so both laps
    were driven on the same track at about the same time. It is NaN when a team ran one car
    or the pair never shared a session.
    """
    frame = qualifying[[*KEYS, "constructor_id", *Q_SESSIONS]].copy()
    best = frame[list(Q_SESSIONS)].min(axis=1)
    fastest = best.groupby([frame["season"], frame["round"]]).transform("min")
    frame["quali_gap_pole_pct"] = (best / fastest - 1) * 100

    frame["quali_gap_teammate_pct"] = np.nan
    for _, team in frame.groupby(["season", "round", "constructor_id"]):
        if len(team) != 2:
            continue
        first, second = team.index
        for session in Q_SESSIONS:
            a, b = frame.at[first, session], frame.at[second, session]
            if pd.notna(a) and pd.notna(b):
                frame.at[first, "quali_gap_teammate_pct"] = (a / b - 1) * 100
                frame.at[second, "quali_gap_teammate_pct"] = (b / a - 1) * 100
                break
    return frame[[*KEYS, "quali_gap_pole_pct", "quali_gap_teammate_pct"]]
