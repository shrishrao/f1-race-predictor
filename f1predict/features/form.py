"""Rolling form features: recent finishes, team points, and DNF rates.

Every value for race N comes only from races before N. The same pattern is used for each
feature:

1. Build a history table (one row per driver, team or circuit per race).
2. Compute a rolling statistic that includes each row's own race.
3. Attach to race N the latest value from a race *strictly before* N, using pandas
   ``merge_asof`` with ``allow_exact_matches=False``.

Step 3 is what makes this leakage-safe: race N's own result can never reach race N's row,
whatever steps 1 and 2 do.
"""

from __future__ import annotations

import pandas as pd

KEYS = ["season", "round", "driver_id"]
POINTS_TABLE = {1: 25, 2: 18, 3: 15, 4: 12, 5: 10, 6: 8, 7: 6, 8: 4, 9: 2, 10: 1}
FINISH_WINDOW = 5
TEAM_POINTS_WINDOW = 3
DNF_WINDOW = 10
DNF_PRIOR_STARTS = 5
CIRCUIT_PRIOR_STARTS = 20
FORM_COLUMNS = [
    "drv_avg_finish_5",
    "team_pts_3",
    "drv_dnf_rate",
    "team_dnf_rate",
    "circuit_dnf_rate",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def strictly_before(
    rows: pd.DataFrame, history: pd.DataFrame, columns: list[str], by: str | None = None
) -> pd.DataFrame:
    """Attach to each row the ``columns`` from the latest history row of an earlier race.

    Both frames need a ``race_seq`` column (races numbered in date order). With ``by``,
    only history rows with the same value (same driver, team or circuit) are considered.
    Rows with no earlier history get NaN.
    """
    keys = [by] if by else []
    left = rows.sort_values("race_seq", kind="stable")
    right = history.sort_values("race_seq", kind="stable")[["race_seq", *keys, *columns]]
    merged = pd.merge_asof(
        left, right, on="race_seq", by=by, allow_exact_matches=False, direction="backward"
    )
    return merged.set_index(left.index).reindex(rows.index)


def _rolling_sum(frame: pd.DataFrame, by: str, column: str, window: int) -> pd.Series:
    """Rolling sum of ``column`` over each group's last ``window`` rows (current row included)."""
    return frame.groupby(by)[column].transform(lambda s: s.rolling(window, min_periods=1).sum())


def _shrunk_rate(
    events: pd.Series, trials: pd.Series, prior: pd.Series, strength: int
) -> pd.Series:
    """Event rate pulled toward ``prior`` as if ``strength`` extra trials at the prior rate."""
    return (events.fillna(0) + strength * prior) / (trials.fillna(0) + strength)


def fixed_points(results: pd.DataFrame) -> pd.Series:
    """Points from the official position with one fixed table (25-18-15-...-1, race only).

    Using one table for every season keeps team points comparable: no fastest-lap point
    (2019-2024), no sprint points and no half points (Spa 2021).
    """
    classified = results["position_text"].str.fullmatch(r"\d+").fillna(False)
    position = results["position"].where(classified)
    return position.map(POINTS_TABLE).fillna(0.0).astype("float64")


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------


def form_features(entries: pd.DataFrame, results: pd.DataFrame) -> pd.DataFrame:
    """Rolling form for every entry, using only races before the entry's race.

    ``entries`` needs ``season, round, driver_id, team_id, circuit_id, race_seq``; ``results``
    needs the same ids plus ``position``, ``position_text`` and ``status_class``.

    - ``drv_avg_finish_5``: mean finishing position over the driver's last 5 finishes
      (retirements are skipped, so the window reaches further back).
    - ``team_pts_3``: team points per race over the last 3 races, fixed points table.
    - ``drv_dnf_rate`` / ``team_dnf_rate``: DNF share over the driver's last 10 starts / the
      team's last 10 races, shrunk toward the overall DNF rate so that one retirement in a
      rookie's first race doesn't read as "100% DNF".
    - ``circuit_dnf_rate``: DNF share at this circuit in earlier races, shrunk the same way
      (used by the DNF model only).
    """
    history = results.copy()
    history["points"] = fixed_points(history)
    history["dnf"] = history["status_class"].eq("dnf").fillna(False).astype("float64")
    history["start"] = history["status_class"].isin(["finished", "dnf"]).astype("float64")
    history = history.sort_values(["race_seq", "driver_id"], kind="stable")
    out = entries[[*KEYS, "team_id", "circuit_id", "race_seq"]].copy()

    finishes = history[history["status_class"].eq("finished").fillna(False)].copy()
    finishes["position"] = finishes["position"].astype("float64")
    finishes["drv_avg_finish_5"] = finishes.groupby("driver_id")["position"].transform(
        lambda s: s.rolling(FINISH_WINDOW, min_periods=1).mean()
    )
    out["drv_avg_finish_5"] = strictly_before(out, finishes, ["drv_avg_finish_5"], by="driver_id")[
        "drv_avg_finish_5"
    ]

    team_races = history.groupby(["team_id", "race_seq"], as_index=False)[
        ["points", "dnf", "start"]
    ].sum()
    team_races["team_pts_3"] = team_races.groupby("team_id")["points"].transform(
        lambda s: s.rolling(TEAM_POINTS_WINDOW, min_periods=1).mean()
    )
    team_races["team_dnfs"] = _rolling_sum(team_races, "team_id", "dnf", DNF_WINDOW)
    team_races["team_starts"] = _rolling_sum(team_races, "team_id", "start", DNF_WINDOW)
    team = strictly_before(
        out, team_races, ["team_pts_3", "team_dnfs", "team_starts"], by="team_id"
    )

    starts = history[history["start"] == 1].copy()
    starts["drv_dnfs"] = _rolling_sum(starts, "driver_id", "dnf", DNF_WINDOW)
    starts["drv_starts"] = _rolling_sum(starts, "driver_id", "start", DNF_WINDOW)
    driver = strictly_before(out, starts, ["drv_dnfs", "drv_starts"], by="driver_id")

    circuit_races = history.groupby(["circuit_id", "race_seq"], as_index=False)[
        ["dnf", "start"]
    ].sum()
    circuit_races["circuit_dnfs"] = circuit_races.groupby("circuit_id")["dnf"].cumsum()
    circuit_races["circuit_starts"] = circuit_races.groupby("circuit_id")["start"].cumsum()
    circuit = strictly_before(
        out, circuit_races, ["circuit_dnfs", "circuit_starts"], by="circuit_id"
    )

    overall = history.groupby("race_seq", as_index=False)[["dnf", "start"]].sum()
    overall["prior"] = overall["dnf"].cumsum() / overall["start"].cumsum()
    prior = strictly_before(out, overall, ["prior"])["prior"]

    out["team_pts_3"] = team["team_pts_3"]
    out["drv_dnf_rate"] = _shrunk_rate(
        driver["drv_dnfs"], driver["drv_starts"], prior, DNF_PRIOR_STARTS
    )
    out["team_dnf_rate"] = _shrunk_rate(
        team["team_dnfs"], team["team_starts"], prior, DNF_PRIOR_STARTS
    )
    out["circuit_dnf_rate"] = _shrunk_rate(
        circuit["circuit_dnfs"], circuit["circuit_starts"], prior, CIRCUIT_PRIOR_STARTS
    )
    return out[[*KEYS, *FORM_COLUMNS]]
