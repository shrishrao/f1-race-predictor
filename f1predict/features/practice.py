"""Practice pace features: best-lap gap and long-run pace, as percent gaps to the fastest.

Both use only the race weekend's own practice sessions, which finish before the race, so
they are leakage-safe. Laps by FP1 stand-ins (no ``driver_id``) count toward a session's
fastest lap and fastest long run, because those cars were on track, but stand-ins get no
feature rows of their own. Practice data exists from 2023 (OpenF1); earlier races get NaN.
"""

from __future__ import annotations

import pandas as pd

KEYS = ["season", "round", "driver_id"]
SESSIONS = ("FP1", "FP2", "FP3")
SLICKS = frozenset({"SOFT", "MEDIUM", "HARD"})
LONG_RUN_MIN_LAPS = 5
LONG_RUN_SLOWEST = 1.07
BEST_LAP_COLUMNS = [f"{s.lower()}_best_gap_pct" for s in SESSIONS] + ["fp_best_gap_pct"]
LONG_RUN_COLUMNS = [f"{s.lower()}_longrun_gap_pct" for s in SESSIONS] + ["longrun_best_gap_pct"]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _quality_mask(practice: pd.DataFrame) -> pd.Series:
    """Laps fit to compare: a real lap time, not deleted, sectors consistent, slick tyres."""
    return (
        practice["lap_time_s"].notna()
        & ~practice["is_deleted"]
        & practice["is_accurate"]
        & practice["compound"].isin(SLICKS).fillna(False)
        & ~practice["is_in_lap"]
        & ~practice["is_out_lap"]
    )


def _gaps(pace: pd.DataFrame, group: list[str], column: str) -> pd.Series:
    """Each row's ``pace`` as a percent gap to the fastest in its ``group``."""
    fastest = pace.groupby(group)[column].transform("min")
    return (pace[column] / fastest - 1) * 100


def _wide(pace: pd.DataFrame, name: str, pooled: pd.DataFrame, pooled_name: str) -> pd.DataFrame:
    """Turn per-session gaps into FP1/FP2/FP3 columns and add the pooled column."""
    mapped = pace[pace["driver_id"].notna()]
    wide = mapped.pivot_table(
        index=KEYS, columns="session", values="gap", aggfunc="first", observed=True
    )
    wide = wide.reindex(columns=list(SESSIONS))
    wide.columns = [f"{s.lower()}_{name}_gap_pct" for s in SESSIONS]
    pooled_mapped = pooled[pooled["driver_id"].notna()].set_index(KEYS)["gap"].rename(pooled_name)
    return wide.join(pooled_mapped, how="outer").reset_index()


def _driver_ids(practice: pd.DataFrame) -> pd.DataFrame:
    """One row per (race, session, car number) with that car's driver id (empty for stand-ins)."""
    ids = practice[["season", "round", "session", "driver_number", "driver_id"]]
    return ids.drop_duplicates(["season", "round", "session", "driver_number"])


# ---------------------------------------------------------------------------
# Best lap
# ---------------------------------------------------------------------------


def best_lap_features(practice: pd.DataFrame) -> pd.DataFrame:
    """Each driver's best clean lap as a percent gap to the fastest clean lap.

    One column per session (``fp1_best_gap_pct`` ...) plus ``fp_best_gap_pct``, which takes
    the driver's best lap of the weekend against the weekend's fastest lap.
    """
    clean = practice[_quality_mask(practice)]
    by_session = clean.groupby(["season", "round", "session", "driver_number"], as_index=False)[
        "lap_time_s"
    ].min()
    by_session = by_session.merge(_driver_ids(practice), how="left")
    by_session["gap"] = _gaps(by_session, ["season", "round", "session"], "lap_time_s")

    weekend = by_session.groupby(["season", "round", "driver_number"], as_index=False).agg(
        lap_time_s=("lap_time_s", "min"), driver_id=("driver_id", "first")
    )
    weekend["gap"] = _gaps(weekend, ["season", "round"], "lap_time_s")
    return _wide(by_session, "best", weekend, "fp_best_gap_pct")


# ---------------------------------------------------------------------------
# Long runs
# ---------------------------------------------------------------------------


def long_runs(practice: pd.DataFrame) -> pd.DataFrame:
    """Find long runs and return one row per run with its median lap time.

    A long run is one stint (same tyre set, no pit stop in between) that still has at least
    5 laps after removing in- and out-laps, laps that weren't fully green, deleted or
    inaccurate laps, non-slick laps, and laps slower than 107% of the stint's fastest
    remaining lap. That last cut removes cool-down laps and traffic, so what's left is the
    driver's race-pace rhythm.
    """
    usable = practice[_quality_mask(practice) & practice["is_green"] & practice["stint"].notna()]
    stint_keys = ["season", "round", "session", "driver_number", "stint"]
    fastest = usable.groupby(stint_keys)["lap_time_s"].transform("min")
    usable = usable[usable["lap_time_s"] <= fastest * LONG_RUN_SLOWEST]
    runs = usable.groupby(stint_keys, as_index=False).agg(
        laps=("lap_time_s", "size"),
        median_s=("lap_time_s", "median"),
        compound=("compound", "first"),
    )
    return runs[runs["laps"] >= LONG_RUN_MIN_LAPS].reset_index(drop=True)


def long_run_features(practice: pd.DataFrame) -> pd.DataFrame:
    """Each driver's best long-run median as a percent gap to the fastest long-run median.

    One column per session plus ``longrun_best_gap_pct`` for the whole weekend. Fuel load
    and tyre wear are not corrected for (out of scope), so the gap mixes pace with how hard
    each team ran its programme.
    """
    runs = long_runs(practice)
    by_session = runs.groupby(["season", "round", "session", "driver_number"], as_index=False)[
        "median_s"
    ].min()
    by_session = by_session.merge(_driver_ids(practice), how="left")
    by_session["gap"] = _gaps(by_session, ["season", "round", "session"], "median_s")

    weekend = by_session.groupby(["season", "round", "driver_number"], as_index=False).agg(
        median_s=("median_s", "min"), driver_id=("driver_id", "first")
    )
    weekend["gap"] = _gaps(weekend, ["season", "round"], "median_s")
    return _wide(by_session, "longrun", weekend, "longrun_best_gap_pct")


def practice_features(practice: pd.DataFrame) -> pd.DataFrame:
    """Best-lap and long-run features together, one row per driver per race weekend."""
    if practice.empty:
        return pd.DataFrame(columns=[*KEYS, *BEST_LAP_COLUMNS, *LONG_RUN_COLUMNS])
    best = best_lap_features(practice)
    runs = long_run_features(practice)
    merged = best.merge(runs, on=KEYS, how="outer")
    for column in [*BEST_LAP_COLUMNS, *LONG_RUN_COLUMNS]:
        if column not in merged:
            merged[column] = float("nan")
    return merged[[*KEYS, *BEST_LAP_COLUMNS, *LONG_RUN_COLUMNS]]
