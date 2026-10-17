"""Build the features table: one row per driver per race, every value known before the race.

The rows are the race's entry list (everyone in its results, or, for a race not run yet,
everyone in its qualifying). The columns are grid and qualifying features, rolling form,
the sprint result, and practice pace, plus target columns (the actual outcome) for races
that have happened. ``pre_race_view`` shows exactly what was known on Saturday night of a
race, and the leakage test checks that building features from it changes nothing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from pathlib import Path

import pandas as pd

from f1predict import config
from f1predict.data.lineage import team_lineage
from f1predict.data.tables import TABLE_NAMES, load_table
from f1predict.features.form import form_features
from f1predict.features.practice import BEST_LAP_COLUMNS, LONG_RUN_COLUMNS, practice_features
from f1predict.features.quali import grid_features, quali_features

logger = logging.getLogger(__name__)

KEYS = ["season", "round", "driver_id"]
ID_COLUMNS = [
    "season",
    "round",
    "race_seq",
    "race_start",
    "circuit_id",
    "driver_id",
    "driver_code",
    "constructor_id",
    "team_id",
]
TARGET_COLUMNS = ["position", "status_class", "finished", "finish_rank", "n_finishers"]
OUTCOME_COLUMNS = [
    "position",
    "position_text",
    "points",
    "laps",
    "status",
    "status_class",
    "time_ms",
]

FEATURE_GROUPS = {
    "grid_quali": ["grid", "pitlane_start", "quali_gap_pole_pct", "quali_gap_teammate_pct"],
    "form": ["drv_avg_finish_5", "team_pts_3", "drv_dnf_rate", "team_dnf_rate"],
    "sprint": ["sprint_pos"],
    "practice_simple": BEST_LAP_COLUMNS,
    "practice_longrun": LONG_RUN_COLUMNS,
}
FEATURE_SETS = {
    "full": ["grid_quali", "form", "sprint", "practice_simple", "practice_longrun"],
    "no_practice": ["grid_quali", "form", "sprint"],
    "simple_practice": ["grid_quali", "form", "sprint", "practice_simple"],
    "no_grid_quali": ["form", "sprint", "practice_simple", "practice_longrun"],
}
DNF_ONLY_COLUMNS = ["circuit_dnf_rate"]
FEATURE_COLUMNS = [c for group in FEATURE_GROUPS.values() for c in group] + DNF_ONLY_COLUMNS

# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RaceTables:
    """The interim tables features are built from."""

    schedule: pd.DataFrame
    results: pd.DataFrame
    qualifying: pd.DataFrame
    sprint: pd.DataFrame
    practice: pd.DataFrame


def load_race_tables(interim_dir: Path | None = None) -> RaceTables:
    """Read the saved interim tables; tables never fetched (no sprints, no practice) are empty."""
    frames = {}
    for name in ("schedule", "results", "qualifying", "sprint", "practice"):
        try:
            frames[name] = load_table(name, interim_dir)
        except FileNotFoundError:
            if name in ("schedule", "results", "qualifying"):
                raise
            frames[name] = pd.DataFrame()
    assert set(frames) <= set(TABLE_NAMES)
    return RaceTables(**frames)


def feature_columns(feature_set: str) -> list[str]:
    """The ranker's input columns for one ablation feature set (e.g. ``"no_practice"``)."""
    return [column for group in FEATURE_SETS[feature_set] for column in FEATURE_GROUPS[group]]


# ---------------------------------------------------------------------------
# Views of the data
# ---------------------------------------------------------------------------


def _race_keys(frame: pd.DataFrame) -> pd.Series:
    """(season, round) pairs of a table, for filtering."""
    return pd.Series(list(zip(frame["season"], frame["round"], strict=True)), index=frame.index)


def drop_excluded(tables: RaceTables) -> RaceTables:
    """Remove the races in ``config.EXCLUDED_RACES`` from every table."""
    excluded = set(config.EXCLUDED_RACES)
    kept = {}
    for name in ("schedule", "results", "qualifying", "sprint", "practice"):
        frame = getattr(tables, name)
        kept[name] = frame if frame.empty else frame[~_race_keys(frame).isin(excluded)]
    return RaceTables(**kept)


def pre_race_view(tables: RaceTables, season: int, round_number: int) -> RaceTables:
    """What was known on the eve of one race: earlier races in full, this race before it starts.

    Later races are removed. For the race itself, qualifying, sprint and practice stay (they
    happen first), and its results rows keep only the entry list and grid: finishing
    position, status, points, laps and time are blanked.
    """
    target = (season, round_number)

    def upto(frame: pd.DataFrame) -> pd.DataFrame:
        """Rows of races up to and including the target race."""
        if frame.empty:
            return frame
        return frame[_race_keys(frame).map(lambda key: key <= target)]

    results = upto(tables.results).copy()
    this_race = _race_keys(results) == target
    for column in OUTCOME_COLUMNS:
        results.loc[this_race, column] = pd.NA
    return replace(
        tables,
        schedule=upto(tables.schedule),
        results=results,
        qualifying=upto(tables.qualifying),
        sprint=upto(tables.sprint),
        practice=upto(tables.practice),
    )


# ---------------------------------------------------------------------------
# Building
# ---------------------------------------------------------------------------


def race_order(schedule: pd.DataFrame) -> pd.DataFrame:
    """Number the races in date order (``race_seq``), with each race's circuit and start."""
    order = schedule.sort_values(["season", "round"])[
        ["season", "round", "circuit_id", "race_start"]
    ]
    return order.assign(race_seq=range(len(order))).reset_index(drop=True)


def entry_list(tables: RaceTables) -> pd.DataFrame:
    """Everyone entered in each race, with their grid slot.

    Races that have been run use their results rows (``grid`` is the actual starting slot,
    after penalties). A race with qualifying but no results yet uses the qualifying order as
    the grid; known penalties can be applied with a grid override when predicting.
    """
    columns = ["season", "round", "driver_id", "driver_code", "constructor_id", "grid"]
    run = tables.results[columns]
    run_keys = set(_race_keys(run))
    upcoming = tables.qualifying[~_race_keys(tables.qualifying).isin(run_keys)]
    upcoming = upcoming.rename(columns={"position": "grid"})[columns]
    return pd.concat([run, upcoming], ignore_index=True)


def targets(results: pd.DataFrame) -> pd.DataFrame:
    """The actual outcome per entry: position, status, and finishing rank among finishers.

    ``finish_rank`` ranks only cars that took the flag (status "finished"), so a classified
    retiree ahead of a lapped finisher doesn't leave a gap in the ranking.
    """
    frame = results[[*KEYS, "position", "status_class"]].copy()
    race = [frame["season"], frame["round"]]
    known = frame["status_class"].notna()
    done = frame["status_class"].eq("finished").fillna(False)
    frame["finished"] = done.astype("boolean").where(known)
    frame["finish_rank"] = (
        frame["position"].astype("float64").where(done).groupby(race).rank(method="first")
    )
    frame["n_finishers"] = done.groupby(race).transform("sum").astype("float64").where(known)
    return frame[[*KEYS, *TARGET_COLUMNS]]


def build_features(tables: RaceTables) -> pd.DataFrame:
    """Build the features table for every entry of every race in ``tables``."""
    tables = drop_excluded(tables)
    order = race_order(tables.schedule)
    entries = entry_list(tables).merge(order, on=["season", "round"], how="inner")
    entries["team_id"] = entries["constructor_id"].map(team_lineage)

    history = tables.results.merge(order, on=["season", "round"], how="inner")
    history["team_id"] = history["constructor_id"].map(team_lineage)

    sprint = tables.sprint
    sprint_pos = (
        sprint[[*KEYS, "position"]].rename(columns={"position": "sprint_pos"})
        if not sprint.empty
        else pd.DataFrame(columns=[*KEYS, "sprint_pos"])
    )

    features = (
        entries[
            [
                *KEYS,
                "driver_code",
                "constructor_id",
                "team_id",
                "circuit_id",
                "race_seq",
                "race_start",
            ]
        ]
        .merge(grid_features(entries), on=KEYS, how="left")
        .merge(quali_features(tables.qualifying), on=KEYS, how="left")
        .merge(form_features(entries, history), on=KEYS, how="left")
        .merge(sprint_pos, on=KEYS, how="left")
        .merge(practice_features(tables.practice), on=KEYS, how="left")
        .merge(targets(tables.results), on=KEYS, how="left")
    )
    features["sprint_pos"] = features["sprint_pos"].astype("float64")
    for column in FEATURE_COLUMNS:
        features[column] = features[column].astype("float64")
    return features[[*ID_COLUMNS, *FEATURE_COLUMNS, *TARGET_COLUMNS]].sort_values(
        ["race_seq", "grid", "driver_id"], ignore_index=True
    )


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------


def features_path() -> Path:
    """Where the features table is saved."""
    return config.PROCESSED_DIR / "features.parquet"


def save_features(features: pd.DataFrame, path: Path | None = None) -> Path:
    """Write the features table to parquet and return the path."""
    target = path or features_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(target, index=False)
    return target


def load_features(path: Path | None = None) -> pd.DataFrame:
    """Read the saved features table, with a clear message if it hasn't been built yet."""
    source = path or features_path()
    if not source.exists():
        raise FileNotFoundError(
            f"No features at {source}. Run `python -m f1predict build-features`."
        )
    return pd.read_parquet(source)
