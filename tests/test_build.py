"""Tests for the features table: entries, targets, exclusions, and the leakage test.

The leakage test is the most important test in the project. For a sample of races it
builds features twice: once from all the data, and once from ``pre_race_view``, which only
holds what was known on the eve of that race (later races removed, the race's own result
blanked). If any feature used information from the race itself or from later races, the
two builds would disagree for that race.
"""

from __future__ import annotations

import pandas as pd
import pytest
from synthetic import make_tables, scramble_after
from typer.testing import CliRunner

from f1predict import config
from f1predict.__main__ import app
from f1predict.data.tables import save_season_tables
from f1predict.features.build import (
    FEATURE_COLUMNS,
    FEATURE_GROUPS,
    FEATURE_SETS,
    RaceTables,
    build_features,
    feature_columns,
    load_race_tables,
    pre_race_view,
)

SAMPLED_RACES = [
    (2023, 1),
    (2023, 3),
    (2023, 4),
    (2024, 1),
    (2024, 2),
    (2024, 5),
]
REAL_RACES = [(2019, 5), (2021, 10), (2023, 1), (2024, 6), (2025, 24), (2026, 17)]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tables() -> RaceTables:
    """The synthetic championship (2 seasons x 5 rounds)."""
    return make_tables()


@pytest.fixture(scope="module")
def full(tables) -> pd.DataFrame:
    """Features built from all the synthetic data."""
    return build_features(tables)


def race_rows(features: pd.DataFrame, season: int, round_number: int) -> pd.DataFrame:
    """One race's rows, in a stable order, for comparing two builds."""
    rows = features[(features["season"] == season) & (features["round"] == round_number)]
    return rows.sort_values("driver_id").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Leakage
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("season", "round_number"), SAMPLED_RACES)
def test_features_only_use_what_was_known_before_the_race(
    tables, full, season, round_number
) -> None:
    """Features built on the eve of race N equal the features in the full build."""
    eve = build_features(pre_race_view(tables, season, round_number))
    pd.testing.assert_frame_equal(
        race_rows(eve, season, round_number)[FEATURE_COLUMNS],
        race_rows(full, season, round_number)[FEATURE_COLUMNS],
    )


@pytest.mark.parametrize(("season", "round_number"), SAMPLED_RACES)
def test_reshuffling_later_races_changes_nothing(tables, full, season, round_number) -> None:
    """Scrambling every later result leaves race N's features exactly as they were."""
    scrambled = build_features(scramble_after(tables, season, round_number))
    pd.testing.assert_frame_equal(
        race_rows(scrambled, season, round_number)[FEATURE_COLUMNS],
        race_rows(full, season, round_number)[FEATURE_COLUMNS],
    )


def test_the_eve_view_really_hides_the_outcome(tables, full) -> None:
    """Guard on the test itself: the eve view has no result for the race and no later races."""
    eve = build_features(pre_race_view(tables, 2023, 4))
    assert race_rows(eve, 2023, 4)["position"].isna().all()
    assert race_rows(full, 2023, 4)["position"].notna().all()
    assert eve["race_seq"].max() == race_rows(full, 2023, 4)["race_seq"].iloc[0]


@pytest.fixture(scope="module")
def real_tables() -> RaceTables:
    """The real downloaded data, or skip (it isn't in the repo or in CI)."""
    try:
        return load_race_tables()
    except FileNotFoundError:
        pytest.skip("real data not downloaded; run `python -m f1predict fetch` to include this")


@pytest.mark.parametrize(("season", "round_number"), REAL_RACES)
def test_leakage_on_real_races(real_tables, season, round_number) -> None:
    """The same check on real races, including a 2021 sprint weekend and a 2026 race."""
    full = build_features(real_tables)
    eve = build_features(pre_race_view(real_tables, season, round_number))
    pd.testing.assert_frame_equal(
        race_rows(eve, season, round_number)[FEATURE_COLUMNS],
        race_rows(full, season, round_number)[FEATURE_COLUMNS],
    )


# ---------------------------------------------------------------------------
# Rows and targets
# ---------------------------------------------------------------------------


def test_one_row_per_entrant_and_stand_ins_get_none(full) -> None:
    """6 drivers x 10 races; the FP1 stand-in (car 99) never gets a row."""
    assert len(full) == 60
    assert not full.duplicated(["season", "round", "driver_id"]).any()
    assert set(full["driver_id"]) == {"hamilton", "russell", "leclerc", "sainz", "bottas", "zhou"}


def test_finish_rank_counts_finishers_only(full) -> None:
    """A retiree has no finishing rank; finishers are ranked 1..n with no gaps."""
    race = race_rows(full, 2023, 1)
    finishers = race[race["finished"].astype(bool)]
    assert sorted(finishers["finish_rank"]) == list(range(1, len(finishers) + 1))
    assert race.loc[~race["finished"].astype(bool), "finish_rank"].isna().all()
    assert (race["n_finishers"] == len(finishers)).all()


def test_an_upcoming_race_uses_qualifying_as_its_entry_list(tables) -> None:
    """With no results yet, the entry list and grid come from qualifying; targets are empty."""
    results = tables.results[~((tables.results["season"] == 2024) & (tables.results["round"] == 5))]
    features = build_features(
        RaceTables(tables.schedule, results, tables.qualifying, tables.sprint, tables.practice)
    )
    upcoming = race_rows(features, 2024, 5)
    quali = tables.qualifying[
        (tables.qualifying["season"] == 2024) & (tables.qualifying["round"] == 5)
    ]
    assert len(upcoming) == 6
    assert sorted(upcoming["grid"]) == sorted(quali["position"].astype(float))
    assert upcoming["position"].isna().all()


def test_excluded_races_are_dropped_everywhere(tables, monkeypatch) -> None:
    """An excluded race has no rows and leaves no gap in the race order."""
    monkeypatch.setattr(config, "EXCLUDED_RACES", {(2023, 2): "test"})
    features = build_features(tables)
    assert race_rows(features, 2023, 2).empty
    assert sorted(features["race_seq"].unique()) == list(range(9))


def test_team_form_carries_over_a_rebrand(full) -> None:
    """Alfa Romeo became Sauber in 2024; its first 2024 race still has recent team form."""
    sauber = race_rows(full, 2024, 1).query("constructor_id == 'sauber'")
    assert (sauber["team_id"] == "audi").all()
    assert sauber["team_pts_3"].notna().all()


def test_pit_lane_start_and_sprint_weekend_features(full) -> None:
    """The synthetic pit-lane starter is flagged; only the sprint weekend has sprint_pos."""
    assert race_rows(full, 2023, 4)["pitlane_start"].sum() == 1
    assert race_rows(full, 2023, 3)["sprint_pos"].notna().all()
    assert race_rows(full, 2023, 1)["sprint_pos"].isna().all()


# ---------------------------------------------------------------------------
# Feature sets for the ablations
# ---------------------------------------------------------------------------


def test_ablation_feature_sets(full) -> None:
    """Each ablation drops exactly its group, and every listed column exists."""
    assert set(feature_columns("full")) == {c for g in FEATURE_GROUPS.values() for c in g}
    assert not any(c.startswith("fp") or "longrun" in c for c in feature_columns("no_practice"))
    assert not any("longrun" in c for c in feature_columns("simple_practice"))
    assert "grid" not in feature_columns("no_grid_quali")
    for name in FEATURE_SETS:
        assert set(feature_columns(name)) <= set(full.columns)


# ---------------------------------------------------------------------------
# Command
# ---------------------------------------------------------------------------


def test_build_features_command_writes_the_table(tables, tmp_path, monkeypatch) -> None:
    """``build-features`` reads the interim tables and writes features.parquet."""
    monkeypatch.setattr(config, "INTERIM_DIR", tmp_path / "interim")
    monkeypatch.setattr(config, "PROCESSED_DIR", tmp_path / "processed")
    monkeypatch.setattr(config, "LOG_DIR", tmp_path / "logs")
    for season in (2023, 2024):
        save_season_tables(
            {name: frame[frame["season"] == season] for name, frame in vars(tables).items()},
            season,
            tmp_path / "interim",
        )
    result = CliRunner().invoke(app, ["build-features"])
    assert result.exit_code == 0, result.output
    saved = pd.read_parquet(tmp_path / "processed" / "features.parquet")
    assert len(saved) == 60
    assert "Saved 60 rows" in result.output
