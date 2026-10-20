"""Tests for the rolling-origin backtest and the ``backtest`` command.

The key tests use a spy model that records what the backtest shows it. For every race the
spy must have been trained on exactly the earlier races, and it must never see the race's
outcome columns. If the backtest leaked, a model could score well just by reading the
answer.
"""

from __future__ import annotations

import json
from dataclasses import replace

import pandas as pd
import pytest
from synthetic import make_tables
from typer.testing import CliRunner

from f1predict import config
from f1predict.__main__ import app
from f1predict.evaluate.backtest import (
    PredictionError,
    check_prediction,
    predict_races,
    run_backtest,
    save_backtest,
)
from f1predict.features.build import TARGET_COLUMNS, build_features, load_features, save_features
from f1predict.models.baseline import GridBaseline

KEYS = ["season", "round", "driver_id"]
runner = CliRunner()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class SpyModel:
    """Predicts the grid order and records what the backtest showed it."""

    def __init__(self, log: list[dict]) -> None:
        self.log = log
        self.trained_on: set[int] = set()

    def fit(self, history: pd.DataFrame) -> SpyModel:
        """Remember which races were in the training data."""
        self.trained_on = set(history["race_seq"])
        return self

    def predict(self, race: pd.DataFrame) -> pd.DataFrame:
        """Log the race and the columns it came with, then predict the grid order."""
        self.log.append(
            {
                "race_seq": int(race["race_seq"].iloc[0]),
                "trained_on": self.trained_on,
                "columns": set(race.columns),
            }
        )
        out = race[KEYS].copy()
        out["pred_rank"] = race["grid"].rank(method="first").astype("int64")
        return out


@pytest.fixture(scope="module")
def features() -> pd.DataFrame:
    """Features of the synthetic championship: 2023 and 2024, five races each."""
    return build_features(make_tables())


@pytest.fixture(scope="module")
def real_features() -> pd.DataFrame:
    """The real features table, or skip (it isn't in the repo or in CI)."""
    try:
        return load_features()
    except FileNotFoundError:
        pytest.skip("features not built; run `python -m f1predict build-features` to include this")


# ---------------------------------------------------------------------------
# No peeking
# ---------------------------------------------------------------------------


def test_each_race_is_predicted_from_exactly_the_races_before_it(features) -> None:
    """Race N's model was trained on races 0..N-1: never N itself, never later."""
    log: list[dict] = []
    predict_races(lambda: SpyModel(log), features, [2024])
    assert [call["race_seq"] for call in log] == [5, 6, 7, 8, 9]
    for call in log:
        assert call["trained_on"] == set(range(call["race_seq"]))


def test_the_model_never_sees_the_outcome_of_the_race_it_predicts(features) -> None:
    """Position, status, finishing rank and the number of finishers are removed first."""
    log: list[dict] = []
    predict_races(lambda: SpyModel(log), features, [2024])
    for call in log:
        assert not call["columns"] & set(TARGET_COLUMNS)
        assert {"grid", "quali_gap_pole_pct", "drv_avg_finish_5"} <= call["columns"]


def test_the_same_holds_on_the_real_races(real_features) -> None:
    """The spy checks again on every real race of 2024-2026."""
    log: list[dict] = []
    predict_races(lambda: SpyModel(log), real_features, [2024, 2025, 2026])
    assert len(log) >= 65
    for call in log:
        assert call["trained_on"] == set(range(call["race_seq"]))
        assert not call["columns"] & set(TARGET_COLUMNS)


# ---------------------------------------------------------------------------
# Which races are backtested
# ---------------------------------------------------------------------------


def test_a_race_without_a_result_is_not_backtested() -> None:
    """With 2024 round 5 not run yet, only rounds 1-4 of 2024 are predicted."""
    tables = make_tables()
    results = tables.results
    not_run = (results["season"] == 2024) & (results["round"] == 5)
    upcoming = build_features(replace(tables, results=results[~not_run]))
    predicted = predict_races(GridBaseline, upcoming, [2024])
    assert sorted(predicted["round"].unique()) == [1, 2, 3, 4]


def test_the_very_first_race_is_skipped(features, caplog) -> None:
    """The first race of the data has nothing to learn from."""
    predicted = predict_races(GridBaseline, features, [2023])
    assert sorted(predicted["round"].unique()) == [2, 3, 4, 5]
    assert "no earlier race" in caplog.text


def test_a_prediction_that_cant_be_scored_is_rejected(features) -> None:
    """A missing driver, a repeated rank, or a chance above 100% stops the backtest."""
    race = features[(features["season"] == 2024) & (features["round"] == 1)]
    good = race[KEYS].assign(pred_rank=range(1, len(race) + 1))
    check_prediction(good, race, "2024 round 1")
    with pytest.raises(PredictionError, match="one row per entrant"):
        check_prediction(good.iloc[1:], race, "2024 round 1")
    with pytest.raises(PredictionError, match="pred_rank"):
        check_prediction(good.assign(pred_rank=1), race, "2024 round 1")
    with pytest.raises(PredictionError, match="p_win"):
        check_prediction(good.assign(p_win=1.5), race, "2024 round 1")


# ---------------------------------------------------------------------------
# The grid baseline end to end
# ---------------------------------------------------------------------------


def test_baseline_backtest_end_to_end(features) -> None:
    """Five races scored; chances add up per race; every driver lands in a calibration bin."""
    result = run_backtest(GridBaseline, features, [2024])
    assert len(result.races) == 5
    sums = result.predictions.groupby("round")[["p_win", "p_podium", "p_points"]].sum()
    assert sums["p_win"].to_numpy() == pytest.approx([1] * 5)
    assert sums["p_podium"].to_numpy() == pytest.approx([3] * 5)
    assert sums["p_points"].to_numpy() == pytest.approx([6] * 5)
    summary = result.summary
    assert ((summary["low"] <= summary["value"]) & (summary["value"] <= summary["high"])).all()
    assert result.calibration.groupby("outcome")["n"].sum().tolist() == [30, 30, 30]


def test_save_backtest_writes_four_files(features, tmp_path) -> None:
    """Predictions, per-race metrics, calibration bins and the season summary."""
    folder = save_backtest(run_backtest(GridBaseline, features, [2024]), "baseline", tmp_path)
    assert {path.name for path in folder.iterdir()} == {
        "predictions.parquet",
        "races.csv",
        "calibration.csv",
        "summary.json",
    }
    summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    assert summary["seasons"]["2024"]["races"] == 5
    assert {"winner_correct", "brier_win"} <= set(summary["seasons"]["2024"]["metrics"])
    assert len(pd.read_parquet(folder / "predictions.parquet")) == 30


# ---------------------------------------------------------------------------
# Command
# ---------------------------------------------------------------------------


@pytest.fixture
def processed(features, tmp_path, monkeypatch):
    """Save the synthetic features where the command looks for them."""
    monkeypatch.setattr(config, "PROCESSED_DIR", tmp_path / "processed")
    monkeypatch.setattr(config, "LOG_DIR", tmp_path / "logs")
    save_features(features)
    return tmp_path / "processed"


def test_backtest_command_prints_the_table_and_saves_results(processed) -> None:
    """``backtest --season 2024`` scores the five synthetic 2024 races."""
    result = runner.invoke(app, ["backtest", "--season", "2024"])
    assert result.exit_code == 0, result.output
    assert "Backtest: baseline" in result.output
    assert "Winner accuracy" in result.output
    assert (processed / "backtest" / "baseline" / "summary.json").exists()


def test_backtest_command_with_no_completed_races(processed) -> None:
    """The synthetic data has no 2025 races, so there is nothing to score."""
    result = runner.invoke(app, ["backtest", "--season", "2025"])
    assert result.exit_code == 1
    assert "No completed races" in result.output


def test_backtest_command_without_features_explains(tmp_path, monkeypatch) -> None:
    """Without features.parquet the command says which command builds it."""
    monkeypatch.setattr(config, "PROCESSED_DIR", tmp_path / "processed")
    monkeypatch.setattr(config, "LOG_DIR", tmp_path / "logs")
    result = runner.invoke(app, ["backtest"])
    assert result.exit_code == 1
    assert "build-features" in result.output


def test_backtest_command_rejects_unknown_models_and_seasons() -> None:
    """Typos are usage errors (exit code 2), caught before any work starts."""
    assert runner.invoke(app, ["backtest", "--model", "crystal_ball"]).exit_code == 2
    assert runner.invoke(app, ["backtest", "--season", "2018"]).exit_code == 2
