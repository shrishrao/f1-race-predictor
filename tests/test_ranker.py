"""Tests for the LightGBM ranker, the settings search, and the ``tune``/``train`` commands.

Most tests use a made-up championship with one obvious signal: the finishing order follows
the qualifying gap exactly, the grid is shuffled noise, and one car per race retires. A
ranker that can't learn that is broken.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from synthetic import feature_rows, learnable_features, make_tables
from typer.testing import CliRunner

from f1predict import config
from f1predict.__main__ import app
from f1predict.evaluate import tuning
from f1predict.evaluate.backtest import predict_races
from f1predict.evaluate.metrics import race_metrics
from f1predict.features.build import FEATURE_SETS, TARGET_COLUMNS, build_features, save_features
from f1predict.models import FEATURE_SET_NAMES, make_model, run_name
from f1predict.models.ranker import LINEAR_GAIN, RankerModel, race_sizes, relevance, training_rows

runner = CliRunner()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    """The learnable championship: 2020 and 2021, 20 races each."""
    return learnable_features()


@pytest.fixture(scope="module")
def trained(frame) -> RankerModel:
    """A ranker trained on the 2020 races of the learnable championship."""
    return RankerModel().fit(frame[frame["season"] == 2020])


# ---------------------------------------------------------------------------
# Training data
# ---------------------------------------------------------------------------


def test_training_rows_labels_and_groups() -> None:
    """Two races: 3 finishers and a retiree, then 2 finishers. The retiree isn't trained on."""
    history = pd.DataFrame(
        {
            "race_seq": [0, 0, 0, 0, 1, 1],
            "driver_id": ["a", "b", "c", "x", "a", "b"],
            "finished": [True, True, True, False, True, True],
            "finish_rank": [2.0, 1.0, 3.0, np.nan, 1.0, 2.0],
            "n_finishers": [3.0, 3.0, 3.0, 3.0, 2.0, 2.0],
        }
    )
    train = training_rows(history)
    assert train["driver_id"].tolist() == ["b", "a", "c", "a", "b"]
    assert relevance(train).tolist() == [2, 1, 0, 1, 0]
    assert race_sizes(train).tolist() == [3, 2]


def test_gain_settings_reach_lightgbm() -> None:
    """Linear gain is passed as LightGBM's label_gain; exponential is LightGBM's default."""
    assert RankerModel(gain="linear").lgbm_params()["label_gain"] == LINEAR_GAIN
    assert "label_gain" not in RankerModel(gain="exponential").lgbm_params()
    with pytest.raises(ValueError, match="gain"):
        RankerModel(gain="cubic").lgbm_params()


# ---------------------------------------------------------------------------
# Learning and predicting
# ---------------------------------------------------------------------------


def test_the_ranker_learns_an_obvious_signal(frame) -> None:
    """Backtested over 2021, the predicted order of the finishers is almost exactly right."""
    races = race_metrics(predict_races(RankerModel, frame, [2021]))
    assert len(races) == 20
    assert races["spearman"].mean() > 0.9


def test_every_car_gets_a_score_and_the_score_is_kept(trained, frame) -> None:
    """The retiree is scored too (nobody knows on Saturday), and ranks run 1..n."""
    predicted = trained.predict(feature_rows(frame, 30))
    assert len(predicted) == 10
    assert sorted(predicted["pred_rank"]) == list(range(1, 11))
    assert predicted["score"].notna().all()
    best = predicted.loc[predicted["score"].idxmax()]
    assert best["pred_rank"] == 1


def test_the_same_seed_gives_the_same_scores(frame, trained) -> None:
    """Training twice on the same races gives identical scores, to the last bit."""
    again = RankerModel().fit(frame[frame["season"] == 2020])
    race = feature_rows(frame, 25)
    assert np.array_equal(trained.predict(race)["score"], again.predict(race)["score"])


class FlatBooster:
    """Stands in for trained trees that give every car the same score."""

    def predict(self, rows: pd.DataFrame, validate_features: bool = True) -> np.ndarray:
        """A score of 0 for every row."""
        return np.zeros(len(rows))


@pytest.mark.parametrize(
    ("feature_set", "expected"), [("full", [3, 1, 2]), ("no_grid_quali", [1, 2, 3])]
)
def test_tied_scores_use_the_grid_only_if_the_model_may_see_it(
    frame, feature_set, expected
) -> None:
    """Ties go to the better grid slot, unless the grid is hidden (then by driver id)."""
    model = RankerModel(feature_set=feature_set)
    model.booster = FlatBooster()
    race = feature_rows(frame, 30).head(3).assign(grid=[3.0, 1.0, 2.0])
    race = race.assign(driver_id=["d1", "d2", "d3"])
    predicted = model.predict(race).set_index("driver_id")
    assert predicted.loc[["d1", "d2", "d3"], "pred_rank"].tolist() == expected


def test_importance_finds_the_signal(trained) -> None:
    """The qualifying gap carries all the signal, so it gets most of the gain."""
    importance = trained.importance()
    assert importance.iloc[0]["feature"] == "quali_gap_pole_pct"
    assert importance["gain_share"].sum() == pytest.approx(1)


def test_misuse_fails_loudly(trained, frame) -> None:
    """Predicting or asking for importance before training; predicting two races at once."""
    with pytest.raises(RuntimeError, match="fit"):
        RankerModel().predict(feature_rows(frame, 30))
    with pytest.raises(RuntimeError, match="fit"):
        RankerModel().importance()
    with pytest.raises(ValueError, match="exactly one race"):
        trained.predict(pd.concat([feature_rows(frame, 30), feature_rows(frame, 31)]))


def test_no_feature_set_includes_an_outcome_column() -> None:
    """A feature like ``position`` would hand the ranker the answer it's meant to predict."""
    for name in FEATURE_SETS:
        assert not set(RankerModel(feature_set=name).features) & set(TARGET_COLUMNS)


def test_models_and_feature_sets_by_name() -> None:
    """The command-line choices match the feature sets, and names map to fresh models."""
    assert set(FEATURE_SET_NAMES) == set(FEATURE_SETS)
    model = make_model("ranker", "no_practice")
    assert isinstance(model, RankerModel)
    assert model.feature_set == "no_practice"
    assert run_name("ranker", "no_practice") == "ranker_no_practice"
    assert run_name("baseline", "no_practice") == "baseline"


# ---------------------------------------------------------------------------
# Choosing settings
# ---------------------------------------------------------------------------


def test_the_grid_covers_every_combination() -> None:
    """3 x 3 x 2 x 2 = 36 combinations, each a complete set of ranker settings."""
    combinations = tuning.grid_settings()
    assert len(combinations) == 36
    assert all(set(c) == set(RankerModel().settings()) for c in combinations)


def test_tuning_ranks_settings_best_first(frame) -> None:
    """A tiny grid on 2021: one row per combination, best mean Spearman first."""
    grid = {"n_estimators": [5, 60], "num_leaves": [3], "min_child_samples": [5]}
    table = tuning.tune_ranker(frame, season=2021, grid={**grid, "gain": ["linear"]})
    assert len(table) == 2
    assert table["spearman"].is_monotonic_decreasing
    assert set(tuning.REPORTED_METRICS) <= set(table.columns)


def tuning_row(trees: int, leaves: int, min_rows: int, gain: str, spearman: float) -> dict:
    """One combination's result, with a standard error of 0.02 on its mean Spearman."""
    return {
        "n_estimators": trees,
        "num_leaves": leaves,
        "min_child_samples": min_rows,
        "gain": gain,
        "spearman": spearman,
        "spearman_se": 0.02,
    }


def test_the_simplest_setting_within_one_standard_error_wins() -> None:
    """The big model is best by less than one SE; of the two small ones, the default wins."""
    table = pd.DataFrame(
        [
            tuning_row(200, 15, 20, "exponential", 0.860),
            tuning_row(50, 3, 50, "linear", 0.850),
            tuning_row(50, 3, 20, "exponential", 0.845),
            tuning_row(50, 7, 20, "exponential", 0.800),
        ]
    )
    assert tuning.choose_settings(table) == {
        "n_estimators": 50,
        "num_leaves": 3,
        "min_child_samples": 20,
        "gain": "exponential",
    }


def test_a_simpler_setting_more_than_one_standard_error_behind_loses() -> None:
    """Simplicity only breaks ties; a clearly worse small model isn't chosen."""
    table = pd.DataFrame(
        [tuning_row(100, 7, 20, "exponential", 0.860), tuning_row(50, 3, 20, "exponential", 0.83)]
    )
    assert tuning.choose_settings(table)["num_leaves"] == 7


def test_tuning_refuses_the_test_seasons(frame) -> None:
    """Choosing settings on 2025 or 2026 would leak the results we report."""
    for season in config.TEST_SEASONS:
        with pytest.raises(ValueError, match="test season"):
            tuning.tune_ranker(frame, season=season)


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    """The synthetic championship's features where the commands look for them."""
    monkeypatch.setattr(config, "PROCESSED_DIR", tmp_path / "processed")
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(config, "LOG_DIR", tmp_path / "logs")
    save_features(build_features(make_tables()))
    return tmp_path


def test_backtest_command_compares_the_ranker_with_the_baseline(workspace) -> None:
    """``backtest --model ranker`` prints both tables and saves the comparison."""
    result = runner.invoke(app, ["backtest", "--model", "ranker", "--season", "2024"])
    assert result.exit_code == 0, result.output
    assert "Backtest: ranker_full" in result.output
    assert "ranker_full minus baseline" in result.output
    assert (workspace / "processed" / "backtest" / "ranker_full" / "vs_baseline.csv").exists()


def test_train_command_saves_the_model_and_its_description(workspace) -> None:
    """``train`` fits on every completed race and writes the .txt and .json files."""
    result = runner.invoke(app, ["train"])
    assert result.exit_code == 0, result.output
    assert "Trained ranker" in result.output
    assert (workspace / "models" / "ranker_full.txt").exists()
    assert (workspace / "models" / "ranker_full.json").exists()


def test_tune_command_saves_the_table(workspace, monkeypatch) -> None:
    """``tune`` with a two-combination grid on the synthetic 2024 season."""
    small = {"n_estimators": [5, 10], "num_leaves": [3], "min_child_samples": [5]}
    monkeypatch.setattr(tuning, "GRID", {**small, "gain": ["linear"]})
    result = runner.invoke(app, ["tune"])
    assert result.exit_code == 0, result.output
    assert "best mean Spearman first" in result.output
    saved = pd.read_csv(workspace / "processed" / "tuning" / "ranker_full.csv")
    assert len(saved) == 2
