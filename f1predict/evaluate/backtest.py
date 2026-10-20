"""Rolling-origin backtest: predict each race with a model trained only on the races before it.

For every completed race of the chosen seasons, in date order:

1. Train a fresh model on all earlier races (``race_seq`` lower than this race's).
2. Show it this race's feature rows, with the outcome columns removed.
3. Keep its prediction next to what actually happened.

Each race is a new starting point ("origin") and the training window grows by one race each
time. The features table is already leakage-safe (every row only uses what was known before
its race), so this replays what the model could have predicted on each Saturday night.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from f1predict import config
from f1predict.evaluate.metrics import (
    BOOTSTRAP_SAMPLES,
    CONFIDENCE,
    add_outcomes,
    calibration_table,
    race_metrics,
    summarize,
)
from f1predict.features.build import TARGET_COLUMNS
from f1predict.models import RaceModel

logger = logging.getLogger(__name__)

KEYS = ["season", "round", "driver_id"]
RACE_COLUMNS = [
    "season",
    "round",
    "race_seq",
    "circuit_id",
    "driver_id",
    "driver_code",
    "team_id",
    "grid",
    "pitlane_start",
]
CHANCE_COLUMNS = [f"p_{outcome}" for outcome in config.OUTCOMES]


class PredictionError(ValueError):
    """A model returned a prediction the backtest can't score."""


@dataclass(frozen=True)
class BacktestResult:
    """Everything one backtest run produces.

    ``predictions``: one row per driver per race (prediction next to outcome); ``races``:
    every metric per race; ``summary``: season averages with 95% CIs; ``calibration``:
    chance given vs how often it happened, per season, outcome and chance bin;
    ``comparison``: differences from the grid baseline with paired CIs, when compared.
    """

    predictions: pd.DataFrame
    races: pd.DataFrame
    summary: pd.DataFrame
    calibration: pd.DataFrame
    comparison: pd.DataFrame | None = None


# ---------------------------------------------------------------------------
# Predicting race by race
# ---------------------------------------------------------------------------


def completed_races(features: pd.DataFrame, seasons: Iterable[int]) -> list[int]:
    """``race_seq`` of every race in ``seasons`` that has a result, in date order."""
    by_race = features.groupby("race_seq")
    has_result = by_race["status_class"].agg(lambda s: s.notna().all())
    in_seasons = by_race["season"].first().isin(list(seasons))
    return has_result.index[has_result & in_seasons].tolist()


def check_prediction(predicted: pd.DataFrame, race: pd.DataFrame, label: str) -> None:
    """Make sure a prediction can be scored: every entrant once, ranks 1..n, chances in [0, 1]."""
    expected = sorted(race[KEYS].itertuples(index=False, name=None))
    if sorted(predicted[KEYS].itertuples(index=False, name=None)) != expected:
        raise PredictionError(f"{label}: the prediction needs exactly one row per entrant")
    if sorted(predicted["pred_rank"]) != list(range(1, len(race) + 1)):
        raise PredictionError(f"{label}: pred_rank must use each of 1..{len(race)} once")
    for column in CHANCE_COLUMNS:
        if column in predicted and not predicted[column].between(0, 1).all():
            raise PredictionError(f"{label}: every {column} must be between 0 and 1")


def predict_races(
    make_model: Callable[[], RaceModel],
    features: pd.DataFrame,
    seasons: Iterable[int],
    progress: Callable[[str], None] | None = None,
) -> pd.DataFrame:
    """Predict every completed race of ``seasons``, each by a model trained on earlier races.

    ``make_model`` returns a new, untrained model each time (a model class works). A race
    with no earlier race to learn from is skipped. ``progress``, if given, is called with a
    label like ``"2025 round 3"`` after each race. Returns one row per driver per race: ids,
    grid, the model's prediction columns, and the actual outcome columns.
    """
    seasons = sorted(seasons)
    has_result = features["status_class"].notna()
    predictions = []
    for seq in completed_races(features, seasons):
        race = features[features["race_seq"] == seq]
        history = features[(features["race_seq"] < seq) & has_result]
        label = f"{race['season'].iloc[0]} round {race['round'].iloc[0]}"
        if history.empty:
            logger.warning("Skipping %s: there is no earlier race to learn from", label)
        else:
            model = make_model().fit(history)
            predicted = model.predict(race.drop(columns=TARGET_COLUMNS))
            check_prediction(predicted, race, label)
            new_columns = [c for c in predicted.columns if c in KEYS or c not in race.columns]
            predictions.append(
                race[[*RACE_COLUMNS, *TARGET_COLUMNS]].merge(predicted[new_columns], on=KEYS)
            )
        if progress is not None:
            progress(label)
    if not predictions:
        raise ValueError(f"No completed races to backtest in seasons {seasons}")
    return pd.concat(predictions, ignore_index=True)


def score_predictions(predictions: pd.DataFrame) -> BacktestResult:
    """Score a backtest's predictions: metrics per race, season summaries, calibration."""
    predictions = add_outcomes(predictions)
    races = race_metrics(predictions)
    return BacktestResult(predictions, races, summarize(races), calibration_table(predictions))


def run_backtest(
    make_model: Callable[[], RaceModel],
    features: pd.DataFrame,
    seasons: Iterable[int],
    progress: Callable[[str], None] | None = None,
) -> BacktestResult:
    """The whole backtest for one model: predict every race, then score and summarize."""
    return score_predictions(predict_races(make_model, features, seasons, progress))


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------


def backtest_dir(model_name: str) -> Path:
    """Folder where one model's backtest results are saved."""
    return config.PROCESSED_DIR / "backtest" / model_name


def save_backtest(result: BacktestResult, model_name: str, folder: Path | None = None) -> Path:
    """Save a backtest and return its folder.

    ``predictions.parquet`` (one row per driver per race), ``races.csv`` (metrics per race),
    ``calibration.csv``, ``summary.json`` (season averages with confidence intervals) and,
    for a model compared with the baseline, ``vs_baseline.csv``.
    """
    target = folder or backtest_dir(model_name)
    target.mkdir(parents=True, exist_ok=True)
    result.predictions.to_parquet(target / "predictions.parquet", index=False)
    result.races.to_csv(target / "races.csv", index=False, float_format="%.6g")
    result.calibration.to_csv(target / "calibration.csv", index=False, float_format="%.6g")
    if result.comparison is not None:
        result.comparison.to_csv(target / "vs_baseline.csv", index=False, float_format="%.6g")
    summary = {
        "model": model_name,
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "confidence": CONFIDENCE,
        "seasons": {
            str(season): {
                "races": int(group["races"].iloc[0]),
                "metrics": {
                    row.metric: {"value": row.value, "low": row.low, "high": row.high}
                    for row in group.itertuples()
                },
            }
            for season, group in result.summary.groupby("season")
        },
    }
    (target / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return target
