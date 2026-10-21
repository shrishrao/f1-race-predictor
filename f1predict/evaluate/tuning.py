"""Choose the ranker's settings and the simulation temperature on 2024, and only there.

Every combination in ``GRID`` is backtested race by race over 2024 exactly like the real
backtest: each race is predicted by a model trained only on the races before it. Each
combination's score is its mean Spearman (how closely the predicted order of the finishers
matches the real one), with a standard error from the spread across races.

``choose_settings`` then applies the one-standard-error rule: of all combinations scoring
within one standard error of the best, take the simplest (fewest leaves in total, then
LightGBM's own defaults). Scores closer than that are noise from which races happened to be
in the season, so simplicity decides.

``validation_temperature`` fits the simulation's temperature on the same season, from
out-of-sample scores: each 2024 race is scored by a ranker trained only on earlier races.
Scores from a ranker that had already seen those races would look too sure of themselves
and give a temperature that's too low.

The DNF model is chosen the same way, in two steps: ``validation_dnf_window`` picks how many
recent races the field-wide DNF rate looks back over (the base rate's lowest mean Brier
score on 2024), and ``choose_dnf_kind`` picks the model (base rate, logistic regression or
LightGBM) with the lowest mean Brier score on 2024.

The 2025 and 2026 test seasons are never touched, so their results stay an honest measure
of races the settings weren't chosen on.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable
from functools import partial
from typing import Any

import numpy as np
import pandas as pd

from f1predict import config
from f1predict.evaluate.backtest import predict_races
from f1predict.evaluate.metrics import dnf_race_metrics, race_metrics
from f1predict.models.dnf import WINDOWS, BaseRateDNF
from f1predict.models.ranker import RankerModel
from f1predict.models.simulate import finishing_orders, fit_temperature

GRID: dict[str, list[Any]] = {
    "n_estimators": [50, 100, 200],
    "num_leaves": [3, 7, 15],
    "min_child_samples": [20, 50],
    "gain": ["exponential", "linear"],
}
LIGHTGBM_DEFAULTS = {"min_child_samples": 20, "gain": "exponential"}
SELECTION_METRIC = "spearman"
REPORTED_METRICS = ["spearman", "mae", "winner_correct", "podium_hit_rate"]

# ---------------------------------------------------------------------------
# Grid search
# ---------------------------------------------------------------------------


def _refuse_test_season(season: int) -> None:
    """Choosing anything on a test season would leak the results we report for it."""
    if season in config.TEST_SEASONS:
        raise ValueError(f"{season} is a test season; tune on the validation season instead.")


def grid_settings(grid: dict[str, list[Any]] | None = None) -> list[dict[str, Any]]:
    """Every combination of the grid's values, as keyword arguments for ``RankerModel``."""
    grid = grid or GRID
    return [dict(zip(grid, values, strict=True)) for values in itertools.product(*grid.values())]


def tune_ranker(
    features: pd.DataFrame,
    feature_set: str = "full",
    season: int = config.VALIDATION_SEASON,
    grid: dict[str, list[Any]] | None = None,
    progress: Callable[[str], None] | None = None,
) -> pd.DataFrame:
    """Backtest every setting combination on ``season``; return them best first.

    One row per combination: its settings, the season's mean of each reported metric, and
    ``spearman_se``, the standard error of the mean Spearman across the season's races.
    Refuses the test seasons, because choosing settings on them would leak their results.
    """
    _refuse_test_season(season)
    rows = []
    for settings in grid_settings(grid):
        make_model = partial(RankerModel, feature_set=feature_set, **settings)
        races = race_metrics(predict_races(make_model, features, [season], progress))
        spread = races[SELECTION_METRIC].std(ddof=1) / np.sqrt(len(races))
        rows.append({**settings, **races[REPORTED_METRICS].mean().to_dict(), "spearman_se": spread})
    table = pd.DataFrame(rows)
    return table.sort_values(SELECTION_METRIC, ascending=False, kind="stable", ignore_index=True)


# ---------------------------------------------------------------------------
# Choosing
# ---------------------------------------------------------------------------


def choose_settings(table: pd.DataFrame) -> dict[str, Any]:
    """The one-standard-error rule: the simplest settings within one SE of the best score.

    Simplest means the fewest leaves in total (trees x leaves per tree), then the fewest
    departures from LightGBM's defaults, then the higher score.
    """
    best = table.loc[table[SELECTION_METRIC].idxmax()]
    near = table[table[SELECTION_METRIC] >= best[SELECTION_METRIC] - best["spearman_se"]]
    ranked = near.assign(
        total_leaves=near["n_estimators"] * near["num_leaves"],
        changed=sum((near[key] != value).astype(int) for key, value in LIGHTGBM_DEFAULTS.items()),
    ).sort_values(["total_leaves", "changed", SELECTION_METRIC], ascending=[True, True, False])
    return {key: ranked.iloc[0][key] for key in GRID}


# ---------------------------------------------------------------------------
# Temperature
# ---------------------------------------------------------------------------


def validation_temperature(
    features: pd.DataFrame,
    feature_set: str = "full",
    season: int = config.VALIDATION_SEASON,
    progress: Callable[[str], None] | None = None,
) -> float:
    """Fit the simulation temperature on ``season``'s out-of-sample ranker scores."""
    _refuse_test_season(season)
    make_model = partial(RankerModel, feature_set=feature_set)
    predictions = predict_races(make_model, features, [season], progress)
    return fit_temperature(finishing_orders(predictions))


# ---------------------------------------------------------------------------
# The DNF model
# ---------------------------------------------------------------------------


def validation_dnf_window(
    features: pd.DataFrame,
    season: int = config.VALIDATION_SEASON,
    windows: tuple[int | None, ...] = WINDOWS,
    progress: Callable[[str], None] | None = None,
) -> tuple[int | None, pd.DataFrame]:
    """Choose how many recent races the field-wide DNF rate looks back over.

    Backtests the base rate with every window in ``windows`` (None: all earlier races) on
    ``season`` and returns the one with the lowest mean Brier score, plus a table of every
    window's mean Brier score and log loss.
    """
    _refuse_test_season(season)
    rows = []
    for window in windows:
        make_model = partial(BaseRateDNF, window=window)
        races = dnf_race_metrics(predict_races(make_model, features, [season], progress))
        rows.append(races[["brier_dnf", "log_loss_dnf"]].mean().to_dict())
    table = pd.DataFrame(rows)
    table.insert(0, "window", pd.Series(windows, dtype="object"))
    return windows[int(np.argmin(table["brier_dnf"]))], table


def choose_dnf_kind(summary: pd.DataFrame, season: int = config.VALIDATION_SEASON) -> str:
    """The DNF model with the lowest mean Brier score on ``season``.

    ``summary`` is a DNF comparison's season summary (a ``model`` column, one row per
    model, season and metric).
    """
    _refuse_test_season(season)
    rows = summary[(summary["season"] == season) & (summary["metric"] == "brier_dnf")]
    if rows.empty:
        raise ValueError(f"No {season} DNF results to choose a model on.")
    return str(rows.loc[rows["value"].idxmin(), "model"])
