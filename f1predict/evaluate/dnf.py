"""Backtest the three DNF models side by side.

Each DNF model predicts every race from the races before it, exactly like the main
backtest, and is scored on the cars that started:

- Brier score and log loss per race, averaged over the season, with 95% bootstrap CIs;
- the same two as differences from the base rate, with paired CIs: do the per-car inputs
  add anything beyond knowing how often cars retire these days?
- ROC AUC pooled over each season: how well the chances sort retirees above finishers.

Which model the simulation uses, and its look-back window, are chosen on the 2024
validation season (``evaluate/tuning.py``); the test seasons only show how that choice held
up.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import pandas as pd

from f1predict.evaluate.backtest import backtest_dir, predict_races
from f1predict.evaluate.metrics import compare, dnf_race_metrics, season_auc, summarize
from f1predict.models.dnf import DNF_KINDS, make_dnf_model

REFERENCE = "base_rate"


@dataclass(frozen=True)
class DNFComparison:
    """Everything one DNF comparison produces, with a ``model`` column in every table.

    ``predictions``: one row per driver per race per model; ``races``: Brier and log loss
    per race; ``summary``: season averages and pooled ROC AUC with 95% CIs; ``differences``:
    each learned model minus the base rate, with paired CIs.
    """

    predictions: pd.DataFrame
    races: pd.DataFrame
    summary: pd.DataFrame
    differences: pd.DataFrame


# ---------------------------------------------------------------------------
# Running and scoring
# ---------------------------------------------------------------------------


def compare_dnf_models(
    features: pd.DataFrame,
    seasons: Iterable[int],
    window: int | None,
    progress: Callable[[str], None] | None = None,
    kinds: Iterable[str] = DNF_KINDS,
    inputs: list[str] | None = None,
) -> DNFComparison:
    """Backtest every DNF model in ``kinds`` over ``seasons`` and score them side by side.

    ``window`` is the look-back for the field-wide DNF rate, shared by every model;
    ``inputs`` replaces the learned models' input columns (e.g. for an ablation).
    ``progress`` is called after each race of each model.
    """
    seasons = sorted(seasons)
    runs = {
        kind: predict_races(
            partial(make_dnf_model, kind, window, inputs), features, seasons, progress
        )
        for kind in kinds
    }
    return score_dnf_runs(runs)


def score_dnf_runs(runs: dict[str, pd.DataFrame]) -> DNFComparison:
    """Score each model's backtest predictions and compare each with the base rate."""
    races = {kind: dnf_race_metrics(predictions) for kind, predictions in runs.items()}
    summary = pd.concat(
        [
            pd.concat([summarize(races[kind]), season_auc(predictions)]).assign(model=kind)
            for kind, predictions in runs.items()
        ],
        ignore_index=True,
    )
    differences = [
        compare(races[kind], races[REFERENCE]).assign(model=kind)
        for kind in runs
        if kind != REFERENCE and REFERENCE in races
    ]
    return DNFComparison(
        predictions=pd.concat(
            [predictions.assign(model=kind) for kind, predictions in runs.items()],
            ignore_index=True,
        ),
        races=pd.concat(
            [table.assign(model=kind) for kind, table in races.items()], ignore_index=True
        ),
        summary=summary,
        differences=pd.concat(differences, ignore_index=True) if differences else pd.DataFrame(),
    )


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------


def save_dnf_comparison(comparison: DNFComparison, name: str, folder: Path | None = None) -> Path:
    """Save a DNF comparison under ``backtest/<name>`` (e.g. ``dnf_full``); return the folder.

    ``predictions.parquet`` (every model's ``p_dnf`` per driver per race), ``races.csv``,
    ``summary.csv`` (season averages and ROC AUC with CIs) and ``vs_base_rate.csv``.
    """
    target = folder or backtest_dir(name)
    target.mkdir(parents=True, exist_ok=True)
    comparison.predictions.to_parquet(target / "predictions.parquet", index=False)
    comparison.races.to_csv(target / "races.csv", index=False, float_format="%.6g")
    comparison.summary.to_csv(target / "summary.csv", index=False, float_format="%.6g")
    comparison.differences.to_csv(target / "vs_base_rate.csv", index=False, float_format="%.6g")
    return target
