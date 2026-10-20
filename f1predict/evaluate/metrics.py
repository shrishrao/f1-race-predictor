"""Backtest metrics: how good the predictions were, race by race and per season.

The predicted finishing order is scored with:

- winner accuracy: share of races where the predicted winner won;
- podium hit rate: how many of the predicted top 3 finished in the top 3, out of 3;
- position MAE and Spearman: how close the predicted order of the cars that finished is to
  their actual order. Retirements are left out here: whether a car breaks down is the DNF
  model's question, and it gets its own metrics.

The win, podium and points chances are scored over every entrant with:

- Brier score: mean squared gap between the chance given and what happened (1 or 0);
- log loss: like Brier, but much harsher on confident misses. Chances are floored at
  1 in 10,000 so that one freak result can't swamp a whole season.

A season's value is the average over its races, with a 95% bootstrap confidence interval.
``compare`` works on race-by-race differences between two models, so both are always judged
on the same resampled races (a paired bootstrap).
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import pandas as pd

from f1predict import config

BOOTSTRAP_SAMPLES = 10_000
CONFIDENCE = 0.95
CHANCE_FLOOR = 1e-4
CALIBRATION_BINS = 10
SUMMARY_COLUMNS = ["season", "metric", "value", "low", "high", "races"]
CALIBRATION_COLUMNS = ["season", "outcome", "bin_low", "bin_high", "n", "predicted", "observed"]


class Metric(NamedTuple):
    """How a metric is shown: its name, which direction is better, and its number format."""

    label: str
    better: str
    fmt: str


METRICS = {
    "winner_correct": Metric("Winner accuracy", "higher", "{:.0%}"),
    "podium_hit_rate": Metric("Podium hit rate", "higher", "{:.0%}"),
    "mae": Metric("Position MAE", "lower", "{:.2f}"),
    "spearman": Metric("Spearman", "higher", "{:.2f}"),
    **{f"brier_{o}": Metric(f"Brier: {o}", "lower", "{:.3f}") for o in config.OUTCOMES},
    **{f"log_loss_{o}": Metric(f"Log loss: {o}", "lower", "{:.3f}") for o in config.OUTCOMES},
}

# ---------------------------------------------------------------------------
# Outcomes
# ---------------------------------------------------------------------------


def outcome_flags(predictions: pd.DataFrame) -> pd.DataFrame:
    """Whether each driver won, reached the podium and scored points (official position)."""
    position = predictions["position"].astype("float64")
    return pd.DataFrame(
        {outcome: position <= places for outcome, places in config.OUTCOMES.items()},
        index=predictions.index,
    )


def add_outcomes(predictions: pd.DataFrame) -> pd.DataFrame:
    """A copy of ``predictions`` with boolean ``win``, ``podium`` and ``points`` columns."""
    flags = outcome_flags(predictions)
    return predictions.assign(**{outcome: flags[outcome] for outcome in flags})


# ---------------------------------------------------------------------------
# Scoring one race
# ---------------------------------------------------------------------------


def brier_score(chances: np.ndarray, happened: np.ndarray) -> float:
    """Mean squared difference between each chance and the outcome (1 happened, 0 didn't)."""
    return float(np.mean((chances - happened) ** 2))


def log_loss(chances: np.ndarray, happened: np.ndarray) -> float:
    """Mean negative log of the chance given to what actually happened (chances floored)."""
    p = np.clip(chances, CHANCE_FLOOR, 1 - CHANCE_FLOOR)
    return float(-np.mean(happened * np.log(p) + (1 - happened) * np.log(1 - p)))


def score_race(race: pd.DataFrame) -> dict[str, float]:
    """Every metric for one race; the chance metrics are NaN if the model gave no chances.

    ``race`` holds one race's predictions and outcomes: ``driver_id``, ``pred_rank``,
    ``position``, ``finished``, ``finish_rank`` and any ``p_win``/``p_podium``/``p_points``.
    MAE and Spearman re-number the predicted order of the finishers 1..n before comparing it
    with their actual finishing order.
    """
    flags = outcome_flags(race)
    by_rank = race.sort_values("pred_rank")
    predicted_podium = set(by_rank["driver_id"].head(3))
    actual_podium = set(race.loc[flags["podium"], "driver_id"])
    finishers = race[race["finished"].fillna(False).astype(bool)]
    predicted_order = finishers["pred_rank"].rank(method="first")
    scores = {
        "winner_correct": float(flags.at[by_rank.index[0], "win"]),
        "podium_hit_rate": len(predicted_podium & actual_podium) / 3,
        "mae": float((predicted_order - finishers["finish_rank"]).abs().mean()),
        "spearman": float(predicted_order.corr(finishers["finish_rank"], method="spearman")),
    }
    for outcome in config.OUTCOMES:
        column = f"p_{outcome}"
        happened = flags[outcome].to_numpy("float64")
        chances = race[column].to_numpy("float64") if column in race else None
        scores[f"brier_{outcome}"] = np.nan if chances is None else brier_score(chances, happened)
        scores[f"log_loss_{outcome}"] = np.nan if chances is None else log_loss(chances, happened)
    return scores


def race_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    """Score every race of a backtest: one row per race, one column per metric."""
    rows = [
        {
            "season": season,
            "round": round_number,
            "circuit_id": race["circuit_id"].iloc[0],
            "entries": len(race),
            **score_race(race),
        }
        for (season, round_number), race in predictions.groupby(["season", "round"])
    ]
    return pd.DataFrame(rows, columns=["season", "round", "circuit_id", "entries", *METRICS])


# ---------------------------------------------------------------------------
# Season summaries with bootstrap confidence intervals
# ---------------------------------------------------------------------------


def bootstrap_ci(
    values: np.ndarray,
    samples: int = BOOTSTRAP_SAMPLES,
    seed: int = config.DEFAULT_SEED,
    confidence: float = CONFIDENCE,
) -> tuple[np.ndarray, np.ndarray]:
    """Percentile bootstrap interval for the mean of each column of ``values`` (rows = races).

    Builds ``samples`` imaginary seasons of the same length by drawing races at random with
    replacement, averages each one, and returns the range that holds the middle
    ``confidence`` share of those averages: ``(low, high)``, one entry per column.
    """
    table = np.asarray(values, dtype="float64").reshape(len(values), -1)
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(table), size=(samples, len(table)))
    means = table[picks].mean(axis=1)
    tail = (1 - confidence) / 2 * 100
    low, high = np.percentile(means, [tail, 100 - tail], axis=0)
    return low, high


def summarize(
    races: pd.DataFrame, samples: int = BOOTSTRAP_SAMPLES, seed: int = config.DEFAULT_SEED
) -> pd.DataFrame:
    """Each season's average of every metric, with a bootstrap 95% confidence interval.

    Returns one row per season and metric (``value``, ``low``, ``high``, ``races``).
    Metrics a model doesn't produce (NaN for every race) are left out.
    """
    rows = []
    for season, group in races.groupby("season"):
        metrics = [m for m in METRICS if m in group and group[m].notna().any()]
        values = group[metrics].to_numpy("float64")
        low, high = bootstrap_ci(values, samples, seed)
        for i, metric in enumerate(metrics):
            rows.append(
                {
                    "season": season,
                    "metric": metric,
                    "value": values[:, i].mean(),
                    "low": low[i],
                    "high": high[i],
                    "races": len(group),
                }
            )
    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)


def compare(
    races: pd.DataFrame,
    reference: pd.DataFrame,
    samples: int = BOOTSTRAP_SAMPLES,
    seed: int = config.DEFAULT_SEED,
) -> pd.DataFrame:
    """How a model differs from a reference model (e.g. the baseline), with paired CIs.

    Both tables must score exactly the same races. The difference (model minus reference)
    is taken race by race and then summarized like ``summarize``, so every bootstrap draw
    judges both models on the same races. If the interval excludes 0, the gap is unlikely to
    come from which races happened to be in the season.
    """
    keys = ["season", "round"]
    merged = races.merge(reference, on=keys, suffixes=("", "_ref"), validate="one_to_one")
    if len(merged) != len(races) or len(merged) != len(reference):
        raise ValueError("compare() needs both models scored on exactly the same races")
    differences = merged[keys].copy()
    for metric in METRICS:
        if metric in races and metric in reference:
            differences[metric] = merged[metric] - merged[f"{metric}_ref"]
    return summarize(differences, samples, seed).rename(columns={"value": "difference"})


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------


def calibration_table(predictions: pd.DataFrame, bins: int = CALIBRATION_BINS) -> pd.DataFrame:
    """Chance given vs how often it happened, per season, outcome and chance bin.

    Chances are cut into ``bins`` equal-width bins (0-10%, 10-20%, ...). For each bin with
    drivers in it: how many (``n``), the mean chance they were given (``predicted``) and the
    share for whom it happened (``observed``). In a well-calibrated model the two are close.
    """
    flags = outcome_flags(predictions)
    tables = []
    for outcome in config.OUTCOMES:
        column = f"p_{outcome}"
        if column not in predictions:
            continue
        chance = predictions[column].astype("float64")
        frame = pd.DataFrame(
            {
                "season": predictions["season"],
                "outcome": outcome,
                "bin": np.minimum((chance * bins).astype("int64"), bins - 1),
                "chance": chance,
                "happened": flags[outcome].astype("float64"),
            }
        )
        tables.append(
            frame.groupby(["season", "outcome", "bin"], as_index=False).agg(
                n=("chance", "size"), predicted=("chance", "mean"), observed=("happened", "mean")
            )
        )
    if not tables:
        return pd.DataFrame(columns=CALIBRATION_COLUMNS)
    table = pd.concat(tables, ignore_index=True)
    table["bin_low"] = table["bin"] / bins
    table["bin_high"] = (table["bin"] + 1) / bins
    return table[CALIBRATION_COLUMNS]
