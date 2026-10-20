"""Tests for the backtest metrics, each on a hand-built race with a known answer."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from f1predict.evaluate.metrics import (
    CHANCE_FLOOR,
    add_outcomes,
    bootstrap_ci,
    calibration_table,
    compare,
    race_metrics,
    score_race,
    summarize,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_race(
    positions: list[int],
    pred_ranks: list[int],
    finished: list[bool] | None = None,
    round_number: int = 1,
    **chances: list[float],
) -> pd.DataFrame:
    """One scored race: drivers d1..dn with official positions and predicted ranks.

    ``finish_rank`` is filled in from the positions of the cars that finished. Keyword
    arguments add chance columns, e.g. ``p_win=[...]``.
    """
    frame = pd.DataFrame(
        {
            "season": 2025,
            "round": round_number,
            "circuit_id": "monza",
            "driver_id": [f"d{i}" for i in range(1, len(positions) + 1)],
            "pred_rank": pred_ranks,
            "position": positions,
            "finished": finished if finished is not None else [True] * len(positions),
        }
    )
    frame["finish_rank"] = frame["position"].where(frame["finished"]).rank(method="first")
    for column, values in chances.items():
        frame[column] = values
    return frame


def per_race(values: dict[str, list[float]], season: int = 2025) -> pd.DataFrame:
    """A per-race metrics table with the given metric columns, one row per race."""
    rows = len(next(iter(values.values())))
    return pd.DataFrame({"season": season, "round": range(1, rows + 1), **values})


# ---------------------------------------------------------------------------
# The predicted order
# ---------------------------------------------------------------------------


def test_a_perfect_prediction_scores_perfectly() -> None:
    """Right winner, right podium, every finisher in the right place."""
    scores = score_race(make_race([1, 2, 3, 4, 5], [1, 2, 3, 4, 5]))
    assert scores["winner_correct"] == 1
    assert scores["podium_hit_rate"] == 1
    assert scores["mae"] == 0
    assert scores["spearman"] == pytest.approx(1)


def test_a_reversed_prediction() -> None:
    """Upside-down order: wrong winner, 1 of 3 podium places, MAE 12/5, Spearman -1."""
    scores = score_race(make_race([1, 2, 3, 4, 5], [5, 4, 3, 2, 1]))
    assert scores["winner_correct"] == 0
    assert scores["podium_hit_rate"] == pytest.approx(1 / 3)
    assert scores["mae"] == pytest.approx((4 + 2 + 0 + 2 + 4) / 5)
    assert scores["spearman"] == pytest.approx(-1)


def test_retirements_are_left_out_of_mae_and_spearman() -> None:
    """d1 was predicted to win but retired; the other three finished in the predicted order.

    Re-numbered among finishers, the predicted order of d2-d4 is exactly right.
    """
    race = make_race([4, 1, 2, 3], [1, 2, 3, 4], finished=[False, True, True, True])
    scores = score_race(race)
    assert scores["winner_correct"] == 0
    assert scores["podium_hit_rate"] == pytest.approx(2 / 3)
    assert scores["mae"] == 0
    assert scores["spearman"] == pytest.approx(1)


# ---------------------------------------------------------------------------
# Chances
# ---------------------------------------------------------------------------


def test_brier_and_log_loss_by_hand() -> None:
    """d1 won with a 50% chance; the others had 30%, 20% and 0%."""
    scores = score_race(make_race([1, 2, 3, 4], [1, 2, 3, 4], p_win=[0.5, 0.3, 0.2, 0.0]))
    assert scores["brier_win"] == pytest.approx((0.5**2 + 0.3**2 + 0.2**2 + 0) / 4)
    expected = -(math.log(0.5) + math.log(0.7) + math.log(0.8) + math.log(1 - CHANCE_FLOOR)) / 4
    assert scores["log_loss_win"] == pytest.approx(expected)


def test_a_zero_chance_that_happens_costs_a_lot_but_stays_finite() -> None:
    """A 0% winner and a 100% loser are each charged as if given 1 in 10,000."""
    scores = score_race(make_race([1, 2], [2, 1], p_win=[0.0, 1.0]))
    assert scores["log_loss_win"] == pytest.approx(-math.log(CHANCE_FLOOR))
    assert scores["brier_win"] == 1


def test_a_model_without_chances_gets_no_chance_metrics() -> None:
    """Order metrics only: the chance metrics are NaN and left out of the summary."""
    races = race_metrics(make_race([1, 2, 3], [1, 2, 3]))
    assert np.isnan(races.loc[0, "brier_win"])
    assert set(summarize(races, samples=100)["metric"]) == {
        "winner_correct",
        "podium_hit_rate",
        "mae",
        "spearman",
    }


def test_outcomes_come_from_the_official_position() -> None:
    """A car disqualified from P2 is classified 20th, so it scored nothing."""
    flags = add_outcomes(pd.DataFrame({"position": [1, 3, 10, 11, 20]}))
    assert flags["win"].tolist() == [True, False, False, False, False]
    assert flags["podium"].tolist() == [True, True, False, False, False]
    assert flags["points"].tolist() == [True, True, True, False, False]


# ---------------------------------------------------------------------------
# Season summaries and comparisons
# ---------------------------------------------------------------------------


def test_bootstrap_is_reproducible_and_brackets_the_mean() -> None:
    """Same seed, same interval; the season average sits inside it."""
    values = np.array([0.1, 0.5, 0.2, 0.9, 0.4, 0.3])
    low, high = bootstrap_ci(values, seed=1)
    again_low, again_high = bootstrap_ci(values, seed=1)
    assert np.array_equal(low, again_low)
    assert np.array_equal(high, again_high)
    assert low[0] < values.mean() < high[0]


def test_a_metric_that_never_changes_has_no_uncertainty() -> None:
    """Every race scored 0.7, so every resampled season averages 0.7."""
    summary = summarize(per_race({"mae": [0.7] * 8}), samples=200)
    row = summary.iloc[0]
    assert (row["value"], row["low"], row["high"]) == pytest.approx((0.7, 0.7, 0.7))
    assert row["races"] == 8


def test_compare_takes_race_by_race_differences() -> None:
    """A model exactly 0.5 better on every race differs by 0.5 with no uncertainty."""
    baseline = per_race({"mae": [2.0, 3.5, 1.0, 4.0]})
    model = per_race({"mae": [1.5, 3.0, 0.5, 3.5]})
    row = compare(model, baseline, samples=200).iloc[0]
    assert (row["difference"], row["low"], row["high"]) == pytest.approx((-0.5, -0.5, -0.5))


def test_compare_needs_the_same_races() -> None:
    """Comparing seasons of different races would be meaningless, so it's refused."""
    with pytest.raises(ValueError, match="same races"):
        compare(per_race({"mae": [1.0, 2.0]}), per_race({"mae": [1.0, 2.0, 3.0]}))


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------


def test_calibration_bins_compare_chance_with_frequency() -> None:
    """Two drivers given 5% (one won) and two given 95% and 100% (both won)."""
    predictions = make_race([1, 2, 3, 4], [1, 2, 3, 4], p_win=[0.05, 0.05, 0.95, 1.0])
    predictions["position"] = [2, 1, 1, 1]
    table = calibration_table(predictions).query("outcome == 'win'").set_index("bin_low")
    assert table.loc[0.0, "n"] == 2
    assert table.loc[0.0, "predicted"] == pytest.approx(0.05)
    assert table.loc[0.0, "observed"] == pytest.approx(0.5)
    assert table.loc[0.9, "n"] == 2
    assert table.loc[0.9, "observed"] == 1
