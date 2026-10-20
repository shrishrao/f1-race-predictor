"""Tests for the grid baseline: slot rates, smoothing, the monotone fit, and predictions."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from f1predict.models.baseline import PIT_LANE, GridBaseline, scale_to_total, slot_rates

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def race_rows(race: int, finishing_order: list[int], pit_lane: tuple[int, ...] = ()) -> list[dict]:
    """One past race: ``finishing_order`` lists the cars' grid slots from winner to last.

    Cars whose slot is in ``pit_lane`` started from the pit lane instead.
    """
    return [
        {
            "season": 2024,
            "round": race,
            "driver_id": f"car{slot}",
            "grid": float(slot),
            "pitlane_start": float(slot in pit_lane),
            "position": position,
        }
        for position, slot in enumerate(finishing_order, start=1)
    ]


def history(*races: list[int], pit_lane: tuple[int, ...] = ()) -> pd.DataFrame:
    """Several past races; each argument is one race's finishing order by grid slot."""
    rows = [row for i, order in enumerate(races, start=1) for row in race_rows(i, order, pit_lane)]
    return pd.DataFrame(rows)


def upcoming(grid_cars: int, pit_lane: tuple[str, ...] = (), round_number: int = 1) -> pd.DataFrame:
    """Feature rows of a race to predict: car1..carN on the grid, then pit-lane starters."""
    rows = [
        {
            "season": 2025,
            "round": round_number,
            "driver_id": f"car{slot}",
            "grid": float(slot),
            "pitlane_start": 0.0,
        }
        for slot in range(1, grid_cars + 1)
    ]
    rows += [
        {
            "season": 2025,
            "round": round_number,
            "driver_id": name,
            "grid": float(grid_cars + 1),
            "pitlane_start": 1.0,
        }
        for name in pit_lane
    ]
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def fitted() -> GridBaseline:
    """A baseline fit on 40 made-up 20-car races where the result roughly follows the grid."""
    rng = np.random.default_rng(3)
    races = [list(np.argsort(np.arange(20) + rng.normal(0, 4, 20)) + 1) for _ in range(40)]
    return GridBaseline().fit(history(*races))


# ---------------------------------------------------------------------------
# Rates by grid slot
# ---------------------------------------------------------------------------


def test_rates_are_smoothed_toward_the_average() -> None:
    """Slot 1 won 3 of 4 races, slot 2 one, slot 3 none; 2 extra starts at the 1-in-3 average."""
    rates = slot_rates(history([1, 2, 3], [1, 3, 2], [2, 1, 3], [1, 2, 3]), prior_starts=2)
    average = 4 / 12
    assert rates.loc[1, "win"] == pytest.approx((3 + 2 * average) / (4 + 2))
    assert rates.loc[2, "win"] == pytest.approx((1 + 2 * average) / (4 + 2))
    assert rates.loc[3, "win"] == pytest.approx((0 + 2 * average) / (4 + 2))
    assert rates.loc[1, "starts"] == 4


def test_a_lucky_back_slot_is_pooled_with_the_slot_ahead() -> None:
    """Slot 3 won twice and slot 2 never: the monotone fit gives both one pooled rate."""
    rates = slot_rates(history([1, 2, 3], [1, 3, 2], [3, 1, 2], [3, 2, 1]), prior_starts=2)
    average = 4 / 12
    pooled = (0 + 2 + 2 * 2 * average) / (4 + 4 + 2 * 2)
    assert rates.loc[1, "win"] == pytest.approx((2 + 2 * average) / (4 + 2))
    assert rates.loc[2, "win"] == pytest.approx(pooled)
    assert rates.loc[3, "win"] == pytest.approx(pooled)


def test_pit_lane_has_its_own_rate_smoothed_toward_the_last_slot() -> None:
    """The pit-lane car won 1 of 2; its rate mixes that record with the last slot's rate."""
    rates = slot_rates(history([4, 1, 2, 3], [1, 2, 3, 4], pit_lane=(4,)), prior_starts=2)
    average = 2 / 8
    last_slot = (0 + 2 * average) / (2 + 2)
    assert rates.loc[3, "win"] == pytest.approx(last_slot)
    assert rates.loc[PIT_LANE, "win"] == pytest.approx((1 + 2 * last_slot) / (2 + 2))
    assert rates.loc[PIT_LANE, "starts"] == 2


def test_with_no_pit_lane_history_a_pit_lane_start_counts_as_starting_last() -> None:
    """No pit-lane starts in history: the pit-lane rate equals the last grid slot's."""
    rates = slot_rates(history([1, 2, 3], [2, 1, 3]))
    assert rates.loc[PIT_LANE, "starts"] == 0
    assert rates.loc[PIT_LANE, "win"] == pytest.approx(rates.loc[3, "win"])


# ---------------------------------------------------------------------------
# Scaling chances to the places available
# ---------------------------------------------------------------------------


def test_scaling_hits_the_total_and_stays_between_0_and_1() -> None:
    """Multiplying these up to 10 points places would give the first car 165%."""
    chances = np.array([0.99] + [0.5] * 10)
    scaled = scale_to_total(chances, 10)
    assert scaled.sum() == pytest.approx(10)
    assert ((scaled > 0) & (scaled < 1)).all()
    assert (np.diff(scaled) <= 0).all()


def test_with_no_more_cars_than_places_everyone_gets_one() -> None:
    """Six cars and ten points places: every car scores."""
    assert scale_to_total(np.array([0.2, 0.9, 0.5, 0.1, 0.3, 0.4]), 10).tolist() == [1.0] * 6


# ---------------------------------------------------------------------------
# Predicting a race
# ---------------------------------------------------------------------------


def test_predicted_order_is_the_grid_with_pit_lane_starters_last(fitted) -> None:
    """Grid order; two pit-lane starters share the back and are ordered by driver id."""
    out = fitted.predict(upcoming(18, pit_lane=("zz_late", "aa_early"))).set_index("driver_id")
    assert out.loc["car1", "pred_rank"] == 1
    assert out.loc["car18", "pred_rank"] == 18
    assert out.loc["aa_early", "pred_rank"] == 19
    assert out.loc["zz_late", "pred_rank"] == 20


def test_chances_add_up_per_race_and_never_rise_down_the_grid(fitted) -> None:
    """1 winner, 3 podium places, 10 points places; chances fall (or stay) down the grid."""
    out = fitted.predict(upcoming(22)).sort_values("pred_rank")
    assert out["p_win"].sum() == pytest.approx(1)
    assert out["p_podium"].sum() == pytest.approx(3)
    assert out["p_points"].sum() == pytest.approx(10)
    for column in ("p_win", "p_podium", "p_points"):
        assert out[column].is_monotonic_decreasing
        assert out[column].between(0, 1, inclusive="neither").all()


def test_slots_never_seen_get_the_last_known_slots_chances(fitted) -> None:
    """Trained on 20-car races, the first 22-car race gives P21 and P22 the rates of P20."""
    out = fitted.predict(upcoming(22)).set_index("driver_id")
    for column in ("p_win", "p_podium", "p_points"):
        assert out.loc["car21", column] == pytest.approx(out.loc["car20", column])
        assert out.loc["car22", column] == pytest.approx(out.loc["car20", column])


def test_misuse_fails_loudly(fitted) -> None:
    """Predicting before fitting, fitting on nothing, or predicting two races at once."""
    with pytest.raises(RuntimeError, match="fit"):
        GridBaseline().predict(upcoming(3))
    with pytest.raises(ValueError, match="at least one earlier race"):
        GridBaseline().fit(history())
    with pytest.raises(ValueError, match="exactly one race"):
        fitted.predict(pd.concat([upcoming(3), upcoming(3, round_number=2)]))
