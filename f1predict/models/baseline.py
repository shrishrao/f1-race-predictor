"""Grid baseline: everyone finishes where they start, with chances from history by grid slot.

This is the yardstick every other model is measured against. The predicted finishing order
is the starting grid, with pit-lane starters at the back. The chances come from earlier
races: how often did a car starting from each slot win, reach the podium, or score points?
Three steps keep that lookup table sensible:

1. Smoothing: each slot gets ``PRIOR_STARTS`` imaginary starts at the average rate, so a
   slot that has never produced a winner still gets a small chance instead of zero.
2. A monotone fit (isotonic regression): rates may never rise further down the grid. Raw
   counts say P14 has won more often than P11, which is luck, not a reason to prefer P14.
   Neighbouring slots the data can't tell apart are pooled into one shared rate.
3. Per-race scaling: each race's chances are shifted to add up to exactly 1 winner,
   3 podium places and 10 points places.

Pit-lane starters get their own row (slot 0, Jolpica's own number for the pit lane). They
are often quick cars serving a penalty, so they score points more often than the last few
grid slots and don't fit the "further back is worse" rule.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import expit, logit
from sklearn.isotonic import IsotonicRegression

from f1predict import config
from f1predict.models import require_one_race

KEYS = ["season", "round", "driver_id"]
PIT_LANE = 0
PRIOR_STARTS = 2.0
CHANCE_CLIP = 1e-9

# ---------------------------------------------------------------------------
# Rates by grid slot
# ---------------------------------------------------------------------------


def start_slots(entries: pd.DataFrame) -> pd.Series:
    """Each entry's starting slot as an integer, with every pit-lane start as slot 0."""
    return entries["grid"].where(entries["pitlane_start"] == 0, PIT_LANE).astype("int64")


def slot_rates(history: pd.DataFrame, prior_starts: float = PRIOR_STARTS) -> pd.DataFrame:
    """Smoothed, monotone win/podium/points rates per starting slot, from past races.

    ``history`` needs ``grid``, ``pitlane_start`` and the official ``position`` of every
    entry. Returns one row per slot (0 = pit lane, then each grid slot seen) with the
    number of ``starts`` and one rate column per outcome (``win``, ``podium``, ``points``).
    The pit-lane rate is smoothed toward the last grid slot's rate, so with no pit-lane
    history a pit-lane start counts as starting last.
    """
    if history.empty:
        raise ValueError("The grid baseline needs at least one earlier race to learn from.")
    slots = start_slots(history)
    position = history["position"].astype("float64")
    index = sorted({PIT_LANE, *slots})
    starts = slots.value_counts().reindex(index, fill_value=0).astype("float64")
    on_grid = starts.index != PIT_LANE
    table = pd.DataFrame({"starts": starts})
    for outcome, places in config.OUTCOMES.items():
        hits = (position <= places).groupby(slots).sum().reindex(index, fill_value=0)
        average = hits.sum() / starts.sum()
        smoothed = (hits + prior_starts * average) / (starts + prior_starts)
        monotone = IsotonicRegression(increasing=False).fit(
            starts.index[on_grid],
            smoothed[on_grid],
            sample_weight=(starts + prior_starts)[on_grid],
        )
        rates = pd.Series(monotone.predict(starts.index[on_grid]), index=starts.index[on_grid])
        last = rates.iloc[-1]
        rates[PIT_LANE] = (hits[PIT_LANE] + prior_starts * last) / (starts[PIT_LANE] + prior_starts)
        table[outcome] = rates
    return table


# ---------------------------------------------------------------------------
# Per-race scaling
# ---------------------------------------------------------------------------


def scale_to_total(chances: np.ndarray, total: int) -> np.ndarray:
    """Shift every chance's log-odds by the same amount so the chances add up to ``total``.

    Shifting log-odds keeps each chance between 0 and 1 and keeps the drivers in the same
    order; multiplying instead could push a 95% points chance above 100%. When there are no
    more cars than places (``total``), every car gets them: all chances are 1.
    """
    if len(chances) <= total:
        return np.ones(len(chances))
    log_odds = logit(np.clip(chances, CHANCE_CLIP, 1 - CHANCE_CLIP))
    shift = brentq(lambda s: expit(log_odds + s).sum() - total, -60.0, 60.0)
    return expit(log_odds + shift)


# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------


@dataclass
class GridBaseline:
    """Predict that every car finishes where it starts; chances come from ``slot_rates``."""

    prior_starts: float = PRIOR_STARTS
    rates: pd.DataFrame | None = None

    def fit(self, history: pd.DataFrame) -> GridBaseline:
        """Count wins, podiums and points finishes per starting slot in earlier races."""
        self.rates = slot_rates(history, self.prior_starts)
        return self

    def predict(self, race: pd.DataFrame) -> pd.DataFrame:
        """Predict one race: grid order, and win/podium/points chances by slot.

        Ties (several pit-lane starters share the last place) are broken by driver id, so
        the order is always the same. Slots further back than any seen in training (P21 and
        P22 before the first 22-car race) get the last known slot's rates.
        """
        if self.rates is None:
            raise RuntimeError("Call fit() before predict().")
        require_one_race(race)

        out = race[KEYS].copy()
        order = race.sort_values(["grid", "driver_id"]).index
        out["pred_rank"] = pd.Series(np.arange(1, len(race) + 1), index=order)

        slots = start_slots(race).to_numpy()
        grid_rates = self.rates.drop(index=PIT_LANE)
        for outcome, places in config.OUTCOMES.items():
            chances = np.interp(slots, grid_rates.index, grid_rates[outcome])
            chances = np.where(slots == PIT_LANE, self.rates.at[PIT_LANE, outcome], chances)
            out[f"p_{outcome}"] = scale_to_total(chances, places)
        return out
