"""Plackett-Luce race simulation: from ranker scores to win, podium and points chances.

The Plackett-Luce model draws a finishing order one place at a time. The winner is drawn
first, each car with a chance proportional to ``exp(score / T)``; second place is drawn the
same way from the cars left, and so on. The temperature ``T`` sets how much the scores are
trusted: a small T means the top score nearly always wins, a large T makes the race close
to a lottery.

The Gumbel trick draws a whole order at once: add independent Gumbel noise to every
``score / T`` and sort. That gives exactly the Plackett-Luce odds, and it works on a whole
batch of simulated races in one NumPy step, with no Python loop over drivers.

Retirements are drawn on top of that: in every simulated race each car retires with its
own DNF chance, independently of the others, and every retiree is placed behind every car
that finished. So a car that is quick but fragile wins fewer simulated races than one that
is just as quick and always finishes.

``T`` is fitted by maximum likelihood: the value under which the races that actually
happened (the finishers' order) were most likely, computed exactly rather than simulated.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from f1predict import config
from f1predict.models.dnf import DNFModel
from f1predict.models.ranker import RankerModel

logger = logging.getLogger(__name__)

SIM_BATCH = 25_000
TEMPERATURE_BOUNDS = (0.01, 100.0)
RETIRED_PENALTY = 1e6
DNF_STREAM = 1

# ---------------------------------------------------------------------------
# Simulating races
# ---------------------------------------------------------------------------


def position_probabilities(
    scores: np.ndarray,
    temperature: float,
    sims: int = config.DEFAULT_SIMS,
    seed: int = config.DEFAULT_SEED,
    batch: int = SIM_BATCH,
    p_dnf: np.ndarray | None = None,
) -> np.ndarray:
    """Chance of each car finishing in each position, from ``sims`` simulated races.

    Returns an n x n matrix: row = car (in the order of ``scores``), column = finishing
    position (0 = winner). Every row and every column sums to 1. Races are simulated
    ``batch`` at a time to keep memory small; the batch size doesn't change the result.

    With ``p_dnf`` (one chance per car), each car also retires from each simulated race
    with its own chance. A retiree's noisy strength drops by ``RETIRED_PENALTY``, far more
    than any real gap, so retirees end up behind every finisher and keep their order among
    themselves. Retirements come from a second random stream (``DNF_STREAM``), so the
    Gumbel draws stay the same: ``p_dnf`` of all zeros gives exactly the result without it.
    """
    strengths = np.asarray(scores, dtype="float64") / temperature
    cars = len(strengths)
    rng = np.random.default_rng(seed)
    retire = None if p_dnf is None else np.asarray(p_dnf, dtype="float64")
    dnf_rng = np.random.default_rng([seed, DNF_STREAM])
    counts = np.zeros(cars * cars, dtype="int64")
    done = 0
    while done < sims:
        size = min(batch, sims - done)
        noisy = strengths + rng.gumbel(size=(size, cars))
        if retire is not None:
            retired = dnf_rng.random(size=(size, cars)) < retire
            noisy = noisy - RETIRED_PENALTY * retired
        order = np.argsort(-noisy, axis=1)
        counts += np.bincount((order * cars + np.arange(cars)).ravel(), minlength=cars * cars)
        done += size
    return counts.reshape(cars, cars) / sims


def outcome_chances(positions: np.ndarray) -> dict[str, np.ndarray]:
    """``p_win``, ``p_podium`` and ``p_points`` per car from a position matrix.

    Sums are clipped at 1: adding ten fractions can give 1.0000000000000002 by rounding.
    """
    cumulative = np.minimum(positions.cumsum(axis=1), 1.0)
    cars = positions.shape[1]
    return {
        f"p_{outcome}": cumulative[:, min(places, cars) - 1]
        for outcome, places in config.OUTCOMES.items()
    }


# ---------------------------------------------------------------------------
# Fitting the temperature
# ---------------------------------------------------------------------------


def order_log_likelihood(strengths: np.ndarray) -> float:
    """Exact log-probability of one finishing order; ``strengths`` are listed winner first.

    Each place contributes ``strength - log(sum of exp(strength) over the cars still left)``.
    """
    remaining = np.logaddexp.accumulate(strengths[::-1])[::-1]
    return float(np.sum(strengths - remaining))


def finishing_orders(predictions: pd.DataFrame) -> list[np.ndarray]:
    """Per race: the scores of the cars that finished, in the order they actually finished."""
    finished = predictions[predictions["finished"].fillna(False).astype(bool)]
    return [
        race.sort_values("finish_rank")["score"].to_numpy("float64")
        for _, race in finished.groupby(["season", "round"])
    ]


def fit_temperature(
    orders: Iterable[np.ndarray], bounds: tuple[float, float] = TEMPERATURE_BOUNDS
) -> float:
    """The temperature that makes the given finishing orders most likely.

    ``orders`` holds, per race, the scores of the finishers, winner first. The search runs
    over log T inside ``bounds``; the log-likelihood has a single peak, so a 1-D bounded
    search finds it.
    """
    races = [np.asarray(order, dtype="float64") for order in orders if len(order) > 1]
    if not races:
        raise ValueError("Fitting a temperature needs at least one race with two finishers.")

    def negative_log_likelihood(log_t: float) -> float:
        """Minus the total log-likelihood of every race at temperature exp(log_t)."""
        t = np.exp(log_t)
        return -sum(order_log_likelihood(race / t) for race in races)

    low, high = np.log(bounds)
    result = minimize_scalar(negative_log_likelihood, bounds=(low, high), method="bounded")
    temperature = float(np.exp(result.x))
    if min(result.x - low, high - result.x) < 0.01:
        logger.warning(
            "The fitted temperature %.3g sits at the edge of its search range", temperature
        )
    return temperature


# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------


@dataclass
class SimulationModel:
    """The ranker's scores turned into chances by simulating each race ``sims`` times.

    ``temperature`` must come from races the model isn't scored on (the validation season);
    the same seed is used for every race, so any single race's prediction can be repeated
    on its own. With a ``dnf`` model, cars also retire in the simulated races; without one,
    every car finishes (the model as it was before DNFs were added).
    """

    ranker: RankerModel
    temperature: float
    sims: int = config.DEFAULT_SIMS
    seed: int = config.DEFAULT_SEED
    dnf: DNFModel | None = None

    def fit(self, history: pd.DataFrame) -> SimulationModel:
        """Train the ranker (and the DNF model) on earlier races; the temperature stays."""
        self.ranker.fit(history)
        if self.dnf is not None:
            self.dnf.fit(history)
        return self

    def predict(self, race: pd.DataFrame) -> pd.DataFrame:
        """The ranker's order and scores, plus win, podium and points chances.

        With a DNF model, also each car's raw ``p_dnf``: its chance of retiring.
        """
        out = self.ranker.predict(race)
        p_dnf = None
        if self.dnf is not None:
            chances = self.dnf.predict(race).set_index("driver_id")["p_dnf"]
            p_dnf = chances.reindex(out["driver_id"]).to_numpy("float64")
            out["p_dnf"] = p_dnf
        positions = position_probabilities(
            out["score"].to_numpy(), self.temperature, self.sims, self.seed, p_dnf=p_dnf
        )
        return out.assign(**outcome_chances(positions))
