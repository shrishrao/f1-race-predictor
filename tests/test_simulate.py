"""Tests for the Plackett-Luce simulation and the temperature fit.

The simulation is checked against exact Plackett-Luce probabilities (small races can be
enumerated order by order), and the temperature fit against made-up races generated with a
known temperature.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest
from synthetic import feature_rows, learnable_features, make_tables
from typer.testing import CliRunner

from f1predict import config
from f1predict.__main__ import app
from f1predict.evaluate import tuning
from f1predict.evaluate.backtest import predict_races
from f1predict.features.build import build_features, save_features
from f1predict.models import make_model
from f1predict.models.ranker import RankerModel
from f1predict.models.simulate import (
    SimulationModel,
    finishing_orders,
    fit_temperature,
    order_log_likelihood,
    outcome_chances,
    position_probabilities,
)
from f1predict.models.store import load_ranker

runner = CliRunner()
SCORES = np.array([1.2, 0.3, -0.5, 0.9])

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def exact_positions(strengths: np.ndarray) -> np.ndarray:
    """The exact position matrix of a small race, adding up every possible finishing order."""
    cars = len(strengths)
    matrix = np.zeros((cars, cars))
    for order in itertools.permutations(range(cars)):
        chance = math.exp(order_log_likelihood(strengths[list(order)]))
        for place, car in enumerate(order):
            matrix[car, place] += chance
    return matrix


def made_up_races(temperature: float, races: int = 300, cars: int = 10, seed: int = 5) -> list:
    """Orders drawn from Plackett-Luce at a known temperature: each race's scores, winner first."""
    rng = np.random.default_rng(seed)
    orders = []
    for _ in range(races):
        scores = rng.normal(0, 1, cars)
        finish = np.argsort(-(scores / temperature + rng.gumbel(size=cars)))
        orders.append(scores[finish])
    return orders


# ---------------------------------------------------------------------------
# The exact likelihood
# ---------------------------------------------------------------------------


def test_the_likelihood_is_the_place_by_place_product() -> None:
    """d1 wins from all four, then d4 from the last three, then d2 from the last two."""
    order = [0, 3, 1, 2]
    weights = np.exp(SCORES)
    by_hand = (
        weights[0]
        / weights.sum()
        * weights[3]
        / weights[[3, 1, 2]].sum()
        * weights[1]
        / weights[[1, 2]].sum()
    )
    assert math.exp(order_log_likelihood(SCORES[order])) == pytest.approx(by_hand)


def test_every_possible_order_adds_up_to_one() -> None:
    """The 24 orders of 4 cars are all the possibilities, so their chances sum to 1."""
    total = sum(
        math.exp(order_log_likelihood(SCORES[list(order)]))
        for order in itertools.permutations(range(4))
    )
    assert total == pytest.approx(1)


# ---------------------------------------------------------------------------
# Simulating
# ---------------------------------------------------------------------------


def test_simulated_positions_match_the_exact_odds() -> None:
    """400,000 simulated races land within half a percentage point of the exact matrix."""
    simulated = position_probabilities(SCORES, temperature=1.0, sims=400_000, seed=1)
    assert np.abs(simulated - exact_positions(SCORES)).max() < 0.005
    weights = np.exp(SCORES)
    assert simulated[:, 0] == pytest.approx(weights / weights.sum(), abs=0.005)


def test_chances_add_up_and_nest() -> None:
    """22 cars: 1 winner, 3 podium places, 10 points places, and win <= podium <= points."""
    scores = np.random.default_rng(0).normal(0, 1, 22)
    positions = position_probabilities(scores, temperature=0.8, sims=50_000)
    assert positions.sum(axis=1) == pytest.approx(np.ones(22))
    assert positions.sum(axis=0) == pytest.approx(np.ones(22))
    chances = outcome_chances(positions)
    assert chances["p_win"].sum() == pytest.approx(1)
    assert chances["p_podium"].sum() == pytest.approx(3)
    assert chances["p_points"].sum() == pytest.approx(10)
    assert (chances["p_win"] <= chances["p_podium"]).all()
    assert (chances["p_podium"] <= chances["p_points"]).all()


def test_with_fewer_cars_than_points_places_everyone_scores() -> None:
    """Three cars and ten points places."""
    chances = outcome_chances(position_probabilities(SCORES[:3], temperature=1.0, sims=1_000))
    assert chances["p_points"] == pytest.approx(np.ones(3))


def test_the_same_seed_gives_the_same_answer_whatever_the_batch_size() -> None:
    """Reproducible to the last digit; batching only changes memory use, not the result."""
    one_go = position_probabilities(SCORES, 1.0, sims=30_000, seed=7, batch=30_000)
    batched = position_probabilities(SCORES, 1.0, sims=30_000, seed=7, batch=4_096)
    assert np.array_equal(one_go, batched)
    other_seed = position_probabilities(SCORES, 1.0, sims=30_000, seed=8)
    assert not np.array_equal(one_go, other_seed)
    assert np.abs(one_go - other_seed).max() < 0.02


def test_a_higher_temperature_flattens_the_chances() -> None:
    """The favourite's win chance falls as T rises, toward 1 in 4 for four cars."""
    favourite = [
        position_probabilities(SCORES, t, sims=20_000)[0, 0] for t in (0.25, 1.0, 4.0, 100.0)
    ]
    assert favourite == sorted(favourite, reverse=True)
    assert favourite[-1] == pytest.approx(0.25, abs=0.02)


# ---------------------------------------------------------------------------
# Fitting the temperature
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("true_temperature", [0.5, 2.0])
def test_the_fit_recovers_a_known_temperature(true_temperature) -> None:
    """300 made-up races drawn at a known T: the fit lands within 10% of it."""
    fitted = fit_temperature(made_up_races(true_temperature))
    assert fitted == pytest.approx(true_temperature, rel=0.1)


def test_a_fit_needs_at_least_one_real_race() -> None:
    """With no race of two or more finishers there's nothing to learn from."""
    with pytest.raises(ValueError, match="two finishers"):
        fit_temperature([np.array([1.0])])


def test_finishing_orders_use_finishers_in_their_actual_order() -> None:
    """The retiree is left out; the rest are listed by finishing rank, winner first."""
    frame = learnable_features(races=2)
    frame["score"] = -frame["quali_gap_pole_pct"]
    orders = finishing_orders(frame)
    assert len(orders) == 2
    assert all(len(order) == 9 for order in orders)
    assert all(np.all(np.diff(order) <= 0) for order in orders)


# ---------------------------------------------------------------------------
# The model in the backtest
# ---------------------------------------------------------------------------


def fresh_simulation_model() -> SimulationModel:
    """A new, untrained ranker inside a simulation at T = 1, as the backtest needs per race."""
    return SimulationModel(RankerModel(), temperature=1.0, sims=20_000)


def test_the_simulation_model_gives_valid_chances_in_the_backtest() -> None:
    """Every race gets chances that add up; the top-scored car is the favourite to win."""
    predicted = predict_races(fresh_simulation_model, learnable_features(), [2021])
    for _, race in predicted.groupby("round"):
        assert race["p_win"].sum() == pytest.approx(1)
        assert race["p_podium"].sum() == pytest.approx(3)
        assert race.loc[race["score"].idxmax(), "p_win"] >= race["p_win"].max() - 0.01


def test_make_model_adds_the_simulation_only_with_a_temperature() -> None:
    """``ranker`` alone gives an order; with a temperature it also gives chances."""
    assert isinstance(make_model("ranker"), RankerModel)
    model = make_model("ranker", "no_practice", temperature=0.9)
    assert isinstance(model, SimulationModel)
    assert model.temperature == 0.9
    assert model.ranker.feature_set == "no_practice"


def test_the_simulation_model_predicts_one_race_at_a_time() -> None:
    """Its ranker trained, the model's chances for one race add up."""
    frame = learnable_features()
    model = SimulationModel(RankerModel(), 1.0, 10_000).fit(frame[frame["season"] == 2020])
    out = model.predict(feature_rows(frame, 30))
    assert out["p_win"].sum() == pytest.approx(1)
    assert {"score", "pred_rank", "p_win", "p_podium", "p_points"} <= set(out.columns)


def test_the_temperature_is_never_fitted_on_a_test_season() -> None:
    """Fitting T on 2025 or 2026 would leak the results we report for them."""
    for season in config.TEST_SEASONS:
        with pytest.raises(ValueError, match="test season"):
            tuning.validation_temperature(learnable_features(), season=season)


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


def test_backtest_command_fits_t_and_scores_the_chances(workspace) -> None:
    """``backtest --model ranker`` fits T on 2024 and compares chances with the baseline."""
    result = runner.invoke(app, ["backtest", "--model", "ranker", "--season", "2024"])
    assert result.exit_code == 0, result.output
    assert "Temperature T =" in result.output
    assert "Brier: win" in result.output


def test_train_command_saves_the_temperature(workspace) -> None:
    """The model's description file carries the temperature to use with its scores."""
    result = runner.invoke(app, ["train"])
    assert result.exit_code == 0, result.output
    _, description = load_ranker("ranker_full", workspace / "models")
    assert description["temperature"] > 0
