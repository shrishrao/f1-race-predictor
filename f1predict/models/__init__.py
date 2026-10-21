"""Models: grid baseline, LightGBM ranker, DNF classifier, and the race simulator.

Every model the backtest evaluates has the same two methods, so the backtest can treat them
all alike:

- ``fit(history)`` learns from earlier races: their feature columns plus the outcome columns
  (``position``, ``status_class``, ``finished``, ``finish_rank``, ``n_finishers``).
- ``predict(race)`` gets one race's feature rows, with the outcome columns removed, and
  returns one row per driver: ``season``, ``round``, ``driver_id``, ``pred_rank`` (1 = the
  predicted winner, each rank used once) and, for models that give chances, ``p_win``,
  ``p_podium`` and ``p_points``. Models that simulate retirements add ``p_dnf``; a DNF
  model on its own (``models/dnf.py``) returns only ``p_dnf``, with no ``pred_rank``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Protocol, Self, get_args

if TYPE_CHECKING:
    from collections.abc import Callable

    import pandas as pd

    from f1predict.models.dnf import DNFModel

ModelName = Literal["baseline", "ranker"]
MODEL_NAMES: tuple[str, ...] = get_args(ModelName)
FeatureSetName = Literal["full", "no_practice", "simple_practice", "no_grid_quali"]
FEATURE_SET_NAMES: tuple[str, ...] = get_args(FeatureSetName)

# ---------------------------------------------------------------------------
# The interface
# ---------------------------------------------------------------------------


class RaceModel(Protocol):
    """A model the backtest can train on past races and then ask about the next one."""

    def fit(self, history: pd.DataFrame) -> Self:
        """Learn from the feature and outcome rows of earlier races; return the model."""

    def predict(self, race: pd.DataFrame) -> pd.DataFrame:
        """Predict one race from its feature rows."""


def require_one_race(race: pd.DataFrame) -> None:
    """Raise unless ``race`` holds the rows of exactly one race."""
    if race[["season", "round"]].drop_duplicates().shape[0] != 1:
        raise ValueError("predict() takes the rows of exactly one race")


# ---------------------------------------------------------------------------
# Lookup by name
# ---------------------------------------------------------------------------


def run_name(name: str, feature_set: str = "full", dnf: bool = True) -> str:
    """Name for one model's saved results, e.g. ``baseline`` or ``ranker_no_practice``.

    A ranker simulated without retirements gets a ``_no_dnf`` suffix.
    """
    if name == "baseline":
        return name
    return f"{name}_{feature_set}" if dnf else f"{name}_{feature_set}_no_dnf"


def make_model(
    name: str,
    feature_set: str = "full",
    temperature: float | None = None,
    dnf: Callable[[], DNFModel] | None = None,
) -> RaceModel:
    """A new, untrained model by its command-line name (imports only that model's code).

    ``feature_set`` picks the ranker's input columns; the grid baseline only uses the grid.
    With a ``temperature``, the ranker's scores are also turned into win, podium and points
    chances by simulating each race; without one, the ranker only gives an order. ``dnf``
    makes a fresh DNF model, so cars retire in the simulated races too (it needs a
    temperature).
    """
    if name == "baseline":
        from f1predict.models.baseline import GridBaseline

        return GridBaseline()
    if name == "ranker":
        from f1predict.models.ranker import RankerModel

        ranker = RankerModel(feature_set=feature_set)
        if temperature is None:
            if dnf is not None:
                raise ValueError("Retirements are simulated, so a DNF model needs a temperature.")
            return ranker
        from f1predict.models.simulate import SimulationModel

        return SimulationModel(ranker, temperature, dnf=None if dnf is None else dnf())
    raise ValueError(f"Unknown model {name!r}; expected one of {MODEL_NAMES}")
