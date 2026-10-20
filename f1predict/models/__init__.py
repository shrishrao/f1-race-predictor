"""Models: grid baseline, LightGBM ranker, DNF classifier, and the race simulator.

Every model the backtest evaluates has the same two methods, so the backtest can treat them
all alike:

- ``fit(history)`` learns from earlier races: their feature columns plus the outcome columns
  (``position``, ``status_class``, ``finished``, ``finish_rank``, ``n_finishers``).
- ``predict(race)`` gets one race's feature rows, with the outcome columns removed, and
  returns one row per driver: ``season``, ``round``, ``driver_id``, ``pred_rank`` (1 = the
  predicted winner, each rank used once) and, for models that give chances, ``p_win``,
  ``p_podium`` and ``p_points``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Protocol, Self, get_args

if TYPE_CHECKING:
    import pandas as pd

ModelName = Literal["baseline"]
MODEL_NAMES: tuple[str, ...] = get_args(ModelName)

# ---------------------------------------------------------------------------
# The interface
# ---------------------------------------------------------------------------


class RaceModel(Protocol):
    """A model the backtest can train on past races and then ask about the next one."""

    def fit(self, history: pd.DataFrame) -> Self:
        """Learn from the feature and outcome rows of earlier races; return the model."""

    def predict(self, race: pd.DataFrame) -> pd.DataFrame:
        """Predict one race from its feature rows."""


# ---------------------------------------------------------------------------
# Lookup by name
# ---------------------------------------------------------------------------


def make_model(name: str) -> RaceModel:
    """A new, untrained model by its command-line name (imports only that model's code)."""
    if name == "baseline":
        from f1predict.models.baseline import GridBaseline

        return GridBaseline()
    raise ValueError(f"Unknown model {name!r}; expected one of {MODEL_NAMES}")
