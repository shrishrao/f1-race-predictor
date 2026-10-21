"""LightGBM ranker: a score for every driver, where a higher score means a better finish.

``LGBMRanker`` with the lambdarank objective learns from whole races ("groups"). For each
past race it sees the cars that took the chequered flag, their features, and a label for
how well each one finished: the winner gets the highest label, the last car home gets 0.
Gradient boosting then adds small decision trees one at a time, each one fixing the pairs
of cars the model so far puts in the wrong order. Swaps near the front of the field cost
more than swaps at the back, because that is what ranking metrics (NDCG) reward.

Only finishers are used for training: the ranker answers "in what order would they finish
if nobody broke down?". Breakdowns are left to the separate DNF model. When predicting,
every car on the grid gets a score.

The default settings were chosen on the 2024 validation season with
``python -m f1predict tune`` (the simplest settings within one standard error of the best);
the 2025 and 2026 test seasons played no part in choosing them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from f1predict import config
from f1predict.features.build import feature_columns
from f1predict.models import require_one_race

KEYS = ["season", "round", "driver_id"]
MAX_LABEL = 30
LINEAR_GAIN = list(range(MAX_LABEL + 1))
FIXED_PARAMS = {
    "objective": "lambdarank",
    "learning_rate": 0.05,
    "random_state": config.DEFAULT_SEED,
    "n_jobs": 1,
    "deterministic": True,
    "force_row_wise": True,
    "verbose": -1,
}

# ---------------------------------------------------------------------------
# Training data
# ---------------------------------------------------------------------------


def training_rows(history: pd.DataFrame) -> pd.DataFrame:
    """The finishers of every past race, sorted so that each race's rows sit together."""
    finished = history["finished"].fillna(False).astype(bool)
    return history[finished].sort_values(["race_seq", "finish_rank"], kind="stable")


def relevance(train: pd.DataFrame) -> np.ndarray:
    """Each finisher's label: 0 for the last car home, up to ``n_finishers - 1`` for the winner."""
    return (train["n_finishers"] - train["finish_rank"]).astype("int64").to_numpy()


def race_sizes(train: pd.DataFrame) -> np.ndarray:
    """How many rows each race contributes, in row order (LightGBM's ``group``)."""
    return train.groupby("race_seq", sort=True).size().to_numpy()


# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------


@dataclass
class RankerModel:
    """LightGBM lambdarank over one feature set; tunable settings are the dataclass fields.

    ``gain`` is how much a finishing place is worth to the ranking objective. With
    ``"exponential"`` (LightGBM's default) a place is worth 2^label - 1, so the winner counts
    twice as much as second and the back of the field barely counts. With ``"linear"`` each
    place is worth its label, which spreads the model's attention down the whole order.
    """

    feature_set: str = "full"
    n_estimators: int = 50
    num_leaves: int = 3
    min_child_samples: int = 20
    gain: str = "exponential"
    booster: lgb.Booster | None = field(default=None, repr=False)

    @property
    def features(self) -> list[str]:
        """The input columns of this model's feature set."""
        return feature_columns(self.feature_set)

    def settings(self) -> dict[str, Any]:
        """The tunable settings, as saved next to a trained model."""
        return {
            "n_estimators": self.n_estimators,
            "num_leaves": self.num_leaves,
            "min_child_samples": self.min_child_samples,
            "gain": self.gain,
        }

    def lgbm_params(self) -> dict[str, Any]:
        """Everything passed to ``LGBMRanker``: the fixed settings plus the tuned ones."""
        params = {
            **FIXED_PARAMS,
            "n_estimators": self.n_estimators,
            "num_leaves": self.num_leaves,
            "min_child_samples": self.min_child_samples,
        }
        if self.gain == "linear":
            params["label_gain"] = LINEAR_GAIN
        elif self.gain != "exponential":
            raise ValueError(f"gain must be 'exponential' or 'linear', not {self.gain!r}")
        return params

    def fit(self, history: pd.DataFrame) -> RankerModel:
        """Train on the finishers of every race in ``history``, one group per race."""
        train = training_rows(history)
        ranker = lgb.LGBMRanker(**self.lgbm_params())
        ranker.fit(train[self.features], relevance(train), group=race_sizes(train))
        self.booster = ranker.booster_
        return self

    def predict(self, race: pd.DataFrame) -> pd.DataFrame:
        """Score every car in one race and order them by score.

        Ties go to the better grid slot when the grid is one of the model's inputs, and
        otherwise by driver id, so a model that isn't allowed to see the grid never uses it.
        Returns the ids, ``score`` (higher = better; only differences within a race mean
        anything) and ``pred_rank``.
        """
        if self.booster is None:
            raise RuntimeError("Call fit() before predict().")
        require_one_race(race)
        out = race[KEYS].copy()
        out["score"] = self.booster.predict(race[self.features], validate_features=True)
        tie_break = ["grid", "driver_id"] if "grid" in self.features else ["driver_id"]
        order = (
            out.assign(grid=race["grid"])
            .sort_values(["score", *tie_break], ascending=[False] + [True] * len(tie_break))
            .index
        )
        out["pred_rank"] = pd.Series(np.arange(1, len(race) + 1), index=order)
        return out

    def importance(self) -> pd.DataFrame:
        """Each feature's share of the total gain: how much its splits improved the ranking."""
        if self.booster is None:
            raise RuntimeError("Call fit() before importance().")
        gain = self.booster.feature_importance(importance_type="gain")
        share = gain / gain.sum() if gain.sum() > 0 else np.zeros(len(gain))
        table = pd.DataFrame({"feature": self.booster.feature_name(), "gain_share": share})
        return table.sort_values("gain_share", ascending=False, ignore_index=True)
