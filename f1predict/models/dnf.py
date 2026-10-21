"""DNF models: each car's chance of retiring from the race, three ways.

A DNF ("did not finish") is any start that doesn't end at the chequered flag: an engine
failure, a crash, a puncture that ends the race. Three models give every car a chance of
that happening:

- ``BaseRateDNF``: every car gets the same chance, the share of starters who retired over
  the last ``window`` races. It can't tell one car from another, but it does follow the
  field as a whole: retirements fell to 10% of starts in 2024 and nearly doubled to 19%
  with the 2026 regulation reset.
- ``LogisticDNF``: logistic regression on that same recent rate plus each car's own
  inputs: driver, team and circuit DNF rates, grid slot, pit-lane start and qualifying gap.
  Each input moves the log-odds of a retirement up or down by a fixed amount.
- ``LightGBMDNF``: gradient-boosted trees on the same inputs, which can also pick up
  thresholds and combinations (a back-row start at a street circuit, say).

All three learn only from starters: cars that finished or retired. Non-starters (DNS) never
took part, and a disqualified car (DSQ) lost a result that says nothing about whether it
would have reached the flag, so both are left out. ``predict`` returns ``p_dnf`` per car and
uses only races before the one it predicts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal, get_args

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from f1predict import config
from f1predict.features.build import FEATURE_GROUPS, FEATURE_SETS
from f1predict.models import require_one_race

DNFKind = Literal["base_rate", "logistic", "lightgbm"]
DNF_KINDS: tuple[str, ...] = get_args(DNFKind)
KIND_LABELS = {"base_rate": "Base rate", "logistic": "Logistic regression", "lightgbm": "LightGBM"}
SHORT_LABELS = {"base_rate": "Base rate", "logistic": "Logistic", "lightgbm": "LightGBM"}
KEYS = ["season", "round", "driver_id"]
RACE_KEYS = ["season", "round"]
STARTED = ["finished", "dnf"]
FIELD_RATE = "field_dnf_rate"
CAR_FEATURES = [
    "drv_dnf_rate",
    "team_dnf_rate",
    "circuit_dnf_rate",
    "grid",
    "pitlane_start",
    "quali_gap_pole_pct",
]
DNF_FEATURES = [FIELD_RATE, *CAR_FEATURES]
WINDOWS: tuple[int | None, ...] = (5, 10, 20, 40, 80, None)
DEFAULT_WINDOW = 20
LIGHTGBM_PARAMS = {
    "objective": "binary",
    "n_estimators": 50,
    "num_leaves": 3,
    "min_child_samples": 50,
    "learning_rate": 0.05,
    "random_state": config.DEFAULT_SEED,
    "n_jobs": 1,
    "deterministic": True,
    "force_row_wise": True,
    "verbose": -1,
}

# ---------------------------------------------------------------------------
# Starts and retirements per race
# ---------------------------------------------------------------------------


def starters(history: pd.DataFrame) -> pd.DataFrame:
    """The rows of cars that started the race: finished or retired (DNS and DSQ left out)."""
    return history[history["status_class"].isin(STARTED)]


def dnf_target(rows: pd.DataFrame) -> np.ndarray:
    """1 for each car that retired, 0 for each car that finished."""
    return rows["status_class"].eq("dnf").astype("int64").to_numpy()


def race_counts(history: pd.DataFrame) -> pd.DataFrame:
    """Per past race, in date order: how many cars started and how many of them retired."""
    rows = starters(history)
    if rows.empty:
        raise ValueError("A DNF model needs at least one earlier race with starters.")
    counts = (
        rows.assign(dnf=rows["status_class"].eq("dnf").astype("int64"))
        .groupby(RACE_KEYS, as_index=False)
        .agg(starters=("dnf", "size"), dnfs=("dnf", "sum"))
    )
    return counts.sort_values(RACE_KEYS, ignore_index=True)


def rate_before(counts: pd.DataFrame, season: int, round_number: int, window: int | None) -> float:
    """DNF share of all starters over the ``window`` races before this one (None: all of them).

    NaN when no earlier race is known.
    """
    earlier = counts[
        (counts["season"] < season)
        | ((counts["season"] == season) & (counts["round"] < round_number))
    ]
    if window is not None:
        earlier = earlier.tail(window)
    starts = earlier["starters"].sum()
    return float(earlier["dnfs"].sum() / starts) if starts else np.nan


def rates_before(counts: pd.DataFrame, window: int | None) -> pd.Series:
    """``rate_before`` for every race in ``counts`` at once (NaN for the first race)."""

    def total(column: str) -> pd.Series:
        """Sum of ``column`` over each race's look-back window, ending at the race itself."""
        values = counts[column]
        return values.cumsum() if window is None else values.rolling(window, min_periods=1).sum()

    return (total("dnfs") / total("starters")).shift(1)


def with_field_rate(rows: pd.DataFrame, counts: pd.DataFrame, window: int | None) -> pd.DataFrame:
    """``rows`` plus ``field_dnf_rate``, the DNF share over the races before each row's race.

    Every row's race must be in ``counts``, as training rows are (``counts`` is built from
    them). The first race of all has no earlier races and gets NaN.
    """
    rates = counts[RACE_KEYS].assign(**{FIELD_RATE: rates_before(counts, window)})
    merged = rows[RACE_KEYS].merge(rates, on=RACE_KEYS, how="left", validate="many_to_one")
    return rows.assign(**{FIELD_RATE: merged[FIELD_RATE].to_numpy()})


def _prediction_rows(
    counts: pd.DataFrame | None, race: pd.DataFrame, window: int | None
) -> pd.DataFrame:
    """One race's rows with the field rate before it, checking the model was trained."""
    if counts is None:
        raise RuntimeError("Call fit() before predict().")
    require_one_race(race)
    season, round_number = (int(value) for value in race[RACE_KEYS].iloc[0])
    return race.assign(**{FIELD_RATE: rate_before(counts, season, round_number, window)})


# ---------------------------------------------------------------------------
# The models
# ---------------------------------------------------------------------------


@dataclass
class BaseRateDNF:
    """Every car gets the same chance: the DNF share of all starters over recent races.

    ``window`` is how many earlier races count (None: all of them). A short window follows
    a change in the field quickly but is noisy; a long one is steady but slow to notice a
    new era. The backtest chooses it on the 2024 validation season.
    """

    kind: ClassVar[str] = "base_rate"
    window: int | None = DEFAULT_WINDOW
    counts: pd.DataFrame | None = field(default=None, repr=False)

    def fit(self, history: pd.DataFrame) -> BaseRateDNF:
        """Count starts and retirements in every earlier race."""
        self.counts = race_counts(history)
        return self

    def predict(self, race: pd.DataFrame) -> pd.DataFrame:
        """``p_dnf`` for every car of one race: the field's DNF rate over the races before it."""
        rows = _prediction_rows(self.counts, race, self.window)
        return rows[KEYS].assign(p_dnf=rows[FIELD_RATE].to_numpy())


@dataclass
class LogisticDNF:
    """Logistic regression on the recent field rate and each car's own DNF inputs.

    Missing inputs (a rookie's first race, no qualifying time) are filled with the training
    median; every input is then standardized, so the coefficients are comparable. Fitting
    uses scikit-learn; the fitted numbers (fill values, scaling, coefficients) are kept
    as plain lists, so the model can be saved as JSON and predicted with NumPy alone.
    """

    kind: ClassVar[str] = "logistic"
    window: int | None = DEFAULT_WINDOW
    features: list[str] = field(default_factory=lambda: list(DNF_FEATURES))
    params: dict[str, Any] | None = field(default=None, repr=False)
    counts: pd.DataFrame | None = field(default=None, repr=False)

    def fit(self, history: pd.DataFrame) -> LogisticDNF:
        """Fit on every starter of every earlier race (1 = retired)."""
        self.counts = race_counts(history)
        rows = with_field_rate(starters(history), self.counts, self.window)
        pipeline = make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True),
            StandardScaler(),
            LogisticRegression(max_iter=1000),
        )
        pipeline.fit(rows[self.features], dnf_target(rows))
        imputer, scaler, logistic = (step for _, step in pipeline.steps)
        self.params = {
            "fill": imputer.statistics_.tolist(),
            "mean": scaler.mean_.tolist(),
            "scale": scaler.scale_.tolist(),
            "coef": logistic.coef_[0].tolist(),
            "intercept": float(logistic.intercept_[0]),
        }
        return self

    def predict(self, race: pd.DataFrame) -> pd.DataFrame:
        """``p_dnf`` for every car of one race."""
        if self.params is None:
            raise RuntimeError("Call fit() before predict().")
        rows = _prediction_rows(self.counts, race, self.window)
        values = rows[self.features].to_numpy("float64")
        filled = np.where(np.isnan(values), self.params["fill"], values)
        scaled = (filled - self.params["mean"]) / self.params["scale"]
        log_odds = scaled @ np.asarray(self.params["coef"]) + self.params["intercept"]
        return rows[KEYS].assign(p_dnf=expit(log_odds))

    def coefficients(self) -> pd.DataFrame:
        """Each input's coefficient: the change in log-odds of a DNF per standard deviation."""
        if self.params is None:
            raise RuntimeError("Call fit() before coefficients().")
        table = pd.DataFrame({"feature": self.features, "coef": self.params["coef"]})
        return table.sort_values("coef", key=np.abs, ascending=False, ignore_index=True)


@dataclass
class LightGBMDNF:
    """Gradient-boosted trees on the recent field rate and each car's own DNF inputs.

    A deliberately small model (50 trees of 3 leaves, at least 50 cars per leaf), the size
    the ranker's tuning settled on: with a signal this weak, a bigger model mostly learns
    noise. Its settings are fixed rather than tuned, because the DNF model itself is chosen
    on 2024 and tuning on the same season would tilt that choice toward LightGBM.
    """

    kind: ClassVar[str] = "lightgbm"
    window: int | None = DEFAULT_WINDOW
    features: list[str] = field(default_factory=lambda: list(DNF_FEATURES))
    booster: lgb.Booster | None = field(default=None, repr=False)
    counts: pd.DataFrame | None = field(default=None, repr=False)

    def fit(self, history: pd.DataFrame) -> LightGBMDNF:
        """Fit on every starter of every earlier race (1 = retired)."""
        self.counts = race_counts(history)
        rows = with_field_rate(starters(history), self.counts, self.window)
        classifier = lgb.LGBMClassifier(**LIGHTGBM_PARAMS)
        classifier.fit(rows[self.features], dnf_target(rows))
        self.booster = classifier.booster_
        return self

    def predict(self, race: pd.DataFrame) -> pd.DataFrame:
        """``p_dnf`` for every car of one race."""
        if self.booster is None:
            raise RuntimeError("Call fit() before predict().")
        rows = _prediction_rows(self.counts, race, self.window)
        chances = self.booster.predict(rows[self.features], validate_features=True)
        return rows[KEYS].assign(p_dnf=chances)

    def importance(self) -> pd.DataFrame:
        """Each input's share of the total gain: how much its splits improved the fit."""
        if self.booster is None:
            raise RuntimeError("Call fit() before importance().")
        gain = self.booster.feature_importance(importance_type="gain")
        share = gain / gain.sum() if gain.sum() > 0 else np.zeros(len(gain))
        table = pd.DataFrame({"feature": self.booster.feature_name(), "gain_share": share})
        return table.sort_values("gain_share", ascending=False, ignore_index=True)


DNFModel = BaseRateDNF | LogisticDNF | LightGBMDNF
MODEL_CLASSES: dict[str, type[DNFModel]] = {
    model.kind: model for model in (BaseRateDNF, LogisticDNF, LightGBMDNF)
}


# ---------------------------------------------------------------------------
# Lookup by name
# ---------------------------------------------------------------------------


def dnf_inputs(feature_set: str = "full") -> list[str]:
    """The learned DNF models' inputs that go with one of the ranker's feature sets.

    An ablation leaves feature groups out of the whole model, not just the ranker: with
    ``no_grid_quali`` the DNF models lose the grid slot, pit-lane start and qualifying gap
    too. The other sets keep every DNF input (none of them is a practice feature).
    """
    left_out = {
        column
        for group, columns in FEATURE_GROUPS.items()
        if group not in FEATURE_SETS[feature_set]
        for column in columns
    }
    return [column for column in DNF_FEATURES if column not in left_out]


def make_dnf_model(
    kind: str, window: int | None = DEFAULT_WINDOW, features: list[str] | None = None
) -> DNFModel:
    """A new, untrained DNF model by its name (``base_rate``, ``logistic`` or ``lightgbm``).

    ``features`` replaces the learned models' inputs (the base rate has none).
    """
    if kind not in MODEL_CLASSES:
        raise ValueError(f"Unknown DNF model {kind!r}; expected one of {DNF_KINDS}")
    if kind == BaseRateDNF.kind or features is None:
        return MODEL_CLASSES[kind](window=window)
    return MODEL_CLASSES[kind](window=window, features=list(features))


def window_label(window: int | None) -> str:
    """A look-back window in words, e.g. ``last 20 races`` or ``all earlier races``."""
    return "all earlier races" if window is None else f"last {window} races"
