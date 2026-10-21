"""Package API: ``f1predict.predict(season, round)`` predicts one race and returns a dict.

Example::

    from f1predict import predict

    result = predict(2026, 18)
    for driver in result["drivers"][:3]:
        print(driver["name"], driver["p_win"])

Two kinds of race can be predicted, and they are handled differently on purpose:

- **Upcoming** (qualified, not run yet): the models saved by ``train`` or ``update`` are used
  as they are. They must have learned from exactly the races before this one; if a result
  has come in (or changed) since they were trained, the call is refused and says to run
  ``update``.
- **Completed** (the result is in): a new ranker and DNF model are trained on the races
  before it only, as in the backtest, so a past race is never predicted by a model that
  saw its result. The recipe (the ranker's settings, the simulation temperature, the DNF
  model and its look-back window) comes from the saved models. It was chosen on the 2024
  validation season, so predictions for 2024 and earlier aren't fully out of sample.

A race that hasn't been qualified yet can't be predicted; the error says when qualifying
starts. The result holds only JSON types (str, int, float, bool, None, list, dict), so it
can be written with ``json.dumps`` or returned by a web endpoint as it is.
"""

from __future__ import annotations

import json
import logging
import operator
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from f1predict import __version__, config
from f1predict.data.tables import load_table
from f1predict.evaluate.backtest import CHANCE_COLUMNS, check_prediction
from f1predict.features.build import TARGET_COLUMNS, load_features
from f1predict.models import run_name
from f1predict.models.dnf import DNFModel, dnf_inputs, make_dnf_model
from f1predict.models.ranker import RankerModel
from f1predict.models.simulate import SimulationModel
from f1predict.models.store import (
    ModelFileError,
    load_dnf_model,
    load_ranker,
    race_label,
    training_summary,
)
from f1predict.override import GridOverride, apply_grid_override, parse_grid_override

logger = logging.getLogger(__name__)

FEATURE_SET = "full"
DIGITS = 4
KEYS = ["season", "round", "driver_id"]

# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class RaceUnavailableError(ValueError):
    """The race can't be predicted: not on the calendar, left out of the data, or first in it."""


class NotQualifiedError(RaceUnavailableError):
    """The race hasn't been qualified yet, or its qualifying isn't in the data yet.

    ``quali_start`` is when qualifying starts or started (UTC), if the calendar says.
    """

    def __init__(self, message: str, quali_start: datetime | None = None) -> None:
        super().__init__(message)
        self.quali_start = quali_start


class ModelsOutOfDateError(ValueError):
    """The saved models didn't learn from exactly the races before the race asked for."""


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RaceData:
    """What a prediction reads: the features table, the calendar and the race results."""

    features: pd.DataFrame
    schedule: pd.DataFrame
    results: pd.DataFrame


def load_race_data() -> RaceData:
    """Read the features table and the calendar and results tables from the data folder."""
    return RaceData(load_features(), load_table("schedule"), load_table("results"))


# ---------------------------------------------------------------------------
# Predicting
# ---------------------------------------------------------------------------


def predict(
    season: int,
    round: int,
    sims: int = config.DEFAULT_SIMS,
    seed: int = config.DEFAULT_SEED,
    grid_override: GridOverride | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Predict one race: the finishing order and each driver's win, podium, points and DNF chances.

    ``sims`` races are simulated (100,000 to 1,000,000) from random seed ``seed``; the same
    seed always gives the same result. ``grid_override`` replaces the starting grid, e.g.
    after penalties: a ``GridOverride``, or a dict shaped like an override file (see
    ``f1predict.override``).

    The result has five parts:

    - ``race``: ``season``, ``round``, ``name``, ``circuit_id``, ``circuit``, ``locality``,
      ``country``, ``qualifying_start`` and ``race_start`` (UTC, ISO 8601),
      ``sprint_weekend``, and ``status``: ``"upcoming"`` or ``"completed"``.
    - ``model``: ``source`` (``"saved"`` models, or ``"retrained"`` on the earlier races),
      ``trained_on`` (races, first and last), ``ranker_settings``, ``temperature``,
      ``dnf_model`` and ``dnf_window``, ``settings_chosen_on``, ``sims`` and ``seed``.
    - ``grid_override``: the override applied (``grid`` and ``pitlane``), or None.
    - ``drivers``: one entry per driver in predicted order: ``predicted_rank``,
      ``driver_id``, ``code``, ``name``, ``constructor_id``, ``team``, ``grid`` (None for a
      pit-lane start), ``pitlane_start``, ``p_win``, ``p_podium``, ``p_points`` and
      ``p_dnf`` (chances from 0 to 1, to 4 decimals), and ``actual``: None for an upcoming
      race, else ``position``, ``position_text``, ``status`` and ``status_class``.
    - ``f1predict_version``.

    Raises ``ValueError`` for ``sims`` or ``seed`` out of range, ``RaceUnavailableError``
    (``NotQualifiedError`` before qualifying), ``ModelsOutOfDateError``,
    ``GridOverrideError``, ``ModelFileError``, or ``FileNotFoundError`` when the data or the
    models haven't been built yet.
    """
    season, round_number = operator.index(season), operator.index(round)
    sims, seed = _check_simulation(sims, seed)
    override = None if grid_override is None else parse_grid_override(grid_override)
    return predict_race(
        load_race_data(), season, round_number, sims=sims, seed=seed, grid_override=override
    )


def predict_race(
    data: RaceData,
    season: int,
    round_number: int,
    *,
    sims: int = config.DEFAULT_SIMS,
    seed: int = config.DEFAULT_SEED,
    grid_override: GridOverride | Mapping[str, Any] | None = None,
    models_dir: Path | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """``predict`` on data already in memory; ``models_dir`` and ``now`` (UTC) are for tests."""
    season, round_number = operator.index(season), operator.index(round_number)
    sims, seed = _check_simulation(sims, seed)
    override = None if grid_override is None else parse_grid_override(grid_override)
    entry = _calendar_entry(data.schedule, season, round_number)
    label = race_label(season, round_number)

    features = data.features
    rows = features[(features["season"] == season) & (features["round"] == round_number)]
    if rows.empty:
        raise _not_qualified(label, season, entry["quali_start"], now or datetime.now(UTC))
    known = features["status_class"].notna()
    history = features[(features["race_seq"] < rows["race_seq"].iloc[0]) & known]
    if history.empty:
        raise RaceUnavailableError(
            f"{label} is the first race in the data, so there are no earlier races to learn from."
        )

    completed = bool(rows["status_class"].notna().all())
    race = rows.drop(columns=TARGET_COLUMNS)
    if override is not None:
        race = apply_grid_override(race, override)
    build = _retrained_model if completed else _saved_model
    model, about = build(history, sims, seed, models_dir)
    logger.debug(
        "Predicting %s with %s models (%s races up to %s)",
        label,
        about["source"],
        about["trained_on"]["races"],
        about["trained_on"]["last_race"],
    )
    predicted = model.predict(race)
    check_prediction(predicted, race, label)
    return {
        "f1predict_version": __version__,
        "race": _race_info(entry, completed),
        "model": about,
        "grid_override": _override_info(override),
        "drivers": _driver_entries(race, predicted, data.results, completed),
    }


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------


def _check_simulation(sims: int, seed: int) -> tuple[int, int]:
    """Refuse a number of simulated races out of range or a negative seed; return both."""
    sims, seed = operator.index(sims), operator.index(seed)
    if not config.MIN_SIMS <= sims <= config.MAX_SIMS:
        raise ValueError(
            f"sims must be from {config.MIN_SIMS:,} to {config.MAX_SIMS:,}, not {sims:,}"
        )
    if seed < 0:
        raise ValueError(f"seed must be 0 or more, not {seed}")
    return sims, seed


def _calendar_entry(schedule: pd.DataFrame, season: int, round_number: int) -> pd.Series:
    """The race's row in the calendar, or an error saying why it can't be predicted."""
    if not config.FIRST_SEASON <= season <= config.LAST_SEASON:
        raise RaceUnavailableError(
            f"There's no data for {season}: the seasons are {config.FIRST_SEASON} to "
            f"{config.LAST_SEASON}."
        )
    calendar = schedule[schedule["season"] == season]
    if calendar.empty:
        raise RaceUnavailableError(
            f"The {season} calendar isn't in the data. Run "
            f"`python -m f1predict fetch --from {season} --to {season}` first."
        )
    entry = calendar[calendar["round"] == round_number]
    if entry.empty:
        raise RaceUnavailableError(
            f"{season} has rounds 1 to {int(calendar['round'].max())}; "
            f"there's no round {round_number}."
        )
    reason = config.EXCLUDED_RACES.get((season, round_number))
    if reason:
        raise RaceUnavailableError(
            f"{race_label(season, round_number)} is left out of the data ({reason})."
        )
    return entry.iloc[0]


def session_time(moment: datetime | pd.Timestamp) -> str:
    """A session's start in words, e.g. ``Sat 24 Oct 2026, 21:00 UTC``."""
    return pd.Timestamp(moment).tz_convert("UTC").strftime("%a %d %b %Y, %H:%M UTC")


def _not_qualified(label: str, season: int, quali_start: Any, now: datetime) -> NotQualifiedError:
    """The error for a race with no qualifying in the data: still to come, or not downloaded."""
    start = None if pd.isna(quali_start) else pd.Timestamp(quali_start).to_pydatetime()
    refresh = (
        "`python -m f1predict update`"
        if season == config.LAST_SEASON
        else f"`python -m f1predict fetch --from {season} --to {season}` and then "
        "`python -m f1predict build-features`"
    )
    if start is not None and start > now:
        message = (
            f"{label} hasn't been qualified yet: qualifying starts {session_time(start)}. "
            f"Predictions are made after qualifying; run {refresh} once it's over."
        )
    elif start is not None:
        message = (
            f"{label} was qualified on {session_time(start)}, but its qualifying isn't in the "
            f"data yet. Run {refresh} to download it (results can take a few hours to appear)."
        )
    else:
        message = f"{label} has no qualifying in the data. Run {refresh} after qualifying."
    return NotQualifiedError(message, start)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


def _temperature(description: dict[str, Any]) -> float:
    """The simulation temperature saved with the ranker."""
    temperature = description.get("temperature")
    if not temperature:
        raise ModelFileError(
            "The saved ranker has no simulation temperature. Run `python -m f1predict train`."
        )
    return float(temperature)


def _saved_model(
    history: pd.DataFrame, sims: int, seed: int, folder: Path | None
) -> tuple[SimulationModel, dict[str, Any]]:
    """The models saved by ``train``, after checking they learned from exactly ``history``."""
    ranker, description = load_ranker(run_name("ranker", FEATURE_SET), folder)
    dnf, dnf_description = load_dnf_model(run_name("dnf", FEATURE_SET), folder)
    temperature = _temperature(description)
    needed = training_summary(history)
    saved = (description["trained_on"], dnf_description["trained_on"])
    if saved != needed:
        raise ModelsOutOfDateError(_out_of_date(saved[0], needed[0]))
    model = SimulationModel(ranker, temperature, sims, seed, dnf)
    return model, _model_info("saved", description, dnf, needed[0], sims, seed)


def _out_of_date(saved: dict[str, Any], needed: dict[str, Any]) -> str:
    """Why the saved models can't predict this race, and the command that fixes it."""
    fix = "Run `python -m f1predict update` (or `train`) to retrain them."
    if (saved.get("races"), saved.get("last_race")) == (needed["races"], needed["last_race"]):
        return (
            f"The saved models learned from the {needed['races']} races up to "
            f"{needed['last_race']}, but a result has changed since. {fix}"
        )
    return (
        f"The saved models learned from {saved.get('races')} races up to "
        f"{saved.get('last_race')}, but this race needs models trained on the "
        f"{needed['races']} races before it, up to {needed['last_race']}. {fix}"
    )


def _retrained_model(
    history: pd.DataFrame, sims: int, seed: int, folder: Path | None
) -> tuple[SimulationModel, dict[str, Any]]:
    """A ranker and DNF model trained on ``history`` only, to the saved models' recipe."""
    _, description = load_ranker(run_name("ranker", FEATURE_SET), folder)
    saved_dnf, _ = load_dnf_model(run_name("dnf", FEATURE_SET), folder)
    ranker = RankerModel(feature_set=FEATURE_SET, **description["settings"])
    dnf = make_dnf_model(saved_dnf.kind, saved_dnf.window, dnf_inputs(FEATURE_SET))
    model = SimulationModel(ranker, _temperature(description), sims, seed, dnf).fit(history)
    trained_on, _ = training_summary(history)
    return model, _model_info("retrained", description, dnf, trained_on, sims, seed)


def _model_info(
    source: str,
    description: dict[str, Any],
    dnf: DNFModel,
    trained_on: dict[str, Any],
    sims: int,
    seed: int,
) -> dict[str, Any]:
    """How the prediction was made: the result's ``model`` part."""
    return {
        "source": source,
        "feature_set": FEATURE_SET,
        "trained_on": dict(trained_on),
        "ranker_settings": dict(description["settings"]),
        "temperature": round(float(description["temperature"]), DIGITS),
        "dnf_model": dnf.kind,
        "dnf_window": dnf.window,
        "settings_chosen_on": config.VALIDATION_SEASON,
        "sims": sims,
        "seed": seed,
    }


# ---------------------------------------------------------------------------
# The result
# ---------------------------------------------------------------------------


def _text(value: Any) -> str | None:
    """A table cell as text, or None when it's missing."""
    return None if pd.isna(value) else str(value)


def _number(value: Any) -> int | None:
    """A table cell as a whole number, or None when it's missing."""
    return None if pd.isna(value) else int(value)


def _iso(value: Any) -> str | None:
    """A UTC timestamp as ISO 8601 text, e.g. ``2026-10-25T20:00:00Z``."""
    return None if pd.isna(value) else pd.Timestamp(value).strftime("%Y-%m-%dT%H:%M:%SZ")


def _race_info(entry: pd.Series, completed: bool) -> dict[str, Any]:
    """The race's calendar details and whether it has been run."""
    return {
        "season": int(entry["season"]),
        "round": int(entry["round"]),
        "name": _text(entry["race_name"]),
        "circuit_id": _text(entry["circuit_id"]),
        "circuit": _text(entry["circuit_name"]),
        "locality": _text(entry["locality"]),
        "country": _text(entry["country"]),
        "qualifying_start": _iso(entry["quali_start"]),
        "race_start": _iso(entry["race_start"]),
        "sprint_weekend": bool(entry["is_sprint_weekend"]),
        "status": "completed" if completed else "upcoming",
    }


def _override_info(override: GridOverride | None) -> dict[str, Any] | None:
    """The applied override as plain JSON types."""
    if override is None:
        return None
    return {"grid": dict(override.grid), "pitlane": list(override.pitlane)}


def _latest(results: pd.DataFrame, key: str, value: str) -> dict[str, str]:
    """The most recent ``value`` for each ``key`` in the results, e.g. each driver's name."""
    known = results.dropna(subset=[key, value]).drop_duplicates(key, keep="last")
    return {str(k): str(v) for k, v in zip(known[key], known[value], strict=True)}


def _actual(results: pd.DataFrame, season: int, round_number: int) -> dict[str, dict[str, Any]]:
    """What happened to each driver in a race that has been run."""
    rows = results[(results["season"] == season) & (results["round"] == round_number)]
    return {
        str(row.driver_id): {
            "position": _number(row.position),
            "position_text": _text(row.position_text),
            "status": _text(row.status),
            "status_class": _text(row.status_class),
        }
        for row in rows.itertuples(index=False)
    }


def _driver_entries(
    race: pd.DataFrame, predicted: pd.DataFrame, results: pd.DataFrame, completed: bool
) -> list[dict[str, Any]]:
    """One entry per driver, in predicted order (the fields are listed in ``predict``)."""
    season, round_number = (int(value) for value in race[["season", "round"]].iloc[0])
    names = _latest(results, "driver_id", "driver_name")
    teams = _latest(results, "constructor_id", "constructor_name")
    actual = _actual(results, season, round_number) if completed else {}
    table = race[[*KEYS, "driver_code", "constructor_id", "grid", "pitlane_start"]].merge(
        predicted[[*KEYS, "pred_rank", *CHANCE_COLUMNS]], on=KEYS
    )
    entries = []
    for row in table.sort_values("pred_rank").itertuples(index=False):
        driver, constructor = str(row.driver_id), _text(row.constructor_id)
        pitlane = bool(row.pitlane_start == 1)
        chances = {column: round(float(getattr(row, column)), DIGITS) for column in CHANCE_COLUMNS}
        entries.append(
            {
                "predicted_rank": int(row.pred_rank),
                "driver_id": driver,
                "code": _text(row.driver_code),
                "name": names.get(driver),
                "constructor_id": constructor,
                "team": teams.get(constructor) if constructor else None,
                "grid": None if pitlane or pd.isna(row.grid) else int(row.grid),
                "pitlane_start": pitlane,
                **chances,
                "actual": actual.get(driver) if completed else None,
            }
        )
    return entries


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------


def prediction_path(
    season: int, round_number: int, override: bool = False, folder: Path | None = None
) -> Path:
    """Where a prediction is saved, e.g. ``predictions/2026_r18.json`` (``_override`` added)."""
    root = folder or config.PROCESSED_DIR / "predictions"
    return root / f"{season}_r{round_number:02d}{'_override' if override else ''}.json"


def save_prediction(result: dict[str, Any], folder: Path | None = None) -> Path:
    """Write a prediction as JSON and return the path; one with an override gets its own file."""
    race = result["race"]
    path = prediction_path(
        race["season"], race["round"], result["grid_override"] is not None, folder
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(result, indent=2, allow_nan=False) + "\n"
    path.write_text(text, encoding="utf-8", newline="\n")
    return path
