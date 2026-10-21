"""Tests for the package API: predicting upcoming and completed races, and every refusal.

The synthetic championship (2023-2024, five rounds each) is saved to a temporary data folder
with its last race, 2024 round 5, turned into an upcoming race: qualifying is in, the result
isn't. A ranker and a DNF model are trained on the nine races before it and saved the way
``train`` saves them.

The key test ties the two kinds of prediction together. Predicting round 5 as an upcoming
race (saved models), with an override giving the grid it was actually run from, must match
predicting it as a completed race (models retrained on the races before it) exactly.
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import replace
from datetime import UTC, datetime
from functools import partial

import pandas as pd
import pytest
from synthetic import make_tables, scramble_after

import f1predict
from f1predict import config, predict
from f1predict.api import (
    ModelsOutOfDateError,
    NotQualifiedError,
    RaceData,
    RaceUnavailableError,
    predict_race,
    save_prediction,
)
from f1predict.data.tables import save_season_tables
from f1predict.evaluate.backtest import predict_races
from f1predict.features.build import RaceTables, build_features, save_features
from f1predict.models import make_model
from f1predict.models.dnf import dnf_inputs, make_dnf_model
from f1predict.models.ranker import RankerModel
from f1predict.models.store import (
    ModelFileError,
    save_dnf_model,
    save_ranker,
    training_summary,
)
from f1predict.override import GridOverride, GridOverrideError

TEMPERATURE = 0.8
DNF_WINDOW = 5
CHANCES = ["p_win", "p_podium", "p_points", "p_dnf"]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def without_result(tables: RaceTables, season: int, round_number: int) -> RaceTables:
    """The same championship with one race's result removed (its qualifying stays)."""
    results = tables.results
    keep = ~((results["season"] == season) & (results["round"] == round_number))
    return replace(tables, results=results[keep].reset_index(drop=True))


def train_and_save(features: pd.DataFrame, folder, through: tuple[int, int] | None = None) -> None:
    """Fit both models on the races run (up to ``through``) and save them as ``train`` does."""
    history = features[features["status_class"].notna()]
    if through is not None:
        keys = zip(history["season"], history["round"], strict=True)
        history = history[[key <= through for key in keys]]
    ranker = RankerModel().fit(history)
    dnf = make_dnf_model("logistic", DNF_WINDOW, dnf_inputs("full")).fit(history)
    ranker_on, dnf_on = training_summary(history)
    save_ranker(ranker, "ranker_full", ranker_on, folder, temperature=TEMPERATURE)
    save_dnf_model(dnf, "dnf_full", dnf_on, folder)


def race_data(tables: RaceTables) -> RaceData:
    """What ``predict`` reads from disk, built in memory from synthetic tables."""
    return RaceData(build_features(tables), tables.schedule, tables.results)


def chances(result: dict) -> list[tuple]:
    """Each driver's predicted rank, id, grid and chances: everything but the actual result."""
    return [
        (d["predicted_rank"], d["driver_id"], d["grid"], *(d[c] for c in CHANCES))
        for d in result["drivers"]
    ]


@pytest.fixture(scope="module")
def full() -> RaceTables:
    """The whole synthetic championship, every race run."""
    return make_tables()


@pytest.fixture(scope="module")
def workspace(tmp_path_factory, full):
    """Data and trained models in a temporary folder, with 2024 round 5 still to be run."""
    root = tmp_path_factory.mktemp("api")
    tables = without_result(full, 2024, 5)
    with pytest.MonkeyPatch.context() as patch:
        for name, folder in (
            ("INTERIM_DIR", "interim"),
            ("PROCESSED_DIR", "processed"),
            ("MODELS_DIR", "models"),
        ):
            patch.setattr(config, name, root / folder)
        for season in (2023, 2024):
            frames = {
                name: frame[frame["season"] == season] for name, frame in vars(tables).items()
            }
            save_season_tables(frames, season)
        features = build_features(tables)
        save_features(features)
        train_and_save(features, root / "models")
        yield root


# ---------------------------------------------------------------------------
# An upcoming race
# ---------------------------------------------------------------------------


def test_an_upcoming_race_uses_the_saved_models(workspace) -> None:
    """Round 5 has qualifying but no result: the models saved after round 4 predict it."""
    result = predict(2024, 5)
    assert result["race"]["status"] == "upcoming"
    assert result["race"]["name"] == "Race 5"
    assert result["model"]["source"] == "saved"
    assert result["model"]["trained_on"]["races"] == 9
    assert result["model"]["trained_on"]["last_race"] == "2024 round 4"
    assert result["model"]["temperature"] == TEMPERATURE
    assert result["model"]["dnf_model"] == "logistic"
    assert result["grid_override"] is None
    assert all(driver["actual"] is None for driver in result["drivers"])


def test_the_result_holds_every_field_and_only_json_types(workspace) -> None:
    """Writing and reading it back as strict JSON (no NaN) gives the same dict."""
    result = predict(2024, 5)
    assert set(result) == {"f1predict_version", "race", "model", "grid_override", "drivers"}
    assert set(result["drivers"][0]) == {
        "predicted_rank",
        "driver_id",
        "code",
        "name",
        "constructor_id",
        "team",
        "grid",
        "pitlane_start",
        *CHANCES,
        "actual",
    }
    assert json.loads(json.dumps(result, allow_nan=False)) == result
    assert result["race"]["race_start"] == "2024-05-10T15:00:00Z"


def test_the_chances_add_up_like_a_race(workspace) -> None:
    """One winner, three podium places, ranks 1..n, and win <= podium <= points per car."""
    drivers = predict(2024, 5)["drivers"]
    assert [d["predicted_rank"] for d in drivers] == list(range(1, 7))
    assert sum(d["p_win"] for d in drivers) == pytest.approx(1, abs=0.001)
    assert sum(d["p_podium"] for d in drivers) == pytest.approx(3, abs=0.001)
    for d in drivers:
        assert 0 <= d["p_win"] <= d["p_podium"] <= d["p_points"] <= 1
        assert 0 < d["p_dnf"] < 1


def test_the_same_seed_gives_the_same_prediction(workspace) -> None:
    """Determinism: same seed, identical dict; another seed moves the chances a little."""
    assert predict(2024, 5) == predict(2024, 5)
    assert chances(predict(2024, 5, seed=7)) != chances(predict(2024, 5))


def test_a_grid_override_changes_the_grid_and_is_reported(workspace) -> None:
    """Sending the favourite to the pit lane costs them win chance; dict and model agree."""
    before = predict(2024, 5)
    favourite = before["drivers"][0]["driver_id"]
    after = predict(2024, 5, grid_override={"pitlane": [favourite]})
    moved = next(d for d in after["drivers"] if d["driver_id"] == favourite)
    assert moved["pitlane_start"] is True
    assert moved["grid"] is None
    assert moved["p_win"] < before["drivers"][0]["p_win"]
    assert after["grid_override"] == {"grid": {}, "pitlane": [favourite]}
    assert predict(2024, 5, grid_override=GridOverride(pitlane=[favourite])) == after


def test_a_bad_override_is_refused(workspace) -> None:
    """A driver code instead of an id points at the right id."""
    with pytest.raises(GridOverrideError, match="did you mean 'hamilton'"):
        predict(2024, 5, grid_override={"grid": {"HAM": 1}})


# ---------------------------------------------------------------------------
# A completed race
# ---------------------------------------------------------------------------


def test_a_completed_race_is_predicted_by_models_trained_on_earlier_races(workspace) -> None:
    """Round 4: retrained on the 8 races before it, and shown next to what happened."""
    result = predict(2024, 4)
    assert result["race"]["status"] == "completed"
    assert result["model"]["source"] == "retrained"
    assert result["model"]["trained_on"]["races"] == 8
    assert result["model"]["trained_on"]["last_race"] == "2024 round 3"
    actual = [d["actual"] for d in result["drivers"]]
    assert {a["position"] for a in actual} == set(range(1, 7))
    assert {a["status_class"] for a in actual} <= {"finished", "dnf", "dns"}


def test_upcoming_with_the_actual_grid_matches_the_completed_prediction(workspace, full) -> None:
    """The two paths agree: saved models + the real grid == retrained models after the race."""
    results = full.results[(full.results["season"] == 2024) & (full.results["round"] == 5)]
    grid = {str(d): int(g) for d, g in zip(results["driver_id"], results["grid"], strict=True)}
    upcoming = predict(2024, 5, grid_override={"grid": grid})
    completed = predict_race(race_data(full), 2024, 5, models_dir=workspace / "models")
    assert completed["race"]["status"] == "completed"
    assert chances(upcoming) == chances(completed)
    assert upcoming["model"]["trained_on"] == completed["model"]["trained_on"]


def test_the_api_matches_the_backtest_for_a_completed_race(workspace, full) -> None:
    """Same recipe, same races learned from, same seed: the backtest's chances to 4 decimals."""
    features = build_features(full)
    make_dnf = partial(make_dnf_model, "logistic", DNF_WINDOW, dnf_inputs("full"))
    backtest = predict_races(
        partial(make_model, "ranker", "full", TEMPERATURE, make_dnf), features, [2024]
    )
    expected = backtest[backtest["round"] == 4].sort_values("pred_rank")
    result = predict_race(race_data(full), 2024, 4, models_dir=workspace / "models")
    assert [d["driver_id"] for d in result["drivers"]] == expected["driver_id"].tolist()
    for column in CHANCES:
        rounded = [round(float(value), 4) for value in expected[column]]
        assert [d[column] for d in result["drivers"]] == rounded


def test_a_past_prediction_never_sees_its_own_or_later_results(workspace, full) -> None:
    """Scrambling the results of 2024 round 2 onwards leaves round 2's prediction unchanged."""
    scrambled = scramble_after(full, 2024, 1)
    honest = predict_race(race_data(full), 2024, 2, models_dir=workspace / "models")
    mixed = predict_race(race_data(scrambled), 2024, 2, models_dir=workspace / "models")
    assert chances(mixed) == chances(honest)
    assert [d["actual"] for d in mixed["drivers"]] != [d["actual"] for d in honest["drivers"]]


# ---------------------------------------------------------------------------
# Races that can't be predicted
# ---------------------------------------------------------------------------


@pytest.fixture
def unqualified(full) -> RaceData:
    """2024 round 5 on the calendar with neither qualifying nor a result."""
    data = race_data(without_result(full, 2024, 5))
    rows = (data.features["season"] == 2024) & (data.features["round"] == 5)
    return replace(data, features=data.features[~rows])


def test_before_qualifying_the_error_says_when_it_starts(workspace, unqualified) -> None:
    """Asked before Thursday's qualifying: the message gives its time and what to run after."""
    with pytest.raises(NotQualifiedError, match="starts Thu 09 May 2024, 15:00 UTC") as error:
        predict_race(unqualified, 2024, 5, now=datetime(2024, 5, 1, tzinfo=UTC))
    assert error.value.quali_start == datetime(2024, 5, 9, 15, tzinfo=UTC)


def test_after_qualifying_with_no_data_the_error_says_to_download_it(
    workspace, unqualified, monkeypatch
) -> None:
    """Qualifying is over but not downloaded: `fetch` for an old season, `update` for this one."""
    later = datetime(2024, 6, 1, tzinfo=UTC)
    with pytest.raises(NotQualifiedError, match="fetch --from 2024 --to 2024"):
        predict_race(unqualified, 2024, 5, now=later)
    monkeypatch.setattr(config, "LAST_SEASON", 2024)
    with pytest.raises(NotQualifiedError) as error:
        predict_race(unqualified, 2024, 5, now=later)
    assert "isn't in the data yet. Run `python -m f1predict update`" in str(error.value)


@pytest.mark.parametrize(
    ("season", "round_number", "message"),
    [
        (2024, 9, "2024 has rounds 1 to 5; there's no round 9"),
        (2017, 1, "There's no data for 2017"),
        (2025, 1, "The 2025 calendar isn't in the data"),
        (2023, 1, "first race in the data"),
    ],
)
def test_races_that_cant_be_predicted_are_refused(workspace, season, round_number, message):
    """Not on the calendar, out of range, never downloaded, or nothing earlier to learn from."""
    with pytest.raises(RaceUnavailableError, match=message):
        predict(season, round_number)


def test_an_excluded_race_is_refused_with_the_reason(workspace, monkeypatch) -> None:
    """A race left out of the data (like Spa 2021) can't be predicted either."""
    monkeypatch.setattr(config, "EXCLUDED_RACES", {(2024, 2): "a test reason"})
    with pytest.raises(RaceUnavailableError, match=r"left out of the data \(a test reason\)"):
        predict(2024, 2)


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    [
        ({"sims": 10}, ValueError, "sims must be from 100,000 to 1,000,000"),
        ({"sims": 2_000_000}, ValueError, "sims must be from"),
        ({"seed": -1}, ValueError, "seed must be 0 or more"),
        ({"round": "5"}, TypeError, "integer"),
    ],
)
def test_bad_arguments_are_refused_before_any_work(kwargs, error, message) -> None:
    """Simulation settings and argument types are checked first."""
    arguments = {"season": 2024, "round": 5} | kwargs
    with pytest.raises(error, match=message):
        predict(**arguments)


# ---------------------------------------------------------------------------
# Saved models that can't be used
# ---------------------------------------------------------------------------


def test_models_trained_before_the_latest_result_are_refused(workspace, tmp_path) -> None:
    """Saved after round 3 but round 4 has a result now: refused, with the command to run."""
    data = race_data(without_result(make_tables(), 2024, 5))
    train_and_save(data.features, tmp_path, through=(2024, 3))
    with pytest.raises(ModelsOutOfDateError, match="8 races up to 2024 round 3") as error:
        predict_race(data, 2024, 5, models_dir=tmp_path)
    assert "python -m f1predict update" in str(error.value)


def test_models_trained_on_a_result_that_has_changed_are_refused(workspace, tmp_path) -> None:
    """Same races, but a finisher has since been reclassified: the models are out of date."""
    tables = without_result(make_tables(), 2024, 5)
    features = build_features(tables)
    changed = features.copy()
    row = changed.index[(changed["round"] == 4) & changed["finished"].fillna(False)][0]
    changed.loc[row, ["status_class", "finished", "finish_rank"]] = ["dnf", False, float("nan")]
    train_and_save(changed, tmp_path)
    data = RaceData(features, tables.schedule, tables.results)
    with pytest.raises(ModelsOutOfDateError, match="a result has changed since"):
        predict_race(data, 2024, 5, models_dir=tmp_path)


def test_missing_models_say_how_to_make_them(workspace, tmp_path) -> None:
    """No saved models: both kinds of race point to `train` (they need its recipe)."""
    for round_number in (4, 5):
        with pytest.raises(FileNotFoundError, match="f1predict train"):
            predict_race(
                race_data(without_result(make_tables(), 2024, 5)),
                2024,
                round_number,
                models_dir=tmp_path,
            )


def test_a_ranker_saved_without_a_temperature_is_refused(workspace, tmp_path) -> None:
    """The temperature is part of the recipe; without it there are no chances to give."""
    tables = without_result(make_tables(), 2024, 5)
    features = build_features(tables)
    train_and_save(features, tmp_path)
    history = features[features["status_class"].notna()]
    save_ranker(RankerModel().fit(history), "ranker_full", training_summary(history)[0], tmp_path)
    with pytest.raises(ModelFileError, match="temperature"):
        predict_race(race_data(tables), 2024, 5, models_dir=tmp_path)


# ---------------------------------------------------------------------------
# Saving and importing
# ---------------------------------------------------------------------------


def test_predictions_are_saved_as_json_named_after_the_race(workspace, tmp_path) -> None:
    """``2024_r05.json``, and ``_override`` when the grid was changed; LF line endings."""
    plain = save_prediction(predict(2024, 5), tmp_path)
    changed = save_prediction(predict(2024, 5, grid_override={"pitlane": ["zhou"]}), tmp_path)
    assert (plain.name, changed.name) == ("2024_r05.json", "2024_r05_override.json")
    assert json.loads(plain.read_text(encoding="utf-8")) == predict(2024, 5)
    assert b"\r\n" not in plain.read_bytes()


def test_from_f1predict_import_predict_gives_the_api() -> None:
    """``f1predict.predict`` is the API function; other missing names still raise."""
    from f1predict import api

    assert f1predict.predict is api.predict
    with pytest.raises(AttributeError, match="no attribute 'nothing_here'"):
        _ = f1predict.nothing_here


def test_importing_the_package_stays_fast() -> None:
    """``import f1predict`` loads neither pandas nor LightGBM until ``predict`` is used."""
    code = (
        "import sys, f1predict; "
        "print('pandas' in sys.modules, 'lightgbm' in sys.modules); "
        "f1predict.predict; "
        "print('pandas' in sys.modules)"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=120, check=True
    )
    assert completed.stdout.split() == ["False", "False", "True"]
