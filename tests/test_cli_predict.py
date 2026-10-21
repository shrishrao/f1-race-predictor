"""Tests for the ``predict`` and ``update`` commands, on the synthetic championship.

``predict`` runs against a temporary data folder where 2024 round 5 has been qualified but
not run, with models saved by the real ``train`` command. ``update`` gets a fake download
that saves a prepared version of the 2024 tables the way ``fetch`` would, so its report of
what's new (qualifying, a result, a result that changed) can be checked without a network.
"""

from __future__ import annotations

import json
import logging
from dataclasses import replace

import pytest
from synthetic import make_tables
from typer.testing import CliRunner

from f1predict import config, predict
from f1predict.__main__ import app
from f1predict.data import fetch as fetch_module
from f1predict.data.fetch import PracticeSummary, SeasonSummary
from f1predict.data.http import ApiError
from f1predict.data.tables import save_season_tables
from f1predict.features.build import RaceTables, build_features, save_features
from f1predict.log import LOGGER_NAME
from f1predict.models.store import load_ranker

runner = CliRunner()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def without_race(
    tables: RaceTables, season: int, round_number: int, keep_qualifying: bool = True
) -> RaceTables:
    """The championship with one race's result removed, and its qualifying too if asked."""

    def drop(frame):
        """``frame`` without the race's rows."""
        return frame[~((frame["season"] == season) & (frame["round"] == round_number))]

    qualifying = tables.qualifying if keep_qualifying else drop(tables.qualifying)
    return replace(tables, results=drop(tables.results), qualifying=qualifying)


def save_tables(tables: RaceTables, seasons) -> None:
    """Write each season's interim tables, as ``fetch`` does."""
    for season in seasons:
        frames = {name: frame[frame["season"] == season] for name, frame in vars(tables).items()}
        save_season_tables(frames, season)


def words(output: str) -> str:
    """Command output with line breaks and runs of spaces collapsed, for matching phrases."""
    return " ".join(output.split())


def close_log_files() -> None:
    """Close the log files commands opened (Windows can't delete open files)."""
    logger = logging.getLogger(LOGGER_NAME)
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()


# ---------------------------------------------------------------------------
# predict
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def workspace(tmp_path_factory):
    """Data with 2024 round 5 qualified but not run, and models from the `train` command."""
    root = tmp_path_factory.mktemp("cli_predict")
    with pytest.MonkeyPatch.context() as patch:
        for name, folder in (
            ("INTERIM_DIR", "interim"),
            ("PROCESSED_DIR", "processed"),
            ("MODELS_DIR", "models"),
            ("LOG_DIR", "logs"),
        ):
            patch.setattr(config, name, root / folder)
        tables = without_race(make_tables(), 2024, 5)
        save_tables(tables, (2023, 2024))
        save_features(build_features(tables))
        trained = runner.invoke(app, ["train"])
        assert trained.exit_code == 0, trained.output
        yield root
    close_log_files()


def test_predict_prints_the_table_and_saves_the_json(workspace) -> None:
    """An upcoming race: the table, how it was made, and the same dict as the API, saved."""
    result = runner.invoke(app, ["predict", "--season", "2024", "--round", "5"])
    assert result.exit_code == 0, result.output
    text = words(result.output)
    assert "2024 round 5: Race 5" in text
    assert "Not run yet: predicted by the saved models (9 races" in text
    assert "100,000 simulated races (seed 42)" in text
    path = workspace / "processed" / "predictions" / "2024_r05.json"
    assert json.loads(path.read_text(encoding="utf-8")) == predict(2024, 5)


def test_predict_a_race_already_run_shows_how_it_went(workspace) -> None:
    """A completed race: retrained on the races before it, with the result and the winner."""
    result = runner.invoke(app, ["predict", "--season", "2024", "--round", "4"])
    assert result.exit_code == 0, result.output
    text = words(result.output)
    assert "Result" in text
    assert "Already run: predicted by models trained only on the races before it" in text
    assert "Winner:" in text
    assert "aren't fully out of sample" in text


def test_predict_with_a_grid_override_file(workspace, tmp_path) -> None:
    """A pit-lane start from an override file is marked and saved to its own JSON file."""
    path = tmp_path / "penalties.json"
    path.write_text(json.dumps({"season": 2024, "round": 5, "pitlane": ["hamilton"]}))
    result = runner.invoke(
        app, ["predict", "--season", "2024", "--round", "5", "--grid-override", str(path)]
    )
    assert result.exit_code == 0, result.output
    text = words(result.output)
    assert "PL*" in text
    assert "PL: starts from the pit lane." in text
    assert "Starting slot set by the grid override." in text
    saved = workspace / "processed" / "predictions" / "2024_r05_override.json"
    assert json.loads(saved.read_text(encoding="utf-8"))["grid_override"] == {
        "grid": {},
        "pitlane": ["hamilton"],
    }


def test_predict_explains_what_it_cant_do(workspace, tmp_path) -> None:
    """A round that doesn't exist, or a driver code in an override: exit 1 and the reason."""
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"grid": {"HAM": 1}}))
    for args, reason in (
        (["--round", "6"], "2024 has rounds 1 to 5"),
        (["--round", "5", "--grid-override", str(bad)], "did you mean 'hamilton'?"),
    ):
        result = runner.invoke(app, ["predict", "--season", "2024", *args])
        assert result.exit_code == 1, result.output
        assert "Can't predict: " in words(result.output)
        assert reason in words(result.output)


@pytest.mark.parametrize(
    "args",
    [
        ["--season", "2024", "--round", "5", "--sims", "10"],
        ["--season", "2017", "--round", "5"],
        ["--season", "2024"],
        ["--season", "2024", "--round", "5", "--grid-override", "no-such-file.json"],
    ],
    ids=["too few sims", "season before the data", "no round", "missing override file"],
)
def test_predict_refuses_bad_options_as_usage_errors(args) -> None:
    """Typer checks the options before anything loads (exit code 2)."""
    assert runner.invoke(app, ["predict", *args]).exit_code == 2


# ---------------------------------------------------------------------------
# update
# ---------------------------------------------------------------------------


class FakeDownload:
    """Stands in for the two download steps: saves ``tables`` as the new 2024 data."""

    def __init__(self, tables: RaceTables) -> None:
        self.tables = tables
        self.calls = 0

    def jolpica(self, first: int, last: int, **_: object) -> list[SeasonSummary]:
        """Save the prepared tables for the season and report it like a real fetch."""
        self.calls += 1
        save_tables(self.tables, range(first, last + 1))
        results = int((self.tables.results["season"] == last).sum())
        return [SeasonSummary(last, 5, results, 0, 0, 1, 0, 4)]

    def practice(self, first: int, last: int, **_: object) -> list[PracticeSummary]:
        """Practice laps were saved with the rest; just report them."""
        return [PracticeSummary(last, 13, 1000, 6, 1, 0, 0, 13)]


@pytest.fixture
def season_2024(tmp_path, monkeypatch):
    """A data folder where 2023 is a past season and 2024 is the season `update` refreshes."""
    for name, folder in (
        ("INTERIM_DIR", "interim"),
        ("PROCESSED_DIR", "processed"),
        ("MODELS_DIR", "models"),
        ("LOG_DIR", "logs"),
    ):
        monkeypatch.setattr(config, name, tmp_path / folder)
    monkeypatch.setattr(config, "FIRST_SEASON", 2023)
    monkeypatch.setattr(config, "LAST_SEASON", 2024)
    return tmp_path


def use_fake_download(monkeypatch, tables: RaceTables) -> FakeDownload:
    """Swap both download steps for a fake that saves ``tables``."""
    fake = FakeDownload(tables)
    monkeypatch.setattr(fetch_module, "fetch_jolpica", fake.jolpica)
    monkeypatch.setattr(fetch_module, "fetch_practice", fake.practice)
    return fake


def test_update_follows_a_race_weekend(season_2024, monkeypatch) -> None:
    """Nothing new, then qualifying, then the result, then a late penalty, then nothing."""
    full = make_tables()
    before = without_race(full, 2024, 5, keep_qualifying=False)
    save_tables(before, (2023, 2024))
    save_features(build_features(before))
    fake = use_fake_download(monkeypatch, before)

    quiet = runner.invoke(app, ["update", "--no-practice"])
    assert quiet.exit_code == 0, quiet.output
    text = words(quiet.output)
    assert "No new races or changes since the last update." in text
    assert "Next: 2024 round 5, Race 5. Qualifying starts Thu 09 May 2024, 15:00 UTC" in text
    assert "OpenF1 practice download" not in text
    assert "Trained ranker" in text
    assert "What the trees lean on" not in text

    fake.tables = without_race(full, 2024, 5)
    qualified = runner.invoke(app, ["update"])
    assert qualified.exit_code == 0, qualified.output
    text = words(qualified.output)
    assert "OpenF1 practice download" in text
    assert "qualifying: ready to predict" in text
    assert "Ready to predict: python -m f1predict predict --season 2024 --round 5" in text
    assert runner.invoke(app, ["predict", "--season", "2024", "--round", "5"]).exit_code == 0

    fake.tables = full
    raced = runner.invoke(app, ["update"])
    assert raced.exit_code == 0, raced.output
    text = words(raced.output)
    assert "2024 round 5" in text
    assert "result" in text
    assert "No races left on the 2024 calendar." in text
    _, description = load_ranker("ranker_full", season_2024 / "models")
    assert description["trained_on"]["last_race"] == "2024 round 5"

    results = full.results.copy()
    swapped = results.index[(results["season"] == 2024) & (results["round"] == 3)][:2]
    results.loc[swapped, "position"] = results.loc[swapped[::-1], "position"].to_numpy()
    fake.tables = replace(full, results=results)
    penalised = runner.invoke(app, ["update"])
    assert penalised.exit_code == 0, penalised.output
    assert "result changed (e.g. a late penalty)" in words(penalised.output)

    again = runner.invoke(app, ["update"])
    assert "No new races or changes since the last update." in words(again.output)
    assert fake.calls == 5


def test_update_needs_every_earlier_season_first(season_2024, monkeypatch) -> None:
    """Only 2024 on disk: `update` refuses before downloading and points to `fetch`."""
    save_tables(make_tables(), (2024,))
    fake = use_fake_download(monkeypatch, make_tables())
    result = runner.invoke(app, ["update"])
    assert result.exit_code == 1
    text = words(result.output)
    assert "Earlier seasons missing: 2023." in text
    assert "python -m f1predict fetch" in text
    assert fake.calls == 0


def test_update_explains_a_failed_download(season_2024, monkeypatch) -> None:
    """A download error stops the update with the reason and how to retry."""
    save_tables(make_tables(), (2023, 2024))

    def broken(*_: object, **__: object) -> None:
        """A download that fails."""
        raise ApiError("Jolpica is down")

    monkeypatch.setattr(fetch_module, "fetch_jolpica", broken)
    result = runner.invoke(app, ["update"])
    assert result.exit_code == 1
    text = words(result.output)
    assert "Update failed: Jolpica is down" in text
    assert "update --no-practice" in text
