"""Tests for ``publish``: the files the Streamlit app reads, and the command that writes them.

The workspace (built once, in ``conftest.py``) is the synthetic championship with 2024
round 5 qualified but not run, models from ``train``, and the 2024 backtests. ``publish``
must write a prediction per race that matches the API, scores that match the backtest, the
backtest's season numbers, and copies of the models, and publishing again must change
nothing at all.
"""

from __future__ import annotations

import json
from dataclasses import replace

import pandas as pd
import pytest
from synthetic import make_tables
from typer.testing import CliRunner

from f1predict import config, predict
from f1predict import publish as publish_module
from f1predict.__main__ import app
from f1predict.api import load_race_data
from f1predict.evaluate.report import MODEL_METRICS
from f1predict.features.build import build_features
from f1predict.override import GridOverride, GridOverrideError
from f1predict.publish import publish

runner = CliRunner()
SEASONS = (2024,)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def published(root, name: str):
    """One published JSON file, parsed."""
    return json.loads((root / "published" / name).read_text(encoding="utf-8"))


def entries(root) -> dict[int, dict]:
    """The race list's entries by round."""
    return {race["round"]: race for race in published(root, "races.json")["races"]}


# ---------------------------------------------------------------------------
# What's written
# ---------------------------------------------------------------------------


def test_every_file_the_app_reads_is_written(published_workspace) -> None:
    """Five predictions, the race list, the backtest results and the three model files."""
    folder = published_workspace / "published"
    predictions = sorted(path.name for path in (folder / "predictions").iterdir())
    assert predictions == [f"2024_r0{number}.json" for number in range(1, 6)]
    assert (folder / "races.json").exists()
    assert (folder / "metrics.json").exists()
    models = sorted(path.name for path in (folder / "models").iterdir())
    assert models == ["dnf_full.json", "ranker_full.json", "ranker_full.txt"]
    for path in folder.rglob("*.json"):
        assert b"\r\n" not in path.read_bytes()


def test_the_race_list_says_what_happened_and_what_was_predicted(published_workspace) -> None:
    """Rounds 1-4 completed (with winner and scores), round 5 upcoming (favourite only)."""
    races = entries(published_workspace)
    assert [races[number]["status"] for number in range(1, 6)] == ["completed"] * 4 + ["upcoming"]
    for number in range(1, 5):
        assert races[number]["file"] == f"predictions/2024_r0{number}.json"
        assert {"winner", "scores", "favourite"} <= set(races[number])
        assert set(races[number]["scores"]) == {"model", "baseline"}
    assert "winner" not in races[5]
    assert "scores" not in races[5]
    assert races[5]["favourite"]["p_win"] > 0
    assert published(published_workspace, "races.json")["data_through"] == "2024 round 4"


def test_published_trees_score_like_the_saved_model(in_workspace) -> None:
    """The LF-normalized copy of the LightGBM file loads and gives identical scores."""
    import lightgbm as lgb

    from f1predict.features.build import load_features

    rows = load_features().head(30)
    saved = lgb.Booster(model_file=str(in_workspace / "models" / "ranker_full.txt"))
    copy = lgb.Booster(model_file=str(in_workspace / "published" / "models" / "ranker_full.txt"))
    features = saved.feature_name()
    assert (saved.predict(rows[features]) == copy.predict(rows[features])).all()


def test_each_prediction_is_what_the_api_returns(in_workspace) -> None:
    """The published files hold exactly the dicts ``predict`` gives."""
    for number in range(1, 6):
        assert published(in_workspace, f"predictions/2024_r0{number}.json") == predict(2024, number)


def test_publishing_again_changes_nothing(in_workspace) -> None:
    """Same data, same models: every file is left exactly as it was."""
    summary = publish(seasons=SEASONS)
    assert summary.changed == []
    assert summary.removed == []
    assert summary.unchanged == 10
    assert (summary.completed, summary.upcoming, summary.scheduled) == (4, 1, 0)


# ---------------------------------------------------------------------------
# Scores and backtest numbers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("source", "backtest"), [("model", "ranker_full"), ("baseline", "baseline")]
)
def test_race_scores_match_the_backtest(published_workspace, source, backtest) -> None:
    """Each completed race's winner, podium, MAE and Spearman equal the backtest's."""
    saved = pd.read_csv(published_workspace / "processed" / "backtest" / backtest / "races.csv")
    saved = saved.set_index("round")
    for number, race in entries(published_workspace).items():
        if race["status"] != "completed":
            continue
        for metric, value in race["scores"][source].items():
            assert value == pytest.approx(saved.loc[number, metric], abs=1e-4)


def test_metrics_hold_the_backtest_results(published_workspace) -> None:
    """Every comparison row, the DNF models, calibration from both models, importances."""
    metrics = published(published_workspace, "metrics.json")
    assert metrics["test_seasons"] == [2024]
    assert metrics["races"] == {"2024": 4}
    assert [row["metric"] for row in metrics["vs_baseline"]] == MODEL_METRICS
    summary = json.loads(
        (
            published_workspace / "processed" / "backtest" / "ranker_full" / "summary.json"
        ).read_text()
    )
    spearman = summary["seasons"]["2024"]["metrics"]["spearman"]["value"]
    row = next(row for row in metrics["vs_baseline"] if row["metric"] == "spearman")
    assert row["model"]["value"] == pytest.approx(spearman, abs=1e-4)
    assert row["difference"]["verdict"] in (-1, 0, 1)
    assert {row["variant"] for row in metrics["ablations"]} == {"ranker_full", "baseline"}
    assert len(metrics["dnf_models"]["rows"]) == 9
    assert {row["model"] for row in metrics["calibration"]} == {"model", "baseline"}
    assert not [
        row
        for row in metrics["calibration"]
        if row["model"] == "baseline" and row["outcome"] == "dnf"
    ]
    shares = [row["gain_share"] for row in metrics["importance"]["ranker"]]
    assert sum(shares) == pytest.approx(1, abs=1e-3)


# ---------------------------------------------------------------------------
# Overrides, leftovers and warnings
# ---------------------------------------------------------------------------


def test_a_grid_override_goes_to_the_upcoming_race(in_workspace, tmp_path) -> None:
    """The upcoming race is predicted with the override; completed races never are."""
    folder = tmp_path / "published"
    publish(seasons=SEASONS, grid_override=GridOverride(pitlane=["hamilton"]), folder=folder)
    upcoming = json.loads((folder / "predictions" / "2024_r05.json").read_text(encoding="utf-8"))
    assert upcoming["grid_override"] == {"grid": {}, "pitlane": ["hamilton"]}
    races = {
        race["round"]: race for race in json.loads((folder / "races.json").read_text())["races"]
    }
    assert races[5]["grid_override"] is True
    assert races[4]["grid_override"] is False


def test_leftover_files_are_removed(in_workspace, tmp_path) -> None:
    """A prediction for a race no longer listed, or an old model file, doesn't linger."""
    folder = tmp_path / "published"
    for leftover in (folder / "predictions" / "2024_r09.json", folder / "models" / "dnf_full.txt"):
        leftover.parent.mkdir(parents=True, exist_ok=True)
        leftover.write_text("{}")
    summary = publish(seasons=SEASONS, folder=folder)
    assert sorted(path.name for path in summary.removed) == ["2024_r09.json", "dnf_full.txt"]
    assert not (folder / "predictions" / "2024_r09.json").exists()


def test_a_race_not_qualified_is_listed_without_a_prediction(in_workspace, tmp_path, monkeypatch):
    """No qualifying yet: listed as scheduled, no file, and an override has nowhere to go."""
    data = load_race_data()
    later = (data.features["season"] == 2024) & (data.features["round"] == 5)
    monkeypatch.setattr(
        publish_module, "load_race_data", lambda: replace(data, features=data.features[~later])
    )
    folder = tmp_path / "published"
    summary = publish(seasons=SEASONS, folder=folder)
    assert summary.scheduled == 1
    races = {
        race["round"]: race for race in json.loads((folder / "races.json").read_text())["races"]
    }
    assert races[5]["status"] == "scheduled"
    assert races[5]["file"] is None
    assert races[5]["qualifying_start"] == "2024-05-09T15:00:00Z"
    with pytest.raises(GridOverrideError, match="no upcoming qualified race"):
        publish(seasons=SEASONS, folder=folder, grid_override=GridOverride(pitlane=["hamilton"]))


def test_backtests_behind_the_data_are_flagged(in_workspace, tmp_path, monkeypatch) -> None:
    """Round 5 has a result but the backtests stop at round 4: publish says to rerun them."""
    tables = make_tables()
    data = load_race_data()
    full = replace(data, features=build_features(tables), results=tables.results)
    monkeypatch.setattr(publish_module, "load_race_data", lambda: full)
    summary = publish(seasons=SEASONS, folder=tmp_path / "published")
    assert summary.completed == 5
    assert len(summary.warnings) == 1
    assert "scored 4 of the 5 races run in 2024" in summary.warnings[0]


# ---------------------------------------------------------------------------
# The command
# ---------------------------------------------------------------------------


def words(output: str) -> str:
    """Command output with line breaks and runs of spaces collapsed."""
    return " ".join(output.split())


def test_publish_command_reports_that_nothing_changed(in_workspace, monkeypatch) -> None:
    """Run on an up-to-date folder: the table, and nothing to commit."""
    monkeypatch.setattr(config, "TEST_SEASONS", SEASONS)
    result = runner.invoke(app, ["publish"])
    assert result.exit_code == 0, result.output
    assert "Nothing changed, so there's nothing to commit." in words(result.output)


def test_publish_command_with_a_grid_override(in_workspace, monkeypatch, tmp_path) -> None:
    """``--grid-override`` changes the upcoming race's file, and the command lists it."""
    monkeypatch.setattr(config, "TEST_SEASONS", SEASONS)
    monkeypatch.setattr(config, "PUBLISHED_DIR", tmp_path / "published")
    override = tmp_path / "penalties.json"
    override.write_text(json.dumps({"season": 2024, "round": 5, "pitlane": ["zhou"]}))
    result = runner.invoke(app, ["publish", "--grid-override", str(override)])
    assert result.exit_code == 0, result.output
    assert "2024_r05.json" in result.output
    assert "to update the app" in words(result.output)


def test_publish_command_explains_missing_backtests(in_workspace, monkeypatch, tmp_path) -> None:
    """No backtests saved: exit 1 and the command that makes them."""
    monkeypatch.setattr(config, "TEST_SEASONS", SEASONS)
    monkeypatch.setattr(config, "PROCESSED_DIR", tmp_path / "processed")
    result = runner.invoke(app, ["publish"])
    assert result.exit_code == 1
    assert "Publish failed" in result.output
    assert "python -m f1predict backtest" in words(result.output)
