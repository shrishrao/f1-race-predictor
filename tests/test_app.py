"""Tests for the Streamlit app: its helpers, its charts, and every page run headless.

The app only reads ``data/published``, so it's tested against the files ``publish`` wrote
for the synthetic workspace (``conftest.py``): four completed races and an upcoming one.
Streamlit's ``AppTest`` runs the real script without a browser and records every element
it draws, so a page that raises, or loses a table, fails here rather than in front of a
visitor.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import charts
import published

APP = Path(__file__).resolve().parent.parent / "app" / "streamlit_app.py"
TIMEOUT = 60

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def data_dir(published_workspace, monkeypatch) -> Path:
    """The app pointed at the workspace (it reads ``$F1PREDICT_DATA_DIR/published``)."""
    monkeypatch.setenv("F1PREDICT_DATA_DIR", str(published_workspace))
    return published_workspace


def load(root: Path, name: str) -> dict:
    """One published JSON file."""
    return json.loads((root / "published" / name).read_text(encoding="utf-8"))


def race(root: Path, number: int) -> dict:
    """A race's entry in the race list."""
    return next(entry for entry in load(root, "races.json")["races"] if entry["round"] == number)


def run_app() -> AppTest:
    """The app's first page, run once."""
    return AppTest.from_file(str(APP), default_timeout=TIMEOUT).run()


# ---------------------------------------------------------------------------
# Reading published files
# ---------------------------------------------------------------------------


def test_the_published_folder_follows_the_data_dir(monkeypatch, tmp_path) -> None:
    """``F1PREDICT_DATA_DIR`` moves it, as it moves the package's data folder."""
    monkeypatch.setenv("F1PREDICT_DATA_DIR", str(tmp_path))
    assert published.published_dir() == tmp_path.resolve() / "published"
    monkeypatch.delenv("F1PREDICT_DATA_DIR")
    assert published.published_dir() == published.REPO_ROOT / "data" / "published"


def test_reading_outside_the_published_folder_is_refused(data_dir) -> None:
    """A race list pointing at ``../`` can't make the app read other files."""
    with pytest.raises(ValueError, match="outside the published folder"):
        published.read("../processed/features.parquet")
    assert published.read("races.json")["races"]


# ---------------------------------------------------------------------------
# Words, numbers and tables
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("chance", "text"),
    [(0.0, "0.0%"), (0.0004, "<0.1%"), (0.4646, "46.5%"), (0.9996, ">99.9%"), (1.0, "100.0%")],
)
def test_percent_marks_the_extremes(chance, text) -> None:
    """Tiny and near-certain chances don't read as 0% or 100%."""
    assert published.percent(chance) == text


def test_ordinals_and_signs() -> None:
    """11th-13th are "th"; minus signs are true minus signs; zero has no sign."""
    assert [published.ordinal(n) for n in (1, 2, 3, 4, 11, 12, 13, 21, 22)] == [
        "1st", "2nd", "3rd", "4th", "11th", "12th", "13th", "21st", "22nd",
    ]  # fmt: skip
    assert published.number("{:.3f}", -0.0861, signed=True) == "\N{MINUS SIGN}0.086"
    assert published.number("{:.3f}", 0.0001, signed=True) == "0.000"
    assert published.number("{:.0%}", None) == "\N{EN DASH}"
    assert published.session_time("2026-10-24T21:00:00Z") == "Sat 24 Oct 2026, 21:00 UTC"


def test_the_app_opens_on_the_upcoming_race_or_the_latest_one_run() -> None:
    """The next race once it's qualified; otherwise the last race run."""
    races = [
        {"round": 1, "status": "completed"},
        {"round": 2, "status": "completed"},
        {"round": 3, "status": "upcoming"},
        {"round": 4, "status": "scheduled"},
    ]
    assert published.default_race(races)["round"] == 3
    assert published.default_race(races[:2] + races[3:])["round"] == 2
    assert published.default_race(races[3:])["round"] == 4


def test_the_driver_table_marks_grid_changes_and_results(data_dir) -> None:
    """Chances in percent; ``PL`` and ``*`` for an override; a result column once run."""
    upcoming = load(data_dir, "predictions/2024_r05.json")
    upcoming["grid_override"] = {"grid": {}, "pitlane": ["zhou"]}
    for driver in upcoming["drivers"]:
        if driver["driver_id"] == "zhou":
            driver.update(grid=None, pitlane_start=True)
    table = published.driver_table(upcoming)
    assert list(table.columns) == ["#", "Driver", "Team", "Grid", "Win", "Podium", "Points", "DNF"]
    assert "PL*" in set(table["Grid"])
    assert table["Win"].sum() == pytest.approx(100, abs=0.1)
    completed = published.driver_table(load(data_dir, "predictions/2024_r04.json"))
    assert "Result" in completed.columns


def test_tiles_and_headline_for_both_kinds_of_race(data_dir) -> None:
    """Upcoming: favourites and the DNF risk. Completed: the winner and how the call went."""
    upcoming = load(data_dir, "predictions/2024_r05.json")
    labels = [tile[0] for tile in published.race_tiles(upcoming, race(data_dir, 5))]
    assert labels == ["Favourite to win", "Second favourite", "Most likely to retire"]
    assert published.race_headline(upcoming, race(data_dir, 5)).startswith("The favourite is **")
    completed = load(data_dir, "predictions/2024_r04.json")
    tiles = published.race_tiles(completed, race(data_dir, 4))
    assert [tile[0] for tile in tiles] == [
        "Winner's chance",
        "Podium places called",
        "Order match (Spearman)",
        "Places off on average",
    ]
    assert tiles[1][1].endswith("of 3")
    assert " won " in published.race_headline(completed, race(data_dir, 4))


def test_the_backtest_tables_say_which_way_each_gap_goes(data_dir) -> None:
    """Every metric with its interval and a verdict in words, not colour alone."""
    metrics = load(data_dir, "metrics.json")
    table = published.comparison_table(metrics, 2024)
    assert len(table) == 10
    assert set(table["Verdict"]) <= {"model better", "too close to call", "model worse"}
    assert table["Model"].str.contains(r"\[").all()
    assert list(published.ablation_table(metrics, 2024)["Model"]) == ["Full model", "Grid baseline"]
    assert len(published.dnf_table(metrics, 2024)) == 3
    assert [tile[0] for tile in published.season_tiles(metrics, 2024)] == [
        "Spearman",
        "Winner accuracy",
        "Podium hit rate",
    ]


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------


def test_prediction_vs_result_labels_only_the_winner_and_the_biggest_misses(data_dir) -> None:
    """One series of finishers, a diagonal, and at most four labels."""
    figure = charts.prediction_vs_result(load(data_dir, "predictions/2024_r04.json"))
    assert len(figure.data) == 1
    labels = [text for text in figure.data[0].text if text]
    assert 1 <= len(labels) <= 4
    assert len(figure.layout.shapes) == 1


def test_calibration_keeps_the_model_first_so_colours_never_swap(data_dir) -> None:
    """Model then baseline in every panel (theme colour 1 and 2); no baseline for DNFs."""
    metrics = load(data_dir, "metrics.json")
    for outcome in ("win", "podium", "points"):
        names = [trace.name for trace in charts.calibration(metrics, outcome).data]
        assert names == ["Full model", "Grid baseline"]
    assert [trace.name for trace in charts.calibration(metrics, "dnf").data] == ["Full model"]


def test_importance_folds_small_inputs_into_one_bar(data_dir) -> None:
    """Inputs under 1% of the gain share one bar; the DNF chart is None for a base rate."""
    metrics = load(data_dir, "metrics.json")
    bars = charts.ranker_importance(metrics).data[0]
    small = [row for row in metrics["importance"]["ranker"] if row["gain_share"] < 0.01]
    assert (f"{len(small)} other inputs" in bars.y) == bool(small)
    assert sum(bars.x) == pytest.approx(1, abs=1e-3)
    base_rate = {**metrics, "importance": {**metrics["importance"], "dnf": {"inputs": []}}}
    assert charts.dnf_importance(base_rate) is None


# ---------------------------------------------------------------------------
# The app, run headless
# AppTest runs Streamlit's asyncio event loop, whose wake-up pipe is a local socket pair
# (a loopback TCP pair on Windows). These tests allow creating sockets, and connecting
# only to 127.0.0.1; nothing in them reaches the network.
# ---------------------------------------------------------------------------


@pytest.mark.enable_socket
@pytest.mark.allow_hosts(["127.0.0.1"])
def test_the_race_page_opens_on_the_upcoming_race(data_dir) -> None:
    """No exception; the qualified race, its tiles and a row per driver."""
    app = run_app()
    assert not app.exception
    assert app.title[0].value == "Race 5"
    assert [box.label for box in app.sidebar.selectbox] == ["Season", "Race"]
    assert app.metric[0].label == "Favourite to win"
    assert len(app.dataframe[0].value) == 6


@pytest.mark.enable_socket
@pytest.mark.allow_hosts(["127.0.0.1"])
def test_a_completed_race_is_shown_against_its_result(data_dir) -> None:
    """Picking round 4 adds the prediction-vs-result section and its scores."""
    app = run_app()
    app.sidebar.selectbox[1].set_value(4).run()
    assert not app.exception
    assert "Prediction vs result" in [header.value for header in app.subheader]
    assert len(app.get("plotly_chart")) == 1
    assert list(app.dataframe[1].value[""]) == ["This model", "Grid order"]


@pytest.mark.enable_socket
@pytest.mark.allow_hosts(["127.0.0.1"])
def test_a_race_not_qualified_says_when_qualifying_is(data_dir, tmp_path, monkeypatch) -> None:
    """A scheduled race shows when its qualifying starts instead of a table."""
    shutil.copytree(data_dir / "published", tmp_path / "published")
    index = load(tmp_path, "races.json")
    for entry in index["races"]:
        if entry["round"] == 5:
            entry.update(status="scheduled", file=None)
    (tmp_path / "published" / "races.json").write_text(json.dumps(index), encoding="utf-8")
    monkeypatch.setenv("F1PREDICT_DATA_DIR", str(tmp_path))
    app = run_app()
    assert app.title[0].value == "Race 4"
    app.sidebar.selectbox[1].set_value(5).run()
    assert not app.exception
    assert "Not qualified yet. Qualifying starts Thu 09 May 2024, 15:00 UTC" in app.info[0].value
    assert not app.dataframe


@pytest.mark.enable_socket
@pytest.mark.allow_hosts(["127.0.0.1"])
def test_the_other_pages_render(data_dir) -> None:
    """The backtest page (tables, four calibration panels, importances) and About.

    The DNF chart only appears when the DNF model in use has inputs (not the base rate).
    """
    app = run_app()
    app.switch_page("views/performance.py").run()
    assert not app.exception
    assert [header.value for header in app.subheader][:2] == [
        "The full model against the grid baseline",
        "Calibration",
    ]
    dnf_chart = bool(load(data_dir, "metrics.json")["importance"]["dnf"]["inputs"])
    assert len(app.get("plotly_chart")) == 5 + dnf_chart
    app.switch_page("views/about.py").run()
    assert not app.exception
    assert app.title[0].value == "About f1predict"


@pytest.mark.enable_socket
@pytest.mark.allow_hosts(["127.0.0.1"])
def test_without_published_files_the_app_says_what_to_run(tmp_path, monkeypatch) -> None:
    """A fresh clone that hasn't published yet gets an explanation, not a traceback."""
    monkeypatch.setenv("F1PREDICT_DATA_DIR", str(tmp_path))
    app = run_app()
    assert not app.exception
    assert "python -m f1predict publish" in app.error[0].value
