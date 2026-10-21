"""Read and shape the files ``python -m f1predict publish`` writes, for the app's pages.

The app reads only ``data/published``: ``races.json`` (the calendar, with what was predicted
and what happened), one ``predictions/<season>_r<NN>.json`` per race, and ``metrics.json``
(the backtest results). Nothing here imports ``f1predict``, so the deployed app needs neither
LightGBM nor the raw data.

Everything except ``read`` is plain Python that turns those files into tables and text, so
the tests can check it without a browser.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parent.parent
MINUS = "\N{MINUS SIGN}"
DASH = "\N{EN DASH}"
CHANCES = {"p_win": "Win", "p_podium": "Podium", "p_points": "Points", "p_dnf": "DNF"}
VERDICTS = {1: "model better", 0: "too close to call", -1: "model worse"}
MARKS = {1: " (better)", 0: "", -1: " (worse)"}
TILE_METRICS = ["spearman", "winner_correct", "podium_hit_rate"]
ABLATION_METRICS = ["spearman", "winner_correct"]
ENDINGS = {"dnf": "retired", "dns": "didn't start", "dsq": "was disqualified"}

# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------


def published_dir() -> Path:
    """``data/published`` in the repo, or under ``F1PREDICT_DATA_DIR`` when that's set."""
    root = os.environ.get("F1PREDICT_DATA_DIR", "").strip()
    data = Path(root).expanduser().resolve() if root else REPO_ROOT / "data"
    return data / "published"


@st.cache_data(show_spinner=False)
def _parse(path: str, modified: float) -> Any:
    """One JSON file's contents; ``modified`` makes a re-published file a new cache entry."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read(name: str) -> Any:
    """A published JSON file by its path inside the folder, e.g. ``races.json``.

    A path leading outside the folder is refused, so the race list can only point the app
    at published files.
    """
    folder = published_dir().resolve()
    path = (folder / name).resolve()
    if folder not in path.parents:
        raise ValueError(f"{name} is outside the published folder")
    return _parse(str(path), path.stat().st_mtime)


# ---------------------------------------------------------------------------
# Words and numbers
# ---------------------------------------------------------------------------


def session_time(iso: str | None) -> str:
    """``2026-10-24T21:00:00Z`` as ``Sat 24 Oct 2026, 21:00 UTC``."""
    if not iso:
        return "at a time not announced yet"
    moment = datetime.fromisoformat(iso.replace("Z", "+00:00"))
    return moment.strftime("%a %d %b %Y, %H:%M UTC")


def percent(chance: float) -> str:
    """A chance as a percentage, showing the extremes as "<0.1%" and ">99.9%"."""
    if 0 < chance < 0.001:
        return "<0.1%"
    if 0.999 < chance < 1:
        return ">99.9%"
    return f"{chance:.1%}"


def ordinal(number: int) -> str:
    """1st, 2nd, 3rd, 4th, ..., 11th, 12th, 13th, ..., 21st, 22nd."""
    teens = 10 <= number % 100 <= 20
    suffix = "th" if teens else {1: "st", 2: "nd", 3: "rd"}.get(number % 10, "th")
    return f"{number}{suffix}"


def number(fmt: str, value: float | None, signed: bool = False) -> str:
    """A value in a metric's format with a true minus sign (and a + when ``signed``).

    A value that rounds to zero gets no sign, rather than "-0.00" or "+0%".
    """
    if value is None:
        return DASH
    text = (fmt.replace("{:", "{:+") if signed else fmt).format(value)
    if not any(digit in text for digit in "123456789"):
        text = text.lstrip("+-")
    return text.replace("-", MINUS)


def with_interval(fmt: str, cell: dict[str, Any], signed: bool = False) -> str:
    """``0.77 [0.71, 0.83]``: a value and its 95% confidence interval."""
    low, high = number(fmt, cell["low"], signed), number(fmt, cell["high"], signed)
    return f"{number(fmt, cell['value'], signed)} [{low}, {high}]"


def gap_format(fmt: str) -> str:
    """Differences of two-decimal metrics get a third decimal, so small gaps don't read 0.00."""
    return fmt.replace(".2f", ".3f")


def driver_name(driver: dict[str, Any]) -> str:
    """A driver's name, falling back to the code or id when the data has no name."""
    return driver["name"] or driver["code"] or driver["driver_id"]


def finish_text(actual: dict[str, Any] | None) -> str:
    """How a driver's race ended, in words: won, finished 4th, retired, ..."""
    if not actual:
        return "not run yet"
    if actual["status_class"] == "finished":
        return "won" if actual["position"] == 1 else f"finished {ordinal(actual['position'])}"
    return ENDINGS.get(actual["status_class"], str(actual["status_class"]))


def result_cell(actual: dict[str, Any] | None) -> str:
    """The table's result column: a finishing position, or DNF, DNS or DSQ."""
    if not actual:
        return DASH
    if actual["status_class"] == "finished":
        return actual["position_text"] or str(actual["position"])
    return str(actual["status_class"]).upper()


# ---------------------------------------------------------------------------
# Races
# ---------------------------------------------------------------------------


def default_race(races: list[dict[str, Any]]) -> dict[str, Any]:
    """The race to open on: the next race once it's qualified, else the latest one run."""
    upcoming = [race for race in races if race["status"] == "upcoming"]
    completed = [race for race in races if race["status"] == "completed"]
    if upcoming:
        return upcoming[0]
    return completed[-1] if completed else races[0]


def race_option(race: dict[str, Any]) -> str:
    """How a race is listed in the picker, e.g. ``R18 · United States Grand Prix``."""
    note = {"upcoming": " (qualified)", "scheduled": " (not qualified yet)"}
    return f"R{race['round']} · {race['name']}{note.get(race['status'], '')}"


def race_caption(race: dict[str, Any]) -> str:
    """The line under a race's title: season, round, circuit and start time."""
    where = ", ".join(part for part in (race["circuit"], race["country"]) if part)
    start = session_time(race["race_start"])
    return f"{race['season']} · Round {race['round']} · {where} · Race start {start}"


def not_qualified_text(race: dict[str, Any]) -> str:
    """What the race page says for a race that hasn't been qualified yet."""
    return (
        f"Not qualified yet. Qualifying starts {session_time(race['qualifying_start'])}; "
        "the prediction appears here once qualifying is over and it has been published."
    )


def how_it_was_made(prediction: dict[str, Any]) -> str:
    """One or two sentences on which models made the prediction and what they learned from."""
    trained = prediction["model"]["trained_on"]
    span = f"{trained['races']} races, {trained['first_race']} to {trained['last_race']}"
    if prediction["race"]["status"] == "completed":
        text = (
            f"Predicted the way the backtest does it: by models trained only on the races "
            f"before this one ({span}), then shown next to the result."
        )
    else:
        text = f"Predicted after qualifying by the saved models ({span})."
    if prediction["grid_override"]:
        text += " Grid slots marked * were set after qualifying, for penalties or pit-lane starts."
    return text


def model_card(prediction: dict[str, Any]) -> str:
    """The models and settings behind a prediction, as a Markdown list."""
    model, trained = prediction["model"], prediction["model"]["trained_on"]
    source = "retrained on the races before this one" if model["source"] == "retrained" else "saved"
    settings = model["ranker_settings"]
    window = model["dnf_window"]
    look_back = "all earlier races" if window is None else f"the last {window} races"
    lines = [
        f"- **Models**: {source}, from {trained['races']} races "
        f"({trained['first_race']} to {trained['last_race']}).",
        f"- **Ranker**: LightGBM lambdarank, {settings['n_estimators']} trees of "
        f"{settings['num_leaves']} leaves.",
        f"- **Simulation**: {model['sims']:,} races, temperature {model['temperature']:.3f}, "
        f"seed {model['seed']}.",
        f"- **Retirements**: {model['dnf_model']} DNF model, field-wide rate over {look_back}.",
        f"- **Settings chosen on**: {model['settings_chosen_on']}, the validation season.",
    ]
    return "\n".join(lines)


def driver_table(prediction: dict[str, Any]) -> pd.DataFrame:
    """One row per driver in predicted order, with chances in percent (0-100)."""
    override = prediction["grid_override"] or {"grid": {}, "pitlane": []}
    moved = {*override["grid"], *override["pitlane"]}
    completed = prediction["race"]["status"] == "completed"
    rows = []
    for driver in prediction["drivers"]:
        if driver["pitlane_start"]:
            grid = "PL"
        else:
            grid = DASH if driver["grid"] is None else str(driver["grid"])
        row = {
            "#": driver["predicted_rank"],
            "Driver": f"{driver['code'] or ''} {driver_name(driver)}".strip(),
            "Team": driver["team"] or driver["constructor_id"] or "",
            "Grid": grid + ("*" if driver["driver_id"] in moved else ""),
            **{label: 100 * driver[key] for key, label in CHANCES.items()},
        }
        if completed:
            row["Result"] = result_cell(driver["actual"])
        rows.append(row)
    return pd.DataFrame(rows)


def _winner(drivers: list[dict[str, Any]]) -> dict[str, Any] | None:
    """The driver classified first, or None for a race not run yet."""
    return next((d for d in drivers if d["actual"] and d["actual"]["position_text"] == "1"), None)


def race_headline(prediction: dict[str, Any], race: dict[str, Any]) -> str:
    """One sentence on the race: who's favourite, or who won and how they were rated."""
    drivers = prediction["drivers"]
    by_win = sorted(drivers, key=lambda driver: -driver["p_win"])
    favourite = by_win[0]
    odds = f"{driver_name(favourite)} at {percent(favourite['p_win'])}"
    if race["status"] != "completed":
        podium = sorted(drivers, key=lambda driver: -driver["p_podium"])[:3]
        names = ", ".join(driver_name(driver) for driver in podium)
        return (
            f"The favourite is **{driver_name(favourite)}**, at {percent(favourite['p_win'])} "
            f"to win. Most likely podium: {names}."
        )
    winner = _winner(drivers)
    if winner is None:
        return f"The favourite was {odds}, who {finish_text(favourite['actual'])}."
    place = by_win.index(winner) + 1
    if place == 1:
        return f"**{driver_name(winner)}** won as the favourite, at {percent(winner['p_win'])}."
    return (
        f"**{driver_name(winner)}** won with a {percent(winner['p_win'])} chance, the "
        f"{ordinal(place)} favourite. The favourite, {odds}, {finish_text(favourite['actual'])}."
    )


def race_tiles(prediction: dict[str, Any], race: dict[str, Any]) -> list[tuple[str, str, str]]:
    """The stat tiles under the headline, each ``(label, value, detail)``; values are numbers."""
    drivers = prediction["drivers"]
    by_win = sorted(drivers, key=lambda driver: -driver["p_win"])
    if race["status"] != "completed":
        risk = max(drivers, key=lambda driver: driver["p_dnf"])
        return [
            ("Favourite to win", percent(by_win[0]["p_win"]), driver_name(by_win[0])),
            ("Second favourite", percent(by_win[1]["p_win"]), driver_name(by_win[1])),
            ("Most likely to retire", percent(risk["p_dnf"]), driver_name(risk)),
        ]
    tiles = []
    winner = _winner(drivers)
    if winner is not None:
        tiles.append(("Winner's chance", percent(winner["p_win"]), driver_name(winner)))
    model, grid = race["scores"]["model"], race["scores"]["baseline"]
    tiles += [
        (
            "Podium places called",
            f"{round(3 * model['podium_hit_rate'])} of 3",
            f"grid order: {round(3 * grid['podium_hit_rate'])} of 3",
        ),
        (
            "Order match (Spearman)",
            number("{:.2f}", model["spearman"]),
            f"grid order: {number('{:.2f}', grid['spearman'])}",
        ),
        (
            "Places off on average",
            number("{:.2f}", model["mae"]),
            f"grid order: {number('{:.2f}', grid['mae'])}",
        ),
    ]
    return tiles


def scores_table(race: dict[str, Any]) -> pd.DataFrame:
    """The prediction and the grid order in one race, by the backtest's order metrics."""
    rows = []
    for key, label in (("model", "This model"), ("baseline", "Grid order")):
        score = race["scores"][key]
        rows.append(
            {
                "": label,
                "Winner": "right" if score["winner_correct"] else "wrong",
                "Podium": f"{round(3 * score['podium_hit_rate'])} of 3",
                "MAE": number("{:.2f}", score["mae"]),
                "Spearman": number("{:.2f}", score["spearman"]),
            }
        )
    return pd.DataFrame(rows)


def non_finishers_text(prediction: dict[str, Any]) -> str:
    """The cars that didn't finish, each with the DNF chance it was given."""
    drivers = prediction["drivers"]
    out = [d for d in drivers if d["actual"] and d["actual"]["status_class"] != "finished"]
    if not out:
        return "Every car finished."
    listed = ", ".join(
        f"{d['code'] or driver_name(d)} {finish_text(d['actual'])} (given {percent(d['p_dnf'])})"
        for d in out
    )
    average = sum(driver["p_dnf"] for driver in drivers) / len(drivers)
    return (
        f"Didn't finish, with the DNF chance each car was given: {listed}. "
        f"The field's average DNF chance was {percent(average)}."
    )


# ---------------------------------------------------------------------------
# Backtest results
# ---------------------------------------------------------------------------


def season_label(metrics: dict[str, Any], season: int) -> str:
    """A test season with its race count, e.g. ``2026 (17 races)``."""
    return f"{season} ({metrics['races'][str(season)]} races)"


def season_tiles(metrics: dict[str, Any], season: int) -> list[tuple[str, str, str, int, str]]:
    """Headline numbers for one season: ``(label, value, gap to grid, verdict, better)``.

    The gap keeps a plain hyphen for its minus sign: Streamlit reads a tile's direction
    from it.
    """
    rows = {row["metric"]: row for row in metrics["vs_baseline"] if row["season"] == season}
    tiles = []
    for metric in TILE_METRICS:
        row, spec = rows[metric], metrics["metrics"][metric]
        gap = number(gap_format(spec["format"]), row["difference"]["value"], signed=True)
        gap = gap.replace(MINUS, "-")
        tiles.append(
            (
                spec["label"],
                number(spec["format"], row["model"]["value"]),
                f"{gap} vs grid order",
                row["difference"]["verdict"],
                spec["better"],
            )
        )
    return tiles


def comparison_table(metrics: dict[str, Any], season: int) -> pd.DataFrame:
    """The full model against the grid baseline in one season, one row per metric."""
    rows = []
    for row in metrics["vs_baseline"]:
        if row["season"] != season:
            continue
        spec = metrics["metrics"][row["metric"]]
        rows.append(
            {
                "Metric": spec["label"],
                "Better": spec["better"],
                "Grid baseline": with_interval(spec["format"], row["baseline"]),
                "Model": with_interval(spec["format"], row["model"]),
                "Difference": with_interval(gap_format(spec["format"]), row["difference"], True),
                "Verdict": VERDICTS[row["difference"]["verdict"]],
            }
        )
    return pd.DataFrame(rows)


def ablation_table(metrics: dict[str, Any], season: int) -> pd.DataFrame:
    """Each feature set's Spearman and winner accuracy, with its gap to the full model."""
    rows: dict[str, dict[str, str]] = {}
    for row in metrics["ablations"]:
        if row["season"] != season or row["metric"] not in ABLATION_METRICS:
            continue
        spec = metrics["metrics"][row["metric"]]
        cells = rows.setdefault(row["variant"], {"Model": row["label"]})
        cells[spec["label"]] = number(spec["format"], row["value"])
        gap = row.get("difference")
        cells[f"{spec['label']} vs full model"] = (
            DASH
            if gap is None
            else with_interval(gap_format(spec["format"]), gap, True) + MARKS[gap["verdict"]]
        )
    return pd.DataFrame(list(rows.values()))


def dnf_table(metrics: dict[str, Any], season: int) -> pd.DataFrame:
    """The three DNF models in one season: Brier score, log loss and ROC AUC with their CIs."""
    chosen = metrics["dnf_models"]["chosen"]
    rows: dict[str, dict[str, str]] = {}
    for row in metrics["dnf_models"]["rows"]:
        if row["season"] != season:
            continue
        spec = metrics["metrics"][row["metric"]]
        label = row["label"] + (" (used)" if row["model"] == chosen else "")
        cells = rows.setdefault(row["model"], {"Model": label})
        cells[f"{spec['label']} ({spec['better']} is better)"] = with_interval(spec["format"], row)
    return pd.DataFrame(list(rows.values()))
