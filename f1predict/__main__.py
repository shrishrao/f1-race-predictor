"""Command-line interface: ``python -m f1predict <command>``.

Heavy libraries (pandas, requests, ...) are imported inside the commands that need them,
so ``--help`` and ``info`` start instantly.
"""

from __future__ import annotations

import platform
import time
from contextlib import contextmanager
from functools import partial
from typing import TYPE_CHECKING, Annotated

import typer
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from rich.text import Text

from f1predict import __version__, config
from f1predict.log import setup_logging
from f1predict.models import FeatureSetName, ModelName

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    import pandas as pd

    from f1predict.data.fetch import PracticeSummary, SeasonSummary
    from f1predict.models.ranker import RankerModel

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = typer.Typer(
    help="Predict F1 race outcomes after qualifying.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()


def _show_version(value: bool) -> None:
    """Print the package version and stop, when ``--version`` is passed."""
    if value:
        console.print(f"f1predict {__version__}")
        raise typer.Exit()


@app.callback()
def cli(
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            callback=_show_version,
            is_eager=True,
            help="Show the version and exit.",
        ),
    ] = False,
) -> None:
    """Predict F1 race outcomes after qualifying."""


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


@app.command()
def info() -> None:
    """Show the version, the Python in use, and where data and models are stored."""
    table = Table(title="f1predict", show_header=False)
    table.add_column("Setting", style="bold")
    table.add_column("Value")
    table.add_row("Version", __version__)
    table.add_row("Python", platform.python_version())
    table.add_row("Seasons", f"{config.FIRST_SEASON}-{config.LAST_SEASON}")
    table.add_row("Data directory", str(config.DATA_DIR))
    table.add_row("Models directory", str(config.MODELS_DIR))
    console.print(table)


@app.command()
def fetch(
    from_season: Annotated[
        int,
        typer.Option(
            "--from",
            min=config.FIRST_SEASON,
            max=config.LAST_SEASON,
            help="First season to download.",
        ),
    ] = config.FIRST_SEASON,
    to_season: Annotated[
        int,
        typer.Option(
            "--to",
            min=config.FIRST_SEASON,
            max=config.LAST_SEASON,
            help="Last season to download.",
        ),
    ] = config.LAST_SEASON,
    refresh: Annotated[
        bool,
        typer.Option("--refresh", help="Ignore cached responses and download everything again."),
    ] = False,
    practice: Annotated[
        bool,
        typer.Option(
            "--practice/--no-practice",
            help="Also download practice laps from OpenF1 (available from 2023).",
        ),
    ] = True,
) -> None:
    """Download race data from Jolpica and practice laps from OpenF1.

    Safe to stop and re-run at any time: everything already downloaded is cached.
    """
    if from_season > to_season:
        raise typer.BadParameter(
            f"{from_season} is later than --to {to_season}.", param_hint="'--from'"
        )

    from f1predict.data.fetch import fetch_jolpica, fetch_practice, practice_seasons
    from f1predict.data.http import ApiError
    from f1predict.data.mapping import DriverMappingError
    from f1predict.data.openf1 import PracticeAlignmentError
    from f1predict.data.tables import DataValidationError

    setup_logging(log_file=config.LOG_DIR / "fetch.log")
    try:
        _print_fetch_summary(fetch_jolpica(from_season, to_season, refresh=refresh))
        if practice and practice_seasons(from_season, to_season):
            _print_practice_summary(fetch_practice(from_season, to_season, refresh=refresh))
        elif practice:
            console.print(
                f"Practice laps come from OpenF1 and start in {config.FIRST_PRACTICE_SEASON}; "
                "none to download for these seasons."
            )
    except (ApiError, DataValidationError, PracticeAlignmentError, DriverMappingError) as exc:
        console.print(f"[bold red]Fetch failed:[/bold red] {exc}")
        console.print("Everything downloaded so far is cached. Run the same command to resume.")
        raise typer.Exit(code=1) from exc


@app.command("build-features")
def build_features_command() -> None:
    """Build the features table from downloaded data (one row per driver per race)."""
    from f1predict.data.lineage import UnknownConstructorError
    from f1predict.features.build import build_features, load_race_tables, save_features

    setup_logging(log_file=config.LOG_DIR / "build_features.log")
    try:
        features = build_features(load_race_tables())
    except FileNotFoundError as exc:
        console.print(f"[bold red]No data yet:[/bold red] {exc}")
        raise typer.Exit(code=1) from exc
    except UnknownConstructorError as exc:
        console.print(f"[bold red]Build failed:[/bold red] {exc}")
        raise typer.Exit(code=1) from exc
    path = save_features(features)

    table = Table(title="Features")
    for column in ("Season", "Races", "Rows", "Quali gap", "Practice", "Long run"):
        table.add_column(column, justify="right")
    for season, rows in features.groupby("season"):
        table.add_row(
            str(season),
            str(rows["round"].nunique()),
            str(len(rows)),
            f"{rows['quali_gap_pole_pct'].notna().mean():.0%}",
            f"{rows['fp_best_gap_pct'].notna().mean():.0%}",
            f"{rows['longrun_best_gap_pct'].notna().mean():.0%}",
        )
    console.print(table)
    console.print(f"Saved {len(features)} rows to {path}")


FeatureSetOption = Annotated[
    FeatureSetName,
    typer.Option("--features", help="The ranker's input columns (an ablation feature set)."),
]


@app.command("backtest")
def backtest_command(
    model: Annotated[ModelName, typer.Option(help="Model to evaluate.")] = "baseline",
    feature_set: FeatureSetOption = "full",
    seasons: Annotated[
        list[int] | None,
        typer.Option(
            "--season",
            min=config.FIRST_SEASON + 1,
            max=config.LAST_SEASON,
            help="Season to backtest; repeat for several. "
            "Default: 2024 (validation), 2025 and 2026 (test).",
        ),
    ] = None,
) -> None:
    """Predict every race of the chosen seasons using only earlier races, and score it.

    Prints season averages with 95% confidence intervals and saves the per-race results.
    """
    from dataclasses import replace

    from f1predict.evaluate.backtest import (
        PredictionError,
        completed_races,
        predict_races,
        run_backtest,
        save_backtest,
        score_predictions,
    )
    from f1predict.evaluate.metrics import compare
    from f1predict.models import make_model, run_name

    setup_logging(log_file=config.LOG_DIR / "backtest.log")
    chosen = sorted(set(seasons or [config.VALIDATION_SEASON, *config.TEST_SEASONS]))
    features = _load_features_or_exit()
    total = len(completed_races(features, chosen))
    if not total:
        console.print(f"[bold red]No completed races to backtest in {chosen}.[/bold red]")
        raise typer.Exit(code=1)

    name = run_name(model, feature_set)
    temperature = _validation_temperature(features, feature_set) if model == "ranker" else None
    try:
        with _race_progress(total, name) as progress:
            predictions = predict_races(
                partial(make_model, model, feature_set, temperature), features, chosen, progress
            )
        result = score_predictions(predictions)
        if model != "baseline":
            reference = run_backtest(partial(make_model, "baseline"), features, chosen)
            result = replace(result, comparison=compare(result.races, reference.races))
    except PredictionError as exc:
        console.print(f"[bold red]Backtest failed:[/bold red] {exc}")
        raise typer.Exit(code=1) from exc
    folder = save_backtest(result, name)

    _print_season_table(result.summary, f"Backtest: {name}")
    console.print(
        "Season averages over races, with 95% bootstrap confidence intervals below.\n"
        "MAE and Spearman compare the order of the cars that finished."
    )
    if result.comparison is not None:
        _print_season_table(
            result.comparison, f"{name} minus baseline", value="difference", signed=True
        )
        console.print(
            "Same races for both models, paired 95% confidence intervals below.\n"
            "[green]Green[/green]: clearly better than the baseline; "
            "[red]red[/red]: clearly worse."
        )
    missing = sorted(set(chosen) - set(result.summary["season"]))
    if missing:
        console.print(f"[yellow]Warning:[/yellow] no completed races for {missing}")
    console.print(f"Saved per-race results to {folder}")


@app.command()
def tune(feature_set: FeatureSetOption = "full") -> None:
    """Choose the ranker's settings by backtesting a small grid on the 2024 validation season.

    The 2025 and 2026 test seasons are never used. Saves the full table under data/processed.
    """
    from f1predict.evaluate.backtest import completed_races
    from f1predict.evaluate.tuning import choose_settings, grid_settings, tune_ranker
    from f1predict.models.ranker import RankerModel

    setup_logging(log_file=config.LOG_DIR / "tune.log")
    features = _load_features_or_exit()
    season = config.VALIDATION_SEASON
    combinations = len(grid_settings())
    races = len(completed_races(features, [season]))
    if not races:
        console.print(f"[bold red]No completed races in {season} to tune on.[/bold red]")
        raise typer.Exit(code=1)

    console.print(
        f"Backtesting {combinations} combinations of settings on the {races} races of "
        f"{season}: {combinations * races} models, each trained on the races before its race."
    )
    with _race_progress(combinations * races, "Tuning") as progress:
        table = tune_ranker(features, feature_set, season, progress=progress)
    path = config.PROCESSED_DIR / "tuning" / f"ranker_{feature_set}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False, float_format="%.6g")
    _print_tuning(table, choose_settings(table), RankerModel().settings(), season)
    console.print(f"Saved all {len(table)} combinations to {path}")


@app.command()
def train(feature_set: FeatureSetOption = "full") -> None:
    """Train the ranker on every completed race and save it, with its temperature, in models/."""
    from f1predict.models import run_name
    from f1predict.models.ranker import RankerModel, training_rows
    from f1predict.models.store import save_ranker

    setup_logging(log_file=config.LOG_DIR / "train.log")
    features = _load_features_or_exit()
    history = features[features["status_class"].notna()]
    rows = training_rows(history)
    if rows.empty:
        console.print("[bold red]No completed races to train on.[/bold red]")
        raise typer.Exit(code=1)

    temperature = _validation_temperature(features, feature_set)
    started = time.perf_counter()
    model = RankerModel(feature_set=feature_set).fit(history)
    seconds = time.perf_counter() - started
    first, last = rows.iloc[0], rows.iloc[-1]
    trained_on = {
        "races": int(rows["race_seq"].nunique()),
        "finishers": len(rows),
        "first_race": f"{first['season']} round {first['round']}",
        "last_race": f"{last['season']} round {last['round']}",
    }
    path = save_ranker(model, run_name("ranker", feature_set), trained_on, temperature=temperature)
    _print_training(model, trained_on, seconds, temperature)
    console.print(f"Saved the model to {path} (description: {path.with_suffix('.json').name})")


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


def _print_fetch_summary(summaries: list[SeasonSummary]) -> None:
    """Show one row per season: rows saved, plus responses downloaded, retried and cached."""
    table = Table(title="Jolpica download")
    columns = ("Season", "Races", "Results", "Quali", "Sprint", "Fetched", "Retries", "Cached")
    for column in columns:
        table.add_column(column, justify="right")
    for summary in summaries:
        table.add_row(
            str(summary.season),
            str(summary.races),
            str(summary.results),
            str(summary.qualifying),
            str(summary.sprint),
            str(summary.downloads),
            str(summary.retries),
            str(summary.cache_hits),
        )
    console.print(table)
    for summary in summaries:
        for warning in summary.warnings:
            console.print(f"[yellow]Warning:[/yellow] {warning}")
    console.print(f"Tables saved under {config.INTERIM_DIR}")


def _print_practice_summary(summaries: list[PracticeSummary]) -> None:
    """Show one row per season: practice sessions, laps, mapped drivers and stand-ins."""
    table = Table(title="OpenF1 practice download")
    columns = ("Season", "Sessions", "Laps", "Drivers", "Stand-ins", "Fetched", "Retries", "Cached")
    for column in columns:
        table.add_column(column, justify="right")
    for summary in summaries:
        table.add_row(
            str(summary.season),
            str(summary.sessions),
            str(summary.laps),
            str(summary.drivers),
            str(summary.practice_only),
            str(summary.downloads),
            str(summary.retries),
            str(summary.cache_hits),
        )
    console.print(table)
    for summary in summaries:
        for warning in summary.warnings:
            console.print(f"[yellow]Warning:[/yellow] {warning}")


def _validation_temperature(features: pd.DataFrame, feature_set: str) -> float:
    """Fit the simulation temperature on the validation season, with a progress bar."""
    from f1predict.evaluate.backtest import completed_races
    from f1predict.evaluate.tuning import validation_temperature

    season = config.VALIDATION_SEASON
    races = len(completed_races(features, [season]))
    if not races:
        console.print(f"[bold red]No completed {season} races to fit a temperature on.[/bold red]")
        raise typer.Exit(code=1)
    with _race_progress(races, f"Fitting T on {season}") as progress:
        temperature = validation_temperature(features, feature_set, season, progress)
    console.print(
        f"Temperature T = {temperature:.3f}, fitted on the {races} races of {season}, each "
        "scored by a ranker trained only on earlier races."
    )
    return temperature


def _load_features_or_exit() -> pd.DataFrame:
    """The features table, or a clear message and exit code 1 if it hasn't been built."""
    from f1predict.features.build import load_features

    try:
        return load_features()
    except FileNotFoundError as exc:
        console.print(f"[bold red]No features yet:[/bold red] {exc}")
        raise typer.Exit(code=1) from exc


@contextmanager
def _race_progress(total: int, description: str) -> Iterator[Callable[[str], None]]:
    """A live progress bar over ``total`` races; yields the callback to call after each."""
    columns = (
        TextColumn("{task.description}"),
        BarColumn(bar_width=20),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        TextColumn("{task.fields[race]}"),
    )
    with Progress(*columns, console=console) as progress:
        task = progress.add_task(description, total=total, race="")
        yield lambda label: progress.update(task, advance=1, race=label)


def _season_label(season: int) -> str:
    """Column header for a season, marking the validation and test seasons."""
    if season == config.VALIDATION_SEASON:
        return f"{season}\nvalidation"
    if season in config.TEST_SEASONS:
        return f"{season}\ntest"
    return str(season)


def _verdict_style(better: str, low: float, high: float) -> str:
    """Green if a difference's CI is entirely on the good side of 0, red if on the bad side."""
    improved = low > 0 if better == "higher" else high < 0
    worsened = high < 0 if better == "higher" else low > 0
    return "green" if improved else "red" if worsened else ""


def _with_ci(fmt: str, value: float, low: float, high: float, style: str = "") -> Text:
    """A value with its 95% confidence interval dimmed on the line below."""
    interval = f"[{fmt.format(low)}, {fmt.format(high)}]"
    return Text.assemble((fmt.format(value), style), "\n", (interval, "dim"))


def _print_season_table(
    summary: pd.DataFrame, title: str, value: str = "value", signed: bool = False
) -> None:
    """One row per metric, one column per season: the value, then its 95% CI.

    With ``signed``, the values are differences from the baseline, shown with a sign and
    coloured when the interval says the difference is clearly good or clearly bad.
    """
    from f1predict.evaluate.metrics import METRICS

    seasons = sorted(summary["season"].unique())
    table = Table(title=title)
    table.add_column("Metric")
    table.add_column("Better")
    for season in seasons:
        table.add_column(_season_label(season), justify="right")
    races = summary.groupby("season")["races"].first()
    table.add_row("Races", "", *(str(races[season]) for season in seasons))
    for metric, spec in METRICS.items():
        rows = summary[summary["metric"] == metric].set_index("season")
        if rows.empty:
            continue
        fmt = spec.fmt.replace("{:", "{:+") if signed else spec.fmt
        cells: list[Text | str] = []
        for season in seasons:
            if season not in rows.index:
                cells.append("-")
                continue
            row = rows.loc[season]
            style = _verdict_style(spec.better, row["low"], row["high"]) if signed else ""
            cells.append(_with_ci(fmt, row[value], row["low"], row["high"], style))
        table.add_row(spec.label, spec.better, *cells)
    console.print(table)


def _print_tuning(table: pd.DataFrame, chosen: dict, current: dict, season: int) -> None:
    """The best combinations by mean Spearman, marking the one-SE choice with ``*``."""
    is_chosen = [{key: row[key] for key in chosen} == chosen for _, row in table.iterrows()]
    shown = list(range(min(10, len(table))))
    shown += [i for i, flag in enumerate(is_chosen) if flag and i not in shown]
    view = Table(title=f"Ranker settings on {season}, best mean Spearman first")
    for column in ("#", "Trees", "Leaves", "Min rows", "Gain", "Spearman", "MAE", "Winner"):
        view.add_column(column, justify="right")
    for i in shown:
        row = table.iloc[i]
        view.add_row(
            f"{i + 1}*" if is_chosen[i] else str(i + 1),
            str(row["n_estimators"]),
            str(row["num_leaves"]),
            str(row["min_child_samples"]),
            row["gain"],
            f"{row['spearman']:.3f}",
            f"{row['mae']:.2f}",
            f"{row['winner_correct']:.0%}",
            style="bold" if is_chosen[i] else "",
        )
    console.print(view)
    best = table.iloc[0]
    console.print(
        f"Best mean Spearman {best['spearman']:.3f}, give or take {best['spearman_se']:.3f} "
        "(one standard error).\n"
        f"* Simplest within one standard error of it: {_settings_text(chosen)}."
    )
    if chosen == current:
        console.print("The ranker (f1predict/models/ranker.py) uses these settings.")
    else:
        console.print(
            f"[yellow]The ranker uses {_settings_text(current)}; "
            "update it to the choice above.[/yellow]"
        )


def _settings_text(settings: dict) -> str:
    """Settings as ``key=value`` pairs, e.g. ``n_estimators=300, gain=linear``."""
    return ", ".join(f"{key}={value}" for key, value in settings.items())


def _print_training(
    model: RankerModel, trained_on: dict, seconds: float, temperature: float
) -> None:
    """What the final model learned from, and which features its trees leaned on."""
    span = f"{trained_on['first_race']} to {trained_on['last_race']}"
    summary = Table(title="Trained ranker", show_header=False)
    summary.add_column("Setting", style="bold")
    summary.add_column("Value")
    summary.add_row("Feature set", f"{model.feature_set} ({len(model.features)} features)")
    summary.add_row("Races", f"{trained_on['races']} ({span})")
    summary.add_row("Finishers", str(trained_on["finishers"]))
    summary.add_row("Settings", _settings_text(model.settings()))
    summary.add_row("Temperature", f"{temperature:.3f} (fitted on {config.VALIDATION_SEASON})")
    summary.add_row("Training time", f"{seconds:.1f} s")
    console.print(summary)

    importance = model.importance()
    view = Table(title="What the trees lean on (share of total gain)")
    view.add_column("Feature")
    view.add_column("Share", justify="right")
    for row in importance.head(10).itertuples():
        view.add_row(row.feature, f"{row.gain_share:.1%}")
    rest = importance["gain_share"].iloc[10:].sum()
    if len(importance) > 10:
        view.add_row(f"{len(importance) - 10} others", f"{rest:.1%}")
    console.print(view)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the CLI (used by ``python -m f1predict`` and the ``f1predict`` command)."""
    app()


if __name__ == "__main__":
    main()
