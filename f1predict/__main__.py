"""Command-line interface: ``python -m f1predict <command>``.

Heavy libraries (pandas, requests, ...) are imported inside the commands that need them,
so ``--help`` and ``info`` start instantly.
"""

from __future__ import annotations

import platform
from typing import TYPE_CHECKING, Annotated

import typer
from rich.console import Console
from rich.table import Table
from rich.text import Text

from f1predict import __version__, config
from f1predict.log import setup_logging
from f1predict.models import ModelName

if TYPE_CHECKING:
    import pandas as pd

    from f1predict.data.fetch import PracticeSummary, SeasonSummary

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


@app.command("backtest")
def backtest_command(
    model: Annotated[ModelName, typer.Option(help="Model to evaluate.")] = "baseline",
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
    from f1predict.evaluate.backtest import (
        PredictionError,
        completed_races,
        run_backtest,
        save_backtest,
    )
    from f1predict.features.build import load_features
    from f1predict.models import make_model

    setup_logging(log_file=config.LOG_DIR / "backtest.log")
    chosen = sorted(set(seasons or [config.VALIDATION_SEASON, *config.TEST_SEASONS]))
    try:
        features = load_features()
    except FileNotFoundError as exc:
        console.print(f"[bold red]No features yet:[/bold red] {exc}")
        raise typer.Exit(code=1) from exc
    if not completed_races(features, chosen):
        console.print(f"[bold red]No completed races to backtest in {chosen}.[/bold red]")
        raise typer.Exit(code=1)

    try:
        result = run_backtest(lambda: make_model(model), features, chosen)
    except PredictionError as exc:
        console.print(f"[bold red]Backtest failed:[/bold red] {exc}")
        raise typer.Exit(code=1) from exc
    folder = save_backtest(result, model)

    _print_backtest_summary(result.summary, model)
    missing = sorted(set(chosen) - set(result.summary["season"]))
    if missing:
        console.print(f"[yellow]Warning:[/yellow] no completed races for {missing}")
    console.print(f"Saved per-race results to {folder}")


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


def _season_label(season: int) -> str:
    """Column header for a season, marking the validation and test seasons."""
    if season == config.VALIDATION_SEASON:
        return f"{season}\nvalidation"
    if season in config.TEST_SEASONS:
        return f"{season}\ntest"
    return str(season)


def _with_ci(fmt: str, row: pd.Series) -> Text:
    """A season average with its 95% confidence interval dimmed on the line below."""
    interval = f"[{fmt.format(row['low'])}, {fmt.format(row['high'])}]"
    return Text.assemble(fmt.format(row["value"]), "\n", (interval, "dim"))


def _print_backtest_summary(summary: pd.DataFrame, model: str) -> None:
    """Show one row per metric and one column per season: average, then its 95% CI."""
    from f1predict.evaluate.metrics import METRICS

    seasons = sorted(summary["season"].unique())
    table = Table(title=f"Backtest: {model}")
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
        cells = [_with_ci(spec.fmt, rows.loc[s]) if s in rows.index else "-" for s in seasons]
        table.add_row(spec.label, spec.better, *cells)
    console.print(table)
    console.print(
        "Season averages over races, with 95% bootstrap confidence intervals below.\n"
        "MAE and Spearman compare the order of the cars that finished."
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the CLI (used by ``python -m f1predict`` and the ``f1predict`` command)."""
    app()


if __name__ == "__main__":
    main()
