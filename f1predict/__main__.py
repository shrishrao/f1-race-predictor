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
    from f1predict.evaluate.dnf import DNFComparison
    from f1predict.models.dnf import DNFModel
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
    dnf: Annotated[
        bool,
        typer.Option(
            "--dnf/--no-dnf",
            help="Ranker only: let cars retire in the simulated races, using the DNF model "
            "chosen on 2024. --no-dnf simulates every car to the finish.",
        ),
    ] = True,
) -> None:
    """Predict every race of the chosen seasons using only earlier races, and score it.

    Prints season averages with 95% confidence intervals and saves the per-race results.
    For the ranker it first compares the three DNF models and picks one on 2024.
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
    from f1predict.evaluate.dnf import save_dnf_comparison
    from f1predict.evaluate.metrics import compare
    from f1predict.models import make_model, run_name
    from f1predict.models.dnf import make_dnf_model

    setup_logging(log_file=config.LOG_DIR / "backtest.log")
    chosen = sorted(set(seasons or [config.VALIDATION_SEASON, *config.TEST_SEASONS]))
    features = _load_features_or_exit()
    total = len(completed_races(features, chosen))
    if not total:
        console.print(f"[bold red]No completed races to backtest in {chosen}.[/bold red]")
        raise typer.Exit(code=1)

    name = run_name(model, feature_set, dnf)
    temperature = _validation_temperature(features, feature_set) if model == "ranker" else None
    make_dnf = None
    if model == "ranker" and dnf:
        comparison, kind, window = _choose_dnf(features, chosen)
        console.print(f"Saved the DNF comparison to {save_dnf_comparison(comparison)}")
        make_dnf = partial(make_dnf_model, kind, window)
    try:
        with _race_progress(total, name) as progress:
            predictions = predict_races(
                partial(make_model, model, feature_set, temperature, make_dnf),
                features,
                chosen,
                progress,
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
        "MAE and Spearman compare the order of the cars that finished; DNF chances are "
        "scored over the cars that started."
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
    """Train the ranker and the DNF model on every completed race and save them in models/.

    The simulation temperature and the DNF model are chosen on 2024 first.
    """
    from f1predict.models import run_name
    from f1predict.models.dnf import make_dnf_model, starters
    from f1predict.models.ranker import RankerModel, training_rows
    from f1predict.models.store import save_dnf_model, save_ranker

    setup_logging(log_file=config.LOG_DIR / "train.log")
    features = _load_features_or_exit()
    history = features[features["status_class"].notna()]
    rows = training_rows(history)
    if rows.empty:
        console.print("[bold red]No completed races to train on.[/bold red]")
        raise typer.Exit(code=1)

    temperature = _validation_temperature(features, feature_set)
    _, kind, window = _choose_dnf(features, [config.VALIDATION_SEASON])
    started = time.perf_counter()
    model = RankerModel(feature_set=feature_set).fit(history)
    seconds = time.perf_counter() - started
    dnf_model = make_dnf_model(kind, window).fit(history)
    first, last = rows.iloc[0], rows.iloc[-1]
    span = {
        "first_race": f"{first['season']} round {first['round']}",
        "last_race": f"{last['season']} round {last['round']}",
    }
    trained_on = {"races": int(rows["race_seq"].nunique()), "finishers": len(rows), **span}
    started_rows = starters(history)
    dnf_trained_on = {
        "races": int(started_rows["race_seq"].nunique()),
        "starters": len(started_rows),
        "dnfs": int(started_rows["status_class"].eq("dnf").sum()),
        **span,
    }
    path = save_ranker(model, run_name("ranker", feature_set), trained_on, temperature=temperature)
    dnf_path = save_dnf_model(dnf_model, dnf_trained_on)
    _print_training(model, trained_on, seconds, temperature)
    _print_dnf_training(dnf_model, dnf_trained_on)
    console.print(f"Saved the ranker to {path} (description: {path.with_suffix('.json').name})")
    console.print(f"Saved the DNF model to {dnf_path}")


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


def _choose_dnf(
    features: pd.DataFrame, seasons: list[int]
) -> tuple[DNFComparison, str, int | None]:
    """Choose the DNF window and model on the validation season, comparing all three models.

    The models are backtested over ``seasons`` plus the validation season (the choice is
    made there) and the comparison is printed. Returns it with the chosen kind and window.
    """
    from f1predict.evaluate.backtest import completed_races
    from f1predict.evaluate.dnf import compare_dnf_models
    from f1predict.evaluate.tuning import choose_dnf_kind, validation_dnf_window
    from f1predict.models.dnf import DNF_KINDS, WINDOWS

    season = config.VALIDATION_SEASON
    races = len(completed_races(features, [season]))
    if not races:
        console.print(f"[bold red]No completed {season} races to choose a DNF model on.[/bold red]")
        raise typer.Exit(code=1)
    with _race_progress(len(WINDOWS) * races, f"DNF window on {season}") as progress:
        window, windows = validation_dnf_window(features, season, progress=progress)
    compared = sorted({*seasons, season})
    total = len(DNF_KINDS) * len(completed_races(features, compared))
    with _race_progress(total, "DNF models") as progress:
        comparison = compare_dnf_models(features, compared, window, progress)
    kind = choose_dnf_kind(comparison.summary, season)
    _print_dnf_window(windows, window, season)
    _print_dnf_comparison(comparison, kind, window)
    return comparison, kind, window


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


def _print_dnf_window(windows: pd.DataFrame, chosen: int | None, season: int) -> None:
    """The base rate's validation score for every look-back window, marking the choice."""
    from f1predict.models.dnf import window_label

    view = Table(title=f"Field-wide DNF rate: look-back window on {season}")
    for column in ("Look back over", "Brier", "Log loss"):
        view.add_column(column, justify="right")
    for row in windows.itertuples():
        is_chosen = row.window == chosen
        view.add_row(
            window_label(row.window) + ("*" if is_chosen else ""),
            f"{row.brier_dnf:.4f}",
            f"{row.log_loss_dnf:.4f}",
            style="bold" if is_chosen else "",
        )
    console.print(view)
    console.print(f"* Lowest mean Brier score on {season}: {window_label(chosen)}.")


def _print_dnf_comparison(comparison: DNFComparison, chosen: str, window: int | None) -> None:
    """The three DNF models per season, then each learned model minus the base rate."""
    from f1predict.evaluate.metrics import AUC_METRIC, METRICS
    from f1predict.models.dnf import DNF_KINDS, KIND_LABELS, SHORT_LABELS, window_label

    specs = {"brier_dnf": METRICS["brier_dnf"], "log_loss_dnf": METRICS["log_loss_dnf"]}
    tables = (
        ("DNF models, cars that started", comparison.summary, specs | {"auc_dnf": AUC_METRIC}),
        ("DNF models minus the base rate", comparison.differences, specs),
    )
    for title, summary, metrics in tables:
        if summary.empty:
            continue
        signed = "difference" in summary
        seasons = sorted(summary["season"].unique())
        view = Table(title=title)
        view.add_column("Metric")
        view.add_column("Model")
        for season in seasons:
            view.add_column(_season_label(season), justify="right")
        for metric, spec in metrics.items():
            fmt = spec.fmt.replace("{:", "{:+") if signed else spec.fmt
            kinds = [kind for kind in DNF_KINDS if kind in set(summary["model"])]
            for i, kind in enumerate(kinds):
                rows = summary[(summary["metric"] == metric) & (summary["model"] == kind)]
                rows = rows.set_index("season")
                cells: list[Text | str] = []
                for season in seasons:
                    if season not in rows.index:
                        cells.append("-")
                        continue
                    row = rows.loc[season]
                    value = row["difference"] if signed else row["value"]
                    style = _verdict_style(spec.better, row["low"], row["high"]) if signed else ""
                    cells.append(_with_ci(fmt, value, row["low"], row["high"], style))
                label = SHORT_LABELS[kind] + ("*" if kind == chosen else "")
                view.add_row(spec.label if i == 0 else "", label, *cells)
            view.add_section()
        console.print(view)
    console.print(
        "ROC AUC: the chance a retiree was given a higher DNF chance than a finisher "
        "(0.5 = a coin toss).\nThe base rate gives every car in a race the same chance, so "
        "its AUC only reflects race-to-race changes.\n"
        f"* Chosen on {config.VALIDATION_SEASON} (lowest mean Brier score): "
        f"{KIND_LABELS[chosen]}, field-wide rate over the {window_label(window)}."
    )


def _print_dnf_training(model: DNFModel, trained_on: dict) -> None:
    """What the saved DNF model learned from, and what it leans on."""
    from f1predict.models.dnf import KIND_LABELS, LightGBMDNF, LogisticDNF, window_label

    span = f"{trained_on['first_race']} to {trained_on['last_race']}"
    share = trained_on["dnfs"] / trained_on["starters"]
    summary = Table(title="Trained DNF model", show_header=False)
    summary.add_column("Setting", style="bold")
    summary.add_column("Value")
    summary.add_row("Model", KIND_LABELS[model.kind])
    summary.add_row("Field-wide rate", f"over the {window_label(model.window)}")
    summary.add_row("Races", f"{trained_on['races']} ({span})")
    summary.add_row(
        "Starters", f"{trained_on['starters']} ({trained_on['dnfs']} DNFs, {share:.1%})"
    )
    console.print(summary)
    if isinstance(model, LogisticDNF):
        view = Table(title="Coefficients (log-odds of a DNF per standard deviation)")
        view.add_column("Input")
        view.add_column("Coefficient", justify="right")
        for row in model.coefficients().itertuples():
            view.add_row(row.feature, f"{row.coef:+.3f}")
        console.print(view)
    elif isinstance(model, LightGBMDNF):
        view = Table(title="What the trees lean on (share of total gain)")
        view.add_column("Input")
        view.add_column("Share", justify="right")
        for row in model.importance().itertuples():
            view.add_row(row.feature, f"{row.gain_share:.1%}")
        console.print(view)


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
