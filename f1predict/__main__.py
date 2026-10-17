"""Command-line interface: ``python -m f1predict <command>``."""

from __future__ import annotations

import platform
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from f1predict import __version__, config

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


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the CLI (used by ``python -m f1predict`` and the ``f1predict`` command)."""
    app()


if __name__ == "__main__":
    main()
