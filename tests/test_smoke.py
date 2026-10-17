"""Smoke tests: the package imports, every module imports, and the CLI starts."""

from __future__ import annotations

import importlib
import os
import pkgutil
import re
import socket
import subprocess
import sys

import pytest
from pytest_socket import SocketBlockedError
from typer.testing import CliRunner

import f1predict
from f1predict.__main__ import app

runner = CliRunner()

# ---------------------------------------------------------------------------
# Package
# ---------------------------------------------------------------------------


def test_version_is_semver() -> None:
    """The package exposes a MAJOR.MINOR.PATCH version string."""
    assert re.fullmatch(r"\d+\.\d+\.\d+", f1predict.__version__)


def test_every_module_imports() -> None:
    """Every module in the package imports without errors."""
    names = [m.name for m in pkgutil.walk_packages(f1predict.__path__, prefix="f1predict.")]
    assert "f1predict.config" in names
    for name in names:
        importlib.import_module(name)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_help_lists_commands() -> None:
    """``--help`` exits cleanly and lists the available commands."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "info" in result.output


def test_cli_version_prints_version() -> None:
    """``--version`` prints the package version."""
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert f1predict.__version__ in result.output


def test_cli_info_shows_data_directory() -> None:
    """``info`` runs and reports where data is stored."""
    result = runner.invoke(app, ["info"])
    assert result.exit_code == 0
    assert "Data directory" in result.output


def test_python_dash_m_entry_point() -> None:
    """``python -m f1predict --help`` works as a real separate process."""
    completed = subprocess.run(
        [sys.executable, "-m", "f1predict", "--help"],
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        timeout=120,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "info" in completed.stdout


# ---------------------------------------------------------------------------
# Test isolation
# ---------------------------------------------------------------------------


@pytest.mark.filterwarnings("ignore:A test tried to use socket.socket")
def test_network_is_blocked_in_tests() -> None:
    """Opening a socket inside a test fails, so no test can quietly hit the internet."""
    with pytest.raises(SocketBlockedError):
        socket.socket(socket.AF_INET, socket.SOCK_STREAM)
