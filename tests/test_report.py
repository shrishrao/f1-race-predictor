"""Tests for the results report: number formatting, the README markers, and the full command.

The end-to-end tests run the real commands on the synthetic championship (backtests,
``train``, then ``report``) in a temporary folder with its own README, so the repo's own
README and docs are never touched.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest
from synthetic import make_tables
from typer.testing import CliRunner

from f1predict import config
from f1predict.__main__ import app
from f1predict.evaluate import report
from f1predict.evaluate.metrics import METRICS, clear_difference
from f1predict.features.build import build_features, save_features
from f1predict.log import LOGGER_NAME

runner = CliRunner()
README = (
    "# Title\n\nIntro.\n\n## Results\n\n"
    "<!-- results:start -->\nSTALE TABLE\n<!-- results:end -->\n\nEnd.\n"
)
PNG_MAGIC = b"\x89PNG"

# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("fmt", "value", "signed", "text"),
    [
        ("{:.2f}", 0.774, False, "0.77"),
        ("{:.3f}", -0.009, True, "\N{MINUS SIGN}0.009"),
        ("{:.0%}", 0.083, True, "+8%"),
        ("{:.2f}", -0.001, True, "0.00"),
        ("{:.0%}", 0.0, True, "0%"),
        ("{:.2f}", float("nan"), False, "\N{EN DASH}"),
    ],
)
def test_numbers_use_a_true_minus_and_no_sign_on_zero(fmt, value, signed, text) -> None:
    """A rounded zero is plain "0.00", never "-0.00" or "+0%"; missing values are a dash."""
    assert report.number(fmt, value, signed) == text


def test_clear_differences_are_bold_or_italic() -> None:
    """Bold when the CI is all on the good side, italic when all on the bad side."""
    mae, spearman = METRICS["mae"], METRICS["spearman"]
    assert report.difference(mae, -0.32, -0.56, -0.08).startswith("**")
    assert report.difference(spearman, -0.099, -0.145, -0.054).startswith("*\N{MINUS SIGN}")
    unclear = report.difference(spearman, 0.018, -0.012, 0.052)
    assert unclear == "+0.018 [\N{MINUS SIGN}0.012, +0.052]"


def test_which_side_of_zero_is_good_depends_on_the_metric() -> None:
    """A drop in MAE is good, a drop in Spearman is bad; a CI across 0 settles nothing."""
    assert clear_difference("lower", -0.5, -0.1) == 1
    assert clear_difference("higher", -0.5, -0.1) == -1
    assert clear_difference("higher", -0.1, 0.2) == 0


def test_markdown_tables_align_numbers_right() -> None:
    """Text columns left, number columns right, one line per row."""
    table = report.markdown_table(["Metric", "2025"], [["Spearman", "0.77"]])
    assert table.splitlines() == ["| Metric | 2025 |", "| :--- | ---: |", "| Spearman | 0.77 |"]


def test_wilson_intervals_stay_inside_zero_to_one() -> None:
    """0 of 10 and 10 of 10 give one-sided intervals; 5 of 10 is centred on a half."""
    low, high = report.wilson_interval(np.array([0, 5, 10]), np.array([10, 10, 10]))
    assert low[0] == pytest.approx(0) and high[0] == pytest.approx(0.2775, abs=1e-4)
    assert low[1] == pytest.approx(0.2366, abs=1e-4) and high[1] == pytest.approx(0.7634, abs=1e-4)
    assert high[2] == pytest.approx(1) and low[2] == pytest.approx(0.7225, abs=1e-4)


def test_inputs_below_one_percent_fold_into_one_bar() -> None:
    """Three tiny inputs become "3 other inputs" with their summed share."""
    importance = [
        {"feature": "grid", "gain_share": 0.9},
        {"feature": "team_pts_3", "gain_share": 0.094},
        *({"feature": f"x{i}", "gain_share": 0.002} for i in range(3)),
    ]
    labels, values = report.gain_bars(importance)
    assert labels == ["Grid slot", "Team points, last 3 races", "3 other inputs"]
    assert values == pytest.approx([0.9, 0.094, 0.006])


# ---------------------------------------------------------------------------
# The README markers
# ---------------------------------------------------------------------------


def test_only_the_marked_section_is_replaced() -> None:
    """Text before and after the markers survives; running twice changes nothing more."""
    once = report.replace_marked_section(README, "new table")
    assert "STALE TABLE" not in once
    assert once.startswith("# Title") and once.endswith("End.\n")
    assert "<!-- results:start -->\nnew table\n<!-- results:end -->" in once
    assert report.replace_marked_section(once, "new table") == once


def test_a_readme_without_markers_is_refused() -> None:
    """Without both markers, in order, there is no safe place to write."""
    with pytest.raises(ValueError, match="results:start"):
        report.replace_marked_section("# No markers\n", "table")
    with pytest.raises(ValueError, match="results:start"):
        report.replace_marked_section(f"{report.END}\n{report.START}\n", "table")


# ---------------------------------------------------------------------------
# The command, end to end on the synthetic championship
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def workspace(tmp_path_factory):
    """Backtests (baseline, ranker, one ablation) and trained models in a temporary folder."""
    root = tmp_path_factory.mktemp("report")
    with pytest.MonkeyPatch.context() as patch:
        for name, folder in (
            ("PROCESSED_DIR", "processed"),
            ("MODELS_DIR", "models"),
            ("LOG_DIR", "logs"),
            ("DOCS_DIR", "docs"),
        ):
            patch.setattr(config, name, root / folder)
        patch.setattr(config, "README_PATH", root / "README.md")
        (root / "README.md").write_text(README, encoding="utf-8")
        save_features(build_features(make_tables()))
        for args in (
            ["backtest", "--season", "2024"],
            ["backtest", "--model", "ranker", "--season", "2024"],
            ["backtest", "--model", "ranker", "--features", "no_grid_quali", "--season", "2024"],
            ["train"],
        ):
            result = runner.invoke(app, args)
            assert result.exit_code == 0, result.output
        yield root
    logger = logging.getLogger(LOGGER_NAME)
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()


@pytest.fixture(scope="module")
def reported(workspace):
    """The output of ``report`` run once on the workspace."""
    result = runner.invoke(app, ["report"])
    assert result.exit_code == 0, result.output
    return result.output


def test_report_writes_the_tables_and_both_versions_of_each_plot(workspace, reported) -> None:
    """results.md with every section, and light and dark PNGs of both plots."""
    docs = workspace / "docs"
    text = (docs / "results.md").read_text(encoding="utf-8")
    for heading in ("## Full model vs the grid baseline", "## Ablations", "## DNF models"):
        assert heading in text
    assert "No grid or qualifying" in text
    for name in ("calibration", "calibration-dark", "importance", "importance-dark"):
        assert (docs / f"{name}.png").read_bytes().startswith(PNG_MAGIC)


def test_report_fills_the_readme_between_the_markers(workspace, reported) -> None:
    """The generated block replaces the old one; everything else in the README stays."""
    readme = (workspace / "README.md").read_text(encoding="utf-8")
    assert "STALE TABLE" not in readme
    assert report.GENERATED_NOTE in readme
    assert '<source media="(prefers-color-scheme: dark)"' in readme
    assert readme.startswith("# Title") and readme.endswith("End.\n")


def test_report_writes_lf_line_endings_on_every_os(workspace, reported) -> None:
    """Regression guard: Windows text mode would write CRLF, against .gitattributes."""
    for path in (workspace / "README.md", workspace / "docs" / "results.md"):
        assert b"\r\n" not in path.read_bytes()


def test_report_lists_the_backtests_it_left_out(reported) -> None:
    """The practice ablations and the no-DNF run weren't made, so it says how to make them."""
    words = " ".join(reported.split())
    assert "--features no_practice" in words
    assert "--no-dnf" in words


def test_running_the_report_again_changes_nothing(workspace, reported) -> None:
    """The same saved results give the same README, byte for byte."""
    before = (workspace / "README.md").read_bytes()
    assert runner.invoke(app, ["report"]).exit_code == 0
    assert (workspace / "README.md").read_bytes() == before


def test_report_without_backtests_says_what_to_run(tmp_path, monkeypatch) -> None:
    """With nothing saved yet, the error names the backtest command to run first."""
    monkeypatch.setattr(config, "PROCESSED_DIR", tmp_path / "processed")
    monkeypatch.setattr(config, "LOG_DIR", tmp_path / "logs")
    result = runner.invoke(app, ["report"])
    assert result.exit_code == 1
    assert "python -m f1predict backtest" in " ".join(result.output.split())
