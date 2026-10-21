"""The app's Plotly charts.

Colours come from the app's theme (``.streamlit/config.toml``), not from here: each chart
adds its traces in a fixed order, so the theme's first categorical colour (blue) is always
the model and the second (orange) the grid baseline, in light and dark mode alike. Only the
reference lines use a fixed neutral gray, which reads on both backgrounds.

Mark specs follow the data-viz reference: bars at most about 20 px thick with rounded data
ends, 2 px lines, 8 px or larger markers, and a value at the tip of each bar.
"""

from __future__ import annotations

from typing import Any

import plotly.graph_objects as go

NEUTRAL = "#898781"
MIN_SHARE = 0.01
BAR_ROW = 30
OUTCOME_TITLES = {
    "win": "Win",
    "podium": "Podium (top 3)",
    "points": "Points (top 10)",
    "dnf": "Retirement (DNF)",
}
CONFIG = {"displayModeBar": False}

# ---------------------------------------------------------------------------
# Shared layout
# ---------------------------------------------------------------------------


def _layout(fig: go.Figure, height: int, title: str | None = None) -> go.Figure:
    """Fixed height, tight margins, a left-aligned title and the legend in a row on top."""
    fig.update_layout(
        height=height,
        margin={"l": 10, "r": 10, "t": 56 if title else 24, "b": 10},
        title={"text": title, "x": 0, "xanchor": "left"} if title else None,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.0, "x": 0, "xanchor": "left"},
        hoverlabel={"align": "left"},
    )
    return fig


# ---------------------------------------------------------------------------
# One race
# ---------------------------------------------------------------------------


def prediction_vs_result(prediction: dict[str, Any]) -> go.Figure:
    """Each finisher's predicted position (x) against where they finished (y).

    Dots on the diagonal were called exactly; above it, the car finished better than
    predicted. Retirements have no finishing position, so they aren't plotted. Only the
    three biggest misses and the winner carry a label; hovering shows every car.
    """
    drivers = prediction["drivers"]
    finishers = [d for d in drivers if d["actual"] and d["actual"]["status_class"] == "finished"]
    misses = sorted(
        finishers, key=lambda d: abs(d["predicted_rank"] - d["actual"]["position"]), reverse=True
    )
    labelled = {d["driver_id"] for d in misses[:3]}
    labelled |= {d["driver_id"] for d in finishers if d["actual"]["position"] == 1}
    fig = go.Figure(
        go.Scatter(
            x=[d["predicted_rank"] for d in finishers],
            y=[d["actual"]["position"] for d in finishers],
            mode="markers+text",
            text=[d["code"] if d["driver_id"] in labelled else "" for d in finishers],
            textposition="middle right",
            marker={"size": 10},
            customdata=[[d["name"] or d["driver_id"], d["team"] or ""] for d in finishers],
            hovertemplate=(
                "<b>%{customdata[0]}</b> %{customdata[1]}<br>"
                "Predicted %{x}, finished %{y}<extra></extra>"
            ),
            showlegend=False,
        )
    )
    cars = len(drivers)
    fig.add_shape(
        type="line",
        x0=1,
        y0=1,
        x1=cars,
        y1=cars,
        line={"color": NEUTRAL, "width": 1},
        layer="below",
    )
    ticks = [tick for tick in (1, 5, 10, 15, 20, 25) if tick <= cars]
    fig.update_xaxes(title="Predicted position", range=[0.4, cars + 1.6], tickvals=ticks)
    fig.update_yaxes(title="Finishing position", range=[cars + 0.6, 0.4], tickvals=ticks)
    return _layout(fig, 440)


# ---------------------------------------------------------------------------
# Backtest results
# ---------------------------------------------------------------------------


def calibration(metrics: dict[str, Any], outcome: str, show_legend: bool = False) -> go.Figure:
    """For one outcome, the chance given (x) against how often it happened (y).

    The model is added first and the grid baseline second (it gives no DNF chances), so
    each keeps its theme colour. Vertical bars are 95% intervals; the diagonal is perfect
    calibration.
    """
    limit = 0.5 if outcome == "dnf" else 1.0
    fig = go.Figure()
    for name, label in (("model", "Full model"), ("baseline", "Grid baseline")):
        rows = [r for r in metrics["calibration"] if r["model"] == name and r["outcome"] == outcome]
        if not rows and name == "baseline":
            continue
        fig.add_trace(
            go.Scatter(
                x=[row["predicted"] for row in rows],
                y=[row["observed"] for row in rows],
                name=label,
                mode="lines+markers",
                marker={"size": 8},
                line={"width": 2},
                error_y={
                    "type": "data",
                    "symmetric": False,
                    "array": [row["high"] - row["observed"] for row in rows],
                    "arrayminus": [row["observed"] - row["low"] for row in rows],
                    "thickness": 1,
                    "width": 0,
                },
                customdata=[[row["n"], row["low"], row["high"]] for row in rows],
                hovertemplate=(
                    "Given %{x:.0%}, happened %{y:.0%}<br>"
                    "95% interval %{customdata[1]:.0%} to %{customdata[2]:.0%}<br>"
                    "%{customdata[0]} cars<extra>%{fullData.name}</extra>"
                ),
                showlegend=show_legend,
            )
        )
    fig.add_shape(
        type="line",
        x0=0,
        y0=0,
        x1=limit,
        y1=limit,
        line={"color": NEUTRAL, "width": 1},
        layer="below",
    )
    axis = {"range": [0, limit], "tickformat": ".0%", "zeroline": False}
    fig.update_xaxes(title="Chance given", **axis)
    fig.update_yaxes(title="How often it happened", **axis)
    return _layout(fig, 360, OUTCOME_TITLES[outcome])


def _bars(labels: list[str], values: list[float], fmt: str, title: str) -> go.Figure:
    """Thin horizontal bars from a zero line, largest first, each value at its tip."""
    fig = go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker={"cornerradius": 4},
            texttemplate=f"%{{x:{fmt}}}",
            textposition="outside",
            cliponaxis=False,
            hovertemplate=f"%{{y}}: %{{x:{fmt}}}<extra></extra>",
            showlegend=False,
        )
    )
    low, high = min(0.0, *values), max(0.0, *values)
    reach = (high - low) or 1.0
    fig.update_xaxes(
        visible=False, range=[low - (0.3 * reach if low < 0 else 0), high + 0.3 * reach]
    )
    fig.update_yaxes(autorange="reversed", title=None)
    fig.add_vline(x=0, line={"color": NEUTRAL, "width": 1})
    fig.update_layout(bargap=0.35)
    return _layout(fig, 70 + BAR_ROW * len(values), title)


def ranker_importance(metrics: dict[str, Any]) -> go.Figure:
    """The ranker's share of total gain per input; inputs under 1% share one bar."""
    gains = metrics["importance"]["ranker"]
    shown = [row for row in gains if row["gain_share"] >= MIN_SHARE]
    rest = [row for row in gains if row["gain_share"] < MIN_SHARE]
    labels = [row["label"] for row in shown]
    values = [row["gain_share"] for row in shown]
    if rest:
        labels.append(f"{len(rest)} other inputs")
        values.append(sum(row["gain_share"] for row in rest))
    return _bars(labels, values, ".1%", "Ranker: share of total gain")


def dnf_importance(metrics: dict[str, Any]) -> go.Figure | None:
    """What the DNF model leans on (None for the base rate, which has no inputs)."""
    dnf = metrics["importance"]["dnf"]
    if not dnf["inputs"]:
        return None
    labels = [row["label"] for row in dnf["inputs"]]
    values = [row["value"] for row in dnf["inputs"]]
    if dnf["kind"] == "logistic":
        return _bars(labels, values, "+.2f", "DNF model: change in log-odds per SD")
    return _bars(labels, values, ".1%", "DNF model: share of total gain")
