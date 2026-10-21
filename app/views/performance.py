"""Performance page: how the model did on the test seasons, against the grid baseline."""

from __future__ import annotations

import streamlit as st

import charts
import published

TILE_WIDTH = 240
PANEL_WIDTH = 480

metrics = published.read("metrics.json")
seasons = metrics["test_seasons"]

st.title("How good is it?")
st.write(
    "Every race of the test seasons was predicted by models trained only on the races "
    "before it, then scored against what happened. The yardstick is the grid baseline: "
    "everyone finishes where they start, with win, podium and points chances from how often "
    "each grid slot produced them in earlier races. The model's settings were chosen on "
    f"{metrics['validation_season']}, so the test seasons played no part in choosing them."
)
season = st.segmented_control(
    "Season",
    seasons,
    default=seasons[-1],
    required=True,
    format_func=lambda year: published.season_label(metrics, year),
)

# A tile's gap is coloured (with an arrow) only when its interval clearly excludes 0.
with st.container(horizontal=True, wrap=True):
    for label, value, gap, verdict, better in published.season_tiles(metrics, season):
        clear = verdict != 0
        st.metric(
            label,
            value,
            gap,
            delta_color=("normal" if better == "higher" else "inverse") if clear else "off",
            delta_arrow="auto" if clear else "off",
            border=True,
            width=TILE_WIDTH,
        )

st.subheader("The full model against the grid baseline")
st.dataframe(published.comparison_table(metrics, season), hide_index=True)
st.caption(
    "Season averages over races, with 95% bootstrap intervals in brackets. The difference "
    "is paired (both models scored on the same races); too close to call means its interval "
    "includes 0. MAE and Spearman compare the order of the cars that finished; the chance "
    "scores (Brier, log loss) count every car."
)

st.subheader("Calibration")
pooled = " and ".join(str(year) for year in seasons)
st.write(
    f"Cars grouped by the chance they were given (0-10%, 10-20%, ...), {pooled} pooled. On "
    "the diagonal, things happened exactly as often as predicted; above it, more often. "
    "Vertical bars are 95% intervals: a long bar means few cars in that group."
)
with st.container(horizontal=True, wrap=True, gap="large"):
    for outcome in charts.OUTCOME_TITLES:
        figure = charts.calibration(metrics, outcome, show_legend=outcome == "win")
        st.plotly_chart(figure, config=charts.CONFIG, width=PANEL_WIDTH)
st.caption("The grid baseline gives no DNF chances, so the retirement panel shows the model only.")

st.subheader("What each group of features is worth")
st.dataframe(published.ablation_table(metrics, season), hide_index=True)
st.caption(
    "Each row is the whole model rebuilt without one group of inputs, with its own "
    "temperature and DNF model chosen on the validation season. Gaps are paired against the "
    "full model."
)

st.subheader("Retirements: the three DNF models")
st.dataframe(published.dnf_table(metrics, season), hide_index=True)
st.caption(
    "Scored over the cars that started. ROC AUC is the chance that a car that retired was "
    "given a higher DNF chance than one that finished (0.5 is a coin toss)."
)

st.subheader("What the models lean on")
with st.container(horizontal=True, wrap=True, gap="large"):
    st.plotly_chart(charts.ranker_importance(metrics), config=charts.CONFIG, width=PANEL_WIDTH)
    dnf_figure = charts.dnf_importance(metrics)
    if dnf_figure is None:
        st.write("The DNF model in use is the base rate: every car gets the field's recent rate.")
    else:
        st.plotly_chart(dnf_figure, config=charts.CONFIG, width=PANEL_WIDTH)
st.caption(
    "Ranker: how much each input's splits improved the ranking. DNF model: how each input "
    "moves the log-odds of a retirement, per standard deviation of that input."
)
