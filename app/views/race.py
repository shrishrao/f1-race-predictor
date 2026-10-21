"""Race page: one race's predicted order and chances, and for a race already run, the result."""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

import charts
import published

TILE_WIDTH = 220
ROW_HEIGHT = 35

# ---------------------------------------------------------------------------
# Picking a race
# ---------------------------------------------------------------------------

index = published.read("races.json")
races = index["races"]
start = published.default_race(races)

with st.sidebar:
    seasons = sorted({race["season"] for race in races}, reverse=True)
    season = st.selectbox("Season", seasons, index=seasons.index(start["season"]))
    in_season = {race["round"]: race for race in races if race["season"] == season}
    opening = (
        start if start["season"] == season else published.default_race(list(in_season.values()))
    )
    rounds = list(in_season)
    chosen = st.selectbox(
        "Race",
        rounds,
        index=rounds.index(opening["round"]),
        format_func=lambda number: published.race_option(in_season[number]),
    )
    st.caption(f"Data through {index['data_through']}. f1predict {index['f1predict_version']}.")

race = in_season[chosen]
st.title(race["name"])
st.caption(published.race_caption(race))
if race["status"] == "scheduled":
    st.info(published.not_qualified_text(race), icon=":material/schedule:")
    st.stop()

# ---------------------------------------------------------------------------
# The prediction
# ---------------------------------------------------------------------------

prediction = published.read(race["file"])
st.markdown(published.race_headline(prediction, race))
st.caption(published.how_it_was_made(prediction))

# Fixed-width tiles in a wrapping row: four across on a laptop, stacked in a narrow window.
with st.container(horizontal=True, wrap=True):
    for label, value, detail in published.race_tiles(prediction, race):
        st.metric(
            label,
            value,
            detail,
            delta_color="off",
            delta_arrow="off",
            border=True,
            width=TILE_WIDTH,
        )


def chance(label: str, help_text: str) -> st.column_config.ProgressColumn:
    """A chance column drawn as a bar from 0 to 100%."""
    return st.column_config.ProgressColumn(
        label, help=help_text, min_value=0, max_value=100, format="%.1f%%", color="primary"
    )


table = published.driver_table(prediction)
st.dataframe(
    table,
    hide_index=True,
    height=ROW_HEIGHT * (len(table) + 1) + 3,
    column_config={
        "#": st.column_config.NumberColumn("#", help="Predicted finishing position"),
        "Grid": st.column_config.TextColumn(
            "Grid", help="Starting slot. PL: pit lane. *: set after qualifying"
        ),
        "Win": chance("Win", "Chance of winning"),
        "Podium": chance("Podium", "Chance of finishing in the top 3"),
        "Points": chance("Points", "Chance of finishing in the top 10"),
        "DNF": chance("DNF", "Chance of retiring from the race"),
        "Result": st.column_config.TextColumn("Result", help="Official result. DNF: retired"),
    },
)
st.caption(
    f"Cars are listed in the ranker's predicted order. The chances come from "
    f"{prediction['model']['sims']:,} simulated races in which cars can also retire, so two "
    "cars close together can have their win chances the other way round."
)

# ---------------------------------------------------------------------------
# Against the result
# ---------------------------------------------------------------------------

if race["status"] == "completed":
    st.subheader("Prediction vs result")
    with st.container(horizontal=True, wrap=True, gap="large"):
        with st.container(width=560):
            st.plotly_chart(charts.prediction_vs_result(prediction), config=charts.CONFIG)
            st.caption(
                "Each dot is a car that finished. On the diagonal: it finished exactly where it "
                "was predicted; above it: better than predicted. Labelled: the winner and the "
                "three biggest misses."
            )
        with st.container(width=440):
            st.dataframe(published.scores_table(race), hide_index=True)
            st.caption(
                "Grid order is the baseline: everyone finishes where they start. Podium: "
                "predicted top 3 that finished top 3. Over the cars that finished, MAE is the "
                "average number of places off, and Spearman how closely the order matches "
                "(1 is perfect)."
            )
            st.write(published.non_finishers_text(prediction))

with st.expander("How this prediction was made"):
    st.markdown(published.model_card(prediction))
    st.download_button(
        "Download this prediction (JSON)",
        data=json.dumps(prediction, indent=2, ensure_ascii=False),
        file_name=Path(race["file"]).name,
        mime="application/json",
        icon=":material/download:",
    )
