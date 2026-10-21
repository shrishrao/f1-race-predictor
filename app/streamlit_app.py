"""f1predict's Streamlit app: Formula 1 race predictions made after qualifying.

Start it from the repo root with ``streamlit run app/streamlit_app.py``. It only reads the
files ``python -m f1predict publish`` writes into ``data/published``; it never trains a
model, predicts or downloads anything.
"""

from __future__ import annotations

import streamlit as st

import published

st.set_page_config(page_title="f1predict", page_icon="🏁", layout="wide")

if not (published.published_dir() / "races.json").exists():
    st.title("f1predict")
    st.error(
        "No published predictions found (data/published/races.json). "
        "Run `python -m f1predict publish` first."
    )
    st.stop()

# Page paths are relative to this file, as st.Page expects.
st.navigation(
    [
        st.Page("views/race.py", title="Race predictions", icon=":material/flag:", default=True),
        st.Page("views/performance.py", title="How good is it?", icon=":material/insights:"),
        st.Page("views/about.py", title="About", icon=":material/info:"),
    ],
    position="top",
).run()
