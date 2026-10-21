"""About page: what the app shows, how a prediction is made, its limits, and the data."""

from __future__ import annotations

import streamlit as st

st.title("About f1predict")
st.markdown(
    """
f1predict predicts a Formula 1 race right after qualifying: the full finishing order, and each
driver's chance to win, finish on the podium, score points (top 10) and retire.

### How a prediction is made

1. **Inputs.** For every car: its grid slot and qualifying gaps, recent form (finishes, team
   points, retirement rates), the sprint result on sprint weekends, and practice pace. Each
   uses only what was known before the race started.
2. **Ranker.** A LightGBM ranking model scores every car. It learned from the order cars
   finished in, in every earlier race since 2018.
3. **Retirements.** A logistic regression gives each car a chance of retiring, from its grid
   slot, qualifying gap, recent retirement rates and the whole field's recent rate.
4. **Simulation.** 100,000 races are simulated. Each one draws a finishing order from the
   scores (a Plackett-Luce model, with its randomness sized on the 2024 season) and lets cars
   retire with their own chances. How often each car wins, reaches the podium or scores
   points gives its chances.

### How it's tested

Every 2025 and 2026 race was predicted by models trained only on the races before it, and
compared with the grid baseline: everyone finishes where they start. The model's settings,
the simulation's randomness and the DNF model were all chosen on 2024, so the test seasons
played no part in choosing them. The results are on **How good is it?**

### What it can't do

- **Qualifying is most of the signal.** The model orders the field a little better than the
  grid alone, but the test seasons can't confirm the gain, and its chances are about as good
  as the baseline's, not better.
- **Retirements are mostly luck.** The DNF model tells riskier cars apart only a little, and
  it's slow to notice a season-wide change like 2026's jump in retirements.
- **2026 is a new era.** New cars and engines mean form from earlier seasons carries over
  less.
- **Practice laps add nothing measurable**, and there are none in the data before 2023.
- No predictions before qualifying, during a race, or for sprint races.

### Data and licences

Results, qualifying, sprints and calendars come from the
[Jolpica F1 API](https://github.com/jolpica/jolpica-f1), and practice laps from
[OpenF1](https://openf1.org), both under
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) (non-commercial use).
The predictions shown here are derived from that data and carry the same licence. The code
is MIT-licensed: [github.com/shrishrao/f1-race-predictor](https://github.com/shrishrao/f1-race-predictor).

This is an unofficial project and is not associated in any way with the Formula 1 companies.
F1, FORMULA ONE, FORMULA 1, FIA FORMULA ONE WORLD CHAMPIONSHIP, GRAND PRIX and related marks
are trademarks of Formula One Licensing B.V.
"""
)
