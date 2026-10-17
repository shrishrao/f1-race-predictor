# f1-race-predictor

[![CI](https://github.com/shrishrao/f1-race-predictor/actions/workflows/ci.yml/badge.svg)](https://github.com/shrishrao/f1-race-predictor/actions/workflows/ci.yml)

Predicts a Formula 1 race right after qualifying: the full finishing order, plus each driver's
chance to **win**, finish on the **podium**, score **points** (top 10), and **retire** (DNF).

> **Status: in development.** The project skeleton, tests and CI are in place. Data collection,
> features, models, the CLI and a live Streamlit demo are being added step by step.

## How it works

1. **Data**: race results, qualifying, sprints and schedules from the
   [Jolpica F1 API](https://github.com/jolpica/jolpica-f1) for 2018 to 2026; practice laps from
   [OpenF1](https://openf1.org) for 2023 to 2026 (earlier seasons have no practice features).
2. **Features**: leakage-safe, form-based features only (grid, qualifying gaps, recent form,
   practice pace). No team or driver identity features, because the 2026 regulation reset broke
   many of the old patterns.
3. **Model**: a LightGBM ranker scores every driver, and a Plackett-Luce simulation turns those
   scores into probabilities over 100,000 simulated races. A separate DNF model decides who
   retires in each simulated race.
4. **Evaluation**: a race-by-race backtest (each race is predicted using only earlier races)
   against a simple baseline: "everyone finishes where they started".

## Quickstart (Windows PowerShell)

Requires Python 3.13 and Git.

```powershell
git clone https://github.com/shrishrao/f1-race-predictor.git
cd f1-race-predictor
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements-dev.txt
pytest
python -m f1predict info
```

## Project layout

```
f1predict/        the Python package: data, features, models, evaluate, CLI
app/              Streamlit demo app
tests/            pytest suite (runs offline; network access is blocked)
data/published/   small precomputed files the demo app reads (raw caches stay local)
models/           trained model files
docs/             plots used in this README
```

## Data sources and licenses

- Code: MIT (see [LICENSE](LICENSE)).
- Race, qualifying, sprint and schedule data come from the Jolpica F1 API under
  [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) (non-commercial use).
  Files in this repo derived from that data (test fixtures and published artifacts) carry the
  same license.
- Practice laps come from the OpenF1 API, also under
  [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) (non-commercial use).
  OpenF1 is used because Formula 1's own timing archive started requiring an F1 TV login in
  October 2026.
- This is an unofficial project and is not associated in any way with the Formula 1 companies.
  F1, FORMULA ONE, FORMULA 1, FIA FORMULA ONE WORLD CHAMPIONSHIP, GRAND PRIX and related marks
  are trademarks of Formula One Licensing B.V.
