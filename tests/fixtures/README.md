# Test fixtures

Small, real API responses saved so the tests never touch the network.

## `jolpica/`

Saved from the [Jolpica F1 API](https://github.com/jolpica/jolpica-f1) on 2026-10-17.
The data is licensed [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).

| File | Request | Why it is here |
|---|---|---|
| `schedule_2026.json` | `2026.json`, rounds 1, 2, 16, 17, 18 kept | normal and sprint weekends, Sepang, a future round |
| `schedule_2018.json` | `2018.json`, round 1 kept | old format: practice sessions have a date but no time |
| `results_2018_1.json` | `2018/1/results.json` | old status wording: "+1 Lap", "Engine", "Wheel" |
| `qualifying_2018_1.json` | `2018/1/qualifying.json` | qualifying in the old format |
| `results_2026_17.json` | `2026/17/results.json` | new wording ("Lapped", "Retired"), a classified retiree at P18 |
| `qualifying_2026_17.json` | `2026/17/qualifying.json` | a blank Q1 time and missing Q2/Q3 |
| `sprint_2026_17.json` | `2026/17/sprint.json` | sprint results, including a last-lap retiree at P14 |

## `openf1/`

Saved from the [OpenF1 API](https://openf1.org) on 2026-10-17. The data is licensed
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).

| File | Request | Why it is here |
|---|---|---|
| `sessions_2026.json` | `sessions?year=2026&session_type=Practice`, rounds 1, 2, 16, 17, 18 kept | normal and sprint weekends, a future round |
| `drivers_11378.json` | `drivers?session_key=11378` (Singapore 2026 FP1) | all 22 drivers, to check every one maps to Jolpica |
| `laps_11378.json` | `laps?session_key=11378`, cars 1, 18, 27 kept | real laps: out-laps, cool-down laps, untimed laps |
| `stints_11378.json` | `stints?session_key=11378`, cars 1, 18, 27 kept | tyre compounds and in-laps |
| `race_control_11378.json` | `race_control?session_key=11378` | yellows, a VSC, and three deleted lap times |
