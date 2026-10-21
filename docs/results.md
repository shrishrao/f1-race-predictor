# Results

Every race was predicted by models trained only on the races before it. Validation season: 2024 (the ranker's settings, the simulation temperature and the DNF model were chosen there). Test seasons: 2025 (24 races) and 2026 (17 races). Values are season averages over races; brackets are 95% bootstrap confidence intervals, and differences are paired (both models scored on the same races). **Bold**: clearly better (the interval excludes 0); *italic*: clearly worse.

## Full model vs the grid baseline

| Metric | Better | 2025 baseline | 2025 model | 2025 difference | 2026 baseline | 2026 model | 2026 difference |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| Winner accuracy | higher | 67% | 58% | −8% [−29%, +12%] | 71% | 76% | +6% [0%, +18%] |
| Podium hit rate | higher | 75% | 78% | +3% [−3%, +8%] | 63% | 65% | +2% [0%, +6%] |
| Position MAE | lower | 2.36 | 2.36 | −0.003 [−0.211, +0.205] | 2.02 | 1.93 | −0.086 [−0.351, +0.207] |
| Spearman | higher | 0.75 | 0.77 | +0.018 [−0.012, +0.052] | 0.81 | 0.84 | +0.022 [−0.020, +0.066] |
| Brier: win | lower | 0.027 | 0.029 | +0.002 [−0.004, +0.007] | 0.023 | 0.025 | +0.002 [−0.001, +0.005] |
| Brier: podium | lower | 0.058 | 0.056 | −0.002 [−0.009, +0.006] | 0.064 | 0.063 | −0.001 [−0.007, +0.004] |
| Brier: points | lower | 0.163 | 0.164 | 0.000 [−0.010, +0.011] | 0.168 | 0.173 | +0.005 [−0.010, +0.019] |
| Log loss: win | lower | 0.092 | 0.094 | +0.002 [−0.015, +0.018] | 0.086 | 0.087 | 0.000 [−0.024, +0.016] |
| Log loss: podium | lower | 0.203 | 0.195 | −0.008 [−0.030, +0.016] | 0.207 | 0.200 | −0.008 [−0.032, +0.012] |
| Log loss: points | lower | 0.500 | 0.499 | −0.001 [−0.026, +0.024] | 0.516 | 0.531 | +0.016 [−0.021, +0.050] |

Validation season (2024):

| Metric | Better | 2024 baseline | 2024 model | 2024 difference |
| :--- | :--- | ---: | ---: | ---: |
| Winner accuracy | higher | 46% | 58% | +12% [−4%, +29%] |
| Podium hit rate | higher | 67% | 69% | +3% [−3%, +8%] |
| Position MAE | lower | 2.17 | 1.85 | **−0.320 [−0.562, −0.076]** |
| Spearman | higher | 0.81 | 0.86 | **+0.049 [+0.009, +0.095]** |
| Brier: win | lower | 0.035 | 0.033 | −0.002 [−0.006, +0.002] |
| Brier: podium | lower | 0.070 | 0.069 | −0.001 [−0.007, +0.005] |
| Brier: points | lower | 0.137 | 0.132 | −0.005 [−0.019, +0.009] |
| Log loss: win | lower | 0.125 | 0.115 | −0.011 [−0.027, +0.004] |
| Log loss: podium | lower | 0.231 | 0.223 | −0.008 [−0.026, +0.009] |
| Log loss: points | lower | 0.441 | 0.429 | −0.012 [−0.044, +0.020] |

## Ablations

Below each value: the difference from the full model.

### 2024 (validation)

| Metric | Better | Full model | No practice | Simple practice only | No grid or qualifying | Grid baseline |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| Winner accuracy | higher | 58% | 58%<br><sub>0% [0%, 0%]</sub> | 58%<br><sub>0% [0%, 0%]</sub> | 25%<br><sub>*−33% [−54%, −17%]*</sub> | 46%<br><sub>−12% [−29%, +4%]</sub> |
| Podium hit rate | higher | 69% | 69%<br><sub>0% [0%, 0%]</sub> | 69%<br><sub>0% [0%, 0%]</sub> | 49%<br><sub>*−21% [−31%, −11%]*</sub> | 67%<br><sub>−3% [−8%, +3%]</sub> |
| Position MAE | lower | 1.85 | 1.85<br><sub>0.000 [0.000, 0.000]</sub> | 1.85<br><sub>0.000 [0.000, 0.000]</sub> | 2.60<br><sub>*+0.753 [+0.564, +0.944]*</sub> | 2.17<br><sub>*+0.320 [+0.076, +0.562]*</sub> |
| Spearman | higher | 0.86 | 0.86<br><sub>0.000 [0.000, 0.000]</sub> | 0.86<br><sub>0.000 [0.000, 0.000]</sub> | 0.76<br><sub>*−0.098 [−0.126, −0.071]*</sub> | 0.81<br><sub>*−0.049 [−0.095, −0.009]*</sub> |
| Brier: win | lower | 0.033 | 0.033<br><sub>0.000 [0.000, 0.000]</sub> | 0.033<br><sub>0.000 [0.000, 0.000]</sub> | 0.040<br><sub>*+0.007 [+0.003, +0.011]*</sub> | 0.035<br><sub>+0.002 [−0.002, +0.006]</sub> |
| Brier: podium | lower | 0.069 | 0.069<br><sub>0.000 [0.000, 0.000]</sub> | 0.069<br><sub>0.000 [0.000, 0.000]</sub> | 0.092<br><sub>*+0.024 [+0.015, +0.033]*</sub> | 0.070<br><sub>+0.001 [−0.005, +0.007]</sub> |
| Brier: points | lower | 0.132 | 0.132<br><sub>0.000 [0.000, 0.000]</sub> | 0.132<br><sub>0.000 [0.000, 0.000]</sub> | 0.156<br><sub>*+0.024 [+0.014, +0.035]*</sub> | 0.137<br><sub>+0.005 [−0.009, +0.019]</sub> |
| Log loss: win | lower | 0.115 | 0.115<br><sub>0.000 [0.000, 0.000]</sub> | 0.115<br><sub>0.000 [0.000, 0.000]</sub> | 0.142<br><sub>*+0.027 [+0.010, +0.043]*</sub> | 0.125<br><sub>+0.011 [−0.004, +0.027]</sub> |
| Log loss: podium | lower | 0.223 | 0.223<br><sub>0.000 [0.000, 0.000]</sub> | 0.223<br><sub>0.000 [0.000, 0.000]</sub> | 0.288<br><sub>*+0.065 [+0.042, +0.088]*</sub> | 0.231<br><sub>+0.008 [−0.009, +0.026]</sub> |
| Log loss: points | lower | 0.429 | 0.428<br><sub>0.000 [−0.001, 0.000]</sub> | 0.428<br><sub>0.000 [−0.001, 0.000]</sub> | 0.488<br><sub>*+0.060 [+0.035, +0.086]*</sub> | 0.441<br><sub>+0.012 [−0.020, +0.044]</sub> |

### 2025 (test)

| Metric | Better | Full model | No practice | Simple practice only | No grid or qualifying | Grid baseline |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| Winner accuracy | higher | 58% | 58%<br><sub>0% [0%, 0%]</sub> | 58%<br><sub>0% [0%, 0%]</sub> | 25%<br><sub>*−33% [−58%, −8%]*</sub> | 67%<br><sub>+8% [−12%, +29%]</sub> |
| Podium hit rate | higher | 78% | 78%<br><sub>0% [0%, 0%]</sub> | 78%<br><sub>0% [0%, 0%]</sub> | 62%<br><sub>*−15% [−25%, −6%]*</sub> | 75%<br><sub>−3% [−8%, +3%]</sub> |
| Position MAE | lower | 2.36 | 2.36<br><sub>+0.004 [−0.031, +0.037]</sub> | 2.37<br><sub>+0.015 [−0.005, +0.042]</sub> | 3.07<br><sub>*+0.711 [+0.427, +0.990]*</sub> | 2.36<br><sub>+0.003 [−0.205, +0.211]</sub> |
| Spearman | higher | 0.77 | 0.77<br><sub>+0.001 [−0.004, +0.006]</sub> | 0.77<br><sub>−0.002 [−0.005, +0.001]</sub> | 0.67<br><sub>*−0.099 [−0.145, −0.054]*</sub> | 0.75<br><sub>−0.018 [−0.052, +0.012]</sub> |
| Brier: win | lower | 0.029 | 0.029<br><sub>*0.000 [0.000, 0.000]*</sub> | 0.029<br><sub>0.000 [0.000, 0.000]</sub> | 0.041<br><sub>*+0.012 [+0.007, +0.016]*</sub> | 0.027<br><sub>−0.002 [−0.007, +0.004]</sub> |
| Brier: podium | lower | 0.056 | 0.056<br><sub>0.000 [0.000, 0.000]</sub> | 0.056<br><sub>0.000 [0.000, 0.000]</sub> | 0.082<br><sub>*+0.026 [+0.015, +0.036]*</sub> | 0.058<br><sub>+0.002 [−0.006, +0.009]</sub> |
| Brier: points | lower | 0.164 | 0.164<br><sub>0.000 [0.000, +0.001]</sub> | 0.164<br><sub>0.000 [0.000, +0.001]</sub> | 0.189<br><sub>*+0.025 [+0.016, +0.035]*</sub> | 0.163<br><sub>0.000 [−0.011, +0.010]</sub> |
| Log loss: win | lower | 0.094 | 0.094<br><sub>0.000 [0.000, +0.001]</sub> | 0.094<br><sub>0.000 [0.000, +0.001]</sub> | 0.141<br><sub>*+0.047 [+0.035, +0.060]*</sub> | 0.092<br><sub>−0.002 [−0.018, +0.015]</sub> |
| Log loss: podium | lower | 0.195 | 0.195<br><sub>0.000 [−0.001, +0.001]</sub> | 0.195<br><sub>0.000 [−0.001, +0.001]</sub> | 0.268<br><sub>*+0.073 [+0.042, +0.103]*</sub> | 0.203<br><sub>+0.008 [−0.016, +0.030]</sub> |
| Log loss: points | lower | 0.499 | 0.500<br><sub>0.000 [0.000, +0.002]</sub> | 0.500<br><sub>+0.001 [0.000, +0.002]</sub> | 0.559<br><sub>*+0.060 [+0.039, +0.083]*</sub> | 0.500<br><sub>+0.001 [−0.024, +0.026]</sub> |

### 2026 (test)

| Metric | Better | Full model | No practice | Simple practice only | No grid or qualifying | Grid baseline |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| Winner accuracy | higher | 76% | 76%<br><sub>0% [0%, 0%]</sub> | 76%<br><sub>0% [0%, 0%]</sub> | 35%<br><sub>*−41% [−71%, −12%]*</sub> | 71%<br><sub>−6% [−18%, 0%]</sub> |
| Podium hit rate | higher | 65% | 65%<br><sub>0% [0%, 0%]</sub> | 65%<br><sub>0% [0%, 0%]</sub> | 51%<br><sub>−14% [−27%, 0%]</sub> | 63%<br><sub>−2% [−6%, 0%]</sub> |
| Position MAE | lower | 1.93 | 1.92<br><sub>−0.016 [−0.047, 0.000]</sub> | 1.93<br><sub>0.000 [0.000, 0.000]</sub> | 2.46<br><sub>*+0.525 [+0.320, +0.725]*</sub> | 2.02<br><sub>+0.086 [−0.207, +0.351]</sub> |
| Spearman | higher | 0.84 | 0.84<br><sub>+0.004 [0.000, +0.011]</sub> | 0.84<br><sub>0.000 [0.000, 0.000]</sub> | 0.79<br><sub>*−0.051 [−0.081, −0.020]*</sub> | 0.81<br><sub>−0.022 [−0.066, +0.020]</sub> |
| Brier: win | lower | 0.025 | 0.025<br><sub>0.000 [0.000, 0.000]</sub> | 0.025<br><sub>*0.000 [0.000, 0.000]*</sub> | 0.037<br><sub>*+0.012 [+0.007, +0.017]*</sub> | 0.023<br><sub>−0.002 [−0.005, +0.001]</sub> |
| Brier: podium | lower | 0.063 | 0.063<br><sub>0.000 [0.000, 0.000]</sub> | 0.063<br><sub>0.000 [0.000, 0.000]</sub> | 0.084<br><sub>*+0.021 [+0.006, +0.035]*</sub> | 0.064<br><sub>+0.001 [−0.004, +0.007]</sub> |
| Brier: points | lower | 0.173 | 0.173<br><sub>0.000 [0.000, 0.000]</sub> | 0.173<br><sub>0.000 [0.000, 0.000]</sub> | 0.183<br><sub>*+0.010 [+0.002, +0.017]*</sub> | 0.168<br><sub>−0.005 [−0.019, +0.010]</sub> |
| Log loss: win | lower | 0.087 | 0.087<br><sub>0.000 [0.000, 0.000]</sub> | 0.087<br><sub>*0.000 [0.000, 0.000]*</sub> | 0.130<br><sub>*+0.043 [+0.024, +0.059]*</sub> | 0.086<br><sub>0.000 [−0.016, +0.024]</sub> |
| Log loss: podium | lower | 0.200 | 0.200<br><sub>0.000 [0.000, +0.001]</sub> | 0.200<br><sub>0.000 [0.000, 0.000]</sub> | 0.261<br><sub>*+0.061 [+0.026, +0.096]*</sub> | 0.207<br><sub>+0.008 [−0.012, +0.032]</sub> |
| Log loss: points | lower | 0.531 | 0.531<br><sub>0.000 [0.000, 0.000]</sub> | 0.531<br><sub>0.000 [0.000, 0.000]</sub> | 0.553<br><sub>+0.021 [−0.002, +0.043]</sub> | 0.516<br><sub>−0.016 [−0.050, +0.021]</sub> |

## DNF models

Scored over the cars that started. `*` marks the model the simulation uses.

| Metric | Model | 2024 | 2025 | 2026 |
| :--- | :--- | ---: | ---: | ---: |
| Brier: DNF | Base rate | 0.094<br><sub>[0.070, 0.119]</sub> | 0.099<br><sub>[0.076, 0.123]</sub> | 0.158<br><sub>[0.138, 0.180]</sub> |
|  | Logistic* | 0.093<br><sub>[0.072, 0.115]</sub> | 0.097<br><sub>[0.075, 0.120]</sub> | 0.149<br><sub>[0.128, 0.172]</sub> |
|  | LightGBM | 0.093<br><sub>[0.073, 0.114]</sub> | 0.096<br><sub>[0.074, 0.119]</sub> | 0.152<br><sub>[0.129, 0.177]</sub> |
| Log loss: DNF | Base rate | 0.339<br><sub>[0.270, 0.409]</sub> | 0.353<br><sub>[0.291, 0.416]</sub> | 0.499<br><sub>[0.449, 0.553]</sub> |
|  | Logistic* | 0.333<br><sub>[0.280, 0.388]</sub> | 0.344<br><sub>[0.288, 0.400]</sub> | 0.470<br><sub>[0.416, 0.529]</sub> |
|  | LightGBM | 0.335<br><sub>[0.284, 0.386]</sub> | 0.339<br><sub>[0.282, 0.397]</sub> | 0.478<br><sub>[0.421, 0.541]</sub> |
| ROC AUC: DNF | Base rate | 0.505<br><sub>[0.399, 0.604]</sub> | 0.473<br><sub>[0.357, 0.585]</sub> | 0.481<br><sub>[0.433, 0.537]</sub> |
|  | Logistic* | 0.623<br><sub>[0.519, 0.716]</sub> | 0.575<br><sub>[0.480, 0.654]</sub> | 0.660<br><sub>[0.608, 0.718]</sub> |
|  | LightGBM | 0.620<br><sub>[0.510, 0.712]</sub> | 0.583<br><sub>[0.485, 0.666]</sub> | 0.664<br><sub>[0.612, 0.719]</sub> |

Each learned model minus the base rate:

| Metric | Model | 2024 | 2025 | 2026 |
| :--- | :--- | ---: | ---: | ---: |
| Brier: DNF | Logistic | −0.001 [−0.005, +0.003] | −0.002 [−0.005, +0.001] | **−0.009 [−0.013, −0.006]** |
|  | LightGBM | −0.001 [−0.005, +0.004] | −0.003 [−0.006, +0.001] | **−0.007 [−0.010, −0.003]** |
| Log loss: DNF | Logistic | −0.006 [−0.025, +0.013] | −0.009 [−0.024, +0.005] | **−0.030 [−0.040, −0.019]** |
|  | LightGBM | −0.004 [−0.026, +0.019] | −0.014 [−0.030, +0.003] | **−0.021 [−0.032, −0.008]** |

## Simulating retirements: with DNFs minus without

| Metric | Better | 2024 | 2025 | 2026 |
| :--- | :--- | ---: | ---: | ---: |
| Brier: win | lower | 0.000 [0.000, +0.001] | 0.000 [0.000, 0.000] | 0.000 [0.000, +0.001] |
| Brier: podium | lower | −0.001 [−0.002, +0.001] | 0.000 [−0.001, +0.001] | 0.000 [−0.002, +0.001] |
| Brier: points | lower | −0.001 [−0.005, +0.003] | −0.002 [−0.005, +0.001] | **−0.012 [−0.018, −0.006]** |
| Log loss: win | lower | +0.001 [−0.001, +0.002] | 0.000 [0.000, +0.001] | +0.001 [0.000, +0.002] |
| Log loss: podium | lower | −0.004 [−0.012, +0.003] | −0.001 [−0.009, +0.005] | −0.001 [−0.009, +0.004] |
| Log loss: points | lower | −0.036 [−0.095, +0.015] | −0.034 [−0.084, +0.007] | **−0.146 [−0.253, −0.061]** |

## Calibration (2025 and 2026 pooled)

The table behind `calibration.png`: the brackets are 95% Wilson intervals.

| Outcome | Chance bin | Model: cars | Model: given → happened | Baseline: cars | Baseline: given → happened |
| :--- | :--- | ---: | ---: | ---: | ---: |
| Win | 0%–10% | 732 | 1% → 0% [0%, 1%] | 730 | 1% → 0% [0%, 1%] |
| Win | 10%–20% | 53 | 15% → 9% [4%, 20%] | 41 | 11% → 7% [3%, 19%] |
| Win | 20%–30% | 26 | 25% → 31% [17%, 50%] | 41 | 23% → 20% [10%, 34%] |
| Win | 30%–40% | 17 | 35% → 53% [31%, 74%] | – | – |
| Win | 40%–50% | 17 | 45% → 82% [59%, 94%] | 12 | 49% → 75% [47%, 91%] |
| Win | 50%–60% | 8 | 52% → 50% [22%, 78%] | 29 | 51% → 66% [47%, 80%] |
| Podium (top 3) | 0%–10% | 605 | 2% → 1% [0%, 2%] | 607 | 2% → 1% [1%, 3%] |
| Podium (top 3) | 10%–20% | 61 | 14% → 16% [9%, 28%] | 82 | 15% → 12% [7%, 21%] |
| Podium (top 3) | 20%–30% | 36 | 24% → 28% [16%, 44%] | – | – |
| Podium (top 3) | 30%–40% | 27 | 33% → 30% [16%, 48%] | 41 | 35% → 44% [30%, 59%] |
| Podium (top 3) | 40%–50% | 21 | 46% → 57% [37%, 76%] | – | – |
| Podium (top 3) | 50%–60% | 29 | 55% → 52% [34%, 69%] | 41 | 54% → 51% [36%, 66%] |
| Podium (top 3) | 60%–70% | 23 | 65% → 78% [58%, 90%] | 11 | 69% → 82% [52%, 95%] |
| Podium (top 3) | 70%–80% | 24 | 77% → 88% [69%, 96%] | 71 | 75% → 79% [68%, 87%] |
| Podium (top 3) | 80%–90% | 27 | 85% → 85% [68%, 94%] | – | – |
| Points (top 10) | 0%–10% | – | – | 8 | 7% → 12% [2%, 47%] |
| Points (top 10) | 10%–20% | 41 | 19% → 2% [0%, 13%] | 230 | 15% → 15% [11%, 20%] |
| Points (top 10) | 20%–30% | 327 | 24% → 21% [17%, 26%] | 82 | 23% → 29% [21%, 40%] |
| Points (top 10) | 30%–40% | 116 | 35% → 43% [34%, 52%] | 82 | 36% → 24% [16%, 35%] |
| Points (top 10) | 40%–50% | 48 | 44% → 56% [42%, 69%] | 41 | 44% → 44% [30%, 59%] |
| Points (top 10) | 50%–60% | 34 | 55% → 65% [48%, 79%] | 74 | 55% → 57% [45%, 67%] |
| Points (top 10) | 60%–70% | 26 | 65% → 77% [58%, 89%] | 56 | 64% → 64% [51%, 76%] |
| Points (top 10) | 70%–80% | 34 | 75% → 79% [63%, 90%] | 116 | 76% → 75% [66%, 82%] |
| Points (top 10) | 80%–90% | 148 | 87% → 83% [76%, 88%] | 149 | 88% → 90% [84%, 94%] |
| Points (top 10) | 90%–100% | 79 | 91% → 91% [83%, 96%] | 15 | 90% → 93% [70%, 99%] |
| Retirement (DNF) | 0%–10% | 92 | 9% → 7% [3%, 14%] | – | – |
| Retirement (DNF) | 10%–20% | 661 | 14% → 13% [11%, 16%] | – | – |
| Retirement (DNF) | 20%–30% | 76 | 24% → 38% [28%, 49%] | – | – |
| Retirement (DNF) | 30%–40% | 8 | 31% → 25% [7%, 59%] | – | – |

## What the models lean on

Ranker: each input's share of the total gain (how much its splits improved the ranking).

| Input | Share of gain |
| :--- | ---: |
| Grid slot | 62.7% |
| Driver's average finish, last 5 | 15.5% |
| Team points, last 3 races | 11.2% |
| Qualifying gap to pole | 10.1% |
| Qualifying gap to teammate | 0.3% |
| Driver DNF rate | 0.1% |
| FP3 best-lap gap | 0.1% |
| Team DNF rate | 0.0% |
| Pit-lane start | 0.0% |
| Sprint result | 0.0% |
| FP1 best-lap gap | 0.0% |
| FP2 best-lap gap | 0.0% |
| Best practice-lap gap | 0.0% |
| FP1 long-run gap | 0.0% |
| FP2 long-run gap | 0.0% |
| FP3 long-run gap | 0.0% |
| Best long-run gap | 0.0% |

DNF model (Logistic regression): coefficient per SD.

| Input | Coefficient per sd |
| :--- | ---: |
| Grid slot | +0.239 |
| Team DNF rate | +0.143 |
| Qualifying gap to pole | +0.073 |
| Field DNF rate, recent races | +0.049 |
| Driver DNF rate | −0.028 |
| Pit-lane start | −0.015 |
| Circuit DNF rate | −0.014 |
