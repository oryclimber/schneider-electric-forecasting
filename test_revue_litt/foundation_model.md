# Literature review: can a tabular foundation model beat LOCF?

Paper: Paramasivaiah (2026), *Bridging Tabular Foundation Models and Commodity
Price Forecasting*, Journal of Data Processing 16(1),
[doi:10.6025/jdp/2026/16/1/31-45](https://doi.org/10.6025/jdp/2026/16/1/31-45).

## The idea of the paper

The paper reviews tabular foundation models such as TabPFN, TabICL and TabFM.
They are pre-trained on a large number of synthetic tables. To forecast, they read the
training rows as context and predict directly, without being trained on our data.
They are designed for small tables, and we only have about ten years of monthly
prices.

The paper does not test these models on commodity prices: its Brent forecasts use
ETS, SARIMAX and a random forest. We test one of them on our data.

## How we use it

We use [TabICL](https://arxiv.org/abs/2502.05564) (Qu et al., 2025). TabPFN needs
a licence accepted with a personal account, and TabFM (6.6 GB) takes several
minutes per forecast on a laptop.

For a material price $P_t$ and a horizon of $h$ months:

$$\text{TabICL:} \quad \hat{P}_{t+h} = P_t \, e^{\, m}$$

- $m$: median of $\ln(P_{t+h} / P_t)$ predicted by TabICL from the past rows
- inputs: $r_1$ and $r_3$ (price changes over 1 and 3 months), $U$ and $\Delta U$
  (US capacity utilisation in primary metals and its 3-month change)
- P10 and P90 use the 10% and 90% quantiles instead of the median

The inputs, dates and training rows are the same as for the ridge in `README.md`.
TabICL keeps its default settings, with no tuning.

## Run it

```bash
uv run --locked --with tabicl python test_revue_litt/run_foundation.py
```

TabICL is not in `uv.lock`, to keep PyTorch out of the team environment. The
first run downloads the model weights from Hugging Face; the data stays on the
computer. It takes about 20 minutes on a laptop CPU. Forecasts are saved in
`data/test_revue_litt/tabicl_predictions.csv` and are not in Git.

## Results

Error divided by the LOCF error, average over 12 materials (below 1 is better):

| Model | 3 months | 6 months | 9 months |
|---|---|---|---|
| TabICL, all dates | 0.999 | 0.996 | 1.082 |
| TabICL, since 2025 | 1.020 | 0.956 | 1.059 |
| Ridge with the same inputs, all dates | 0.956 | 0.937 | 0.942 |

TabICL beats LOCF in 13 of the 36 material and horizon pairs.

Share of real prices compared with the P10-P90 range (target: 80% inside):

| Horizon | Below P10 | Inside | Above P90 |
|---|---|---|---|
| 3 months | 13% | 63% | 23% |
| 6 months | 17% | 55% | 28% |
| 9 months | 20% | 52% | 29% |

## Why we did not keep it

- **No gain.** TabICL does as well as LOCF at 3 and 6 months and 8% worse at
  9 months. A ridge with the same inputs does better.
- **Ranges too narrow.** The P10-P90 range holds 52% to 63% of real prices instead
  of 80%, and most misses are price increases.
- **Not stable.** Since 2025, it beats LOCF only at 6 months.
- **Heavy.** It adds PyTorch and a model download for no gain.
- **Favourable timing.** As in `README.md`, the price of the forecast month is
  assumed known.
