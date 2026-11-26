# Literature review: can a FRED indicator improve our forecasts?

Paper: Bastianin, Rossini and Tonni (2025), *A Real-Time Framework for
Forecasting Metal Prices*, [arXiv:2512.16521](https://arxiv.org/abs/2512.16521).

## The idea of the paper

When metal factories run close to full capacity, demand is strong and prices may
follow. The authors test whether this kind of indicator helps to forecast copper,
aluminium, nickel and zinc.

Such figures are revised after publication, so using today's values to test past
forecasts would cause data leakage. To avoid this, the paper only uses the figures
available at the time of each forecast (possible with ALFRED data).

They find that short horizons are very hard, but that new orders and capacity
utilisation help at medium horizons, mainly for copper and aluminium.

## How we use it

We forecast the materials of our submission at 3, 6 and 9 months with:

| Model | Information used |
|---|---|
| LOCF | Last price (our benchmark) |
| Trend | Price change over the last 1 and 3 months |
| FRED | Trend + capacity utilisation (level and 3-month change) |

For a material price $P_t$ and a horizon of $h$ months:

$$\text{LOCF:} \quad \hat{P}_{t+h} = P_t$$

$$\text{Trend:} \quad \hat{P}_{t+h} = P_t \, e^{\, a \, r_1 + b \, r_3}$$

$$\text{FRED:} \quad \hat{P}_{t+h} = P_t \, e^{\, a' r_1 + b' r_3 + c \, U + d \, \Delta U}$$

- $r_1 = \ln(P_t / P_{t-1})$ and $r_3 = \ln(P_t / P_{t-3})$: recent price changes
- $U$: last capacity utilisation published at date $t$
- $\Delta U$: its change over 3 months

All inputs are standardised. The coefficients are estimated on past data with a
ridge regression without intercept and a fixed penalty of 10, separately for each
material, horizon and date. Trend and FRED use the same settings, so the
difference between them shows what the indicator adds.

For each past month we train on earlier data only, forecast, and compare with the
real price. The result is the error divided by the LOCF error (below 1 is better).

Unlike the paper, we use nominal prices, LOCF as the benchmark, one indicator and
more materials than metals alone. It is an adaptation, not a replication.

## Where the data comes from

The indicator is [CAPUTLG331S](https://fred.stlouisfed.org/series/CAPUTLG331S),
the US capacity utilisation in primary metals, published monthly by the Federal
Reserve and available for free on FRED.

We download it from [ALFRED](https://alfred.stlouisfed.org), the FRED archive,
which keeps the figures as published at each date. No key is needed:

```text
https://alfred.stlouisfed.org/graph/alfredgraph.csv?id=CAPUTLG331S&vintage_date=2020-01-31
```

1. `download_capacity` gets one version per month end since 2016.
2. `capacity_features` keeps the last published level and its 3-month change. The
   figure usually comes out one month late.
3. `build_table` joins them to the price changes of each material. The row of
   March 2022 uses what was published on 31 March 2022.

## Run it

```bash
uv run --locked python test_revue_litt/run.py
```

The first run downloads the indicator. Forecasts and results by material are saved
in `data/test_revue_litt/` and are not in Git.

## Results

Error divided by the LOCF error, average over 12 materials (below 1 is better):

| Model | 3 months | 6 months | 9 months |
|---|---|---|---|
| Trend, all dates | 0.955 | 0.967 | 1.005 |
| FRED, all dates | 0.956 | 0.937 | 0.942 |
| Trend, since 2025 | 1.004 | 1.025 | 1.074 |
| FRED, since 2025 | 0.980 | 0.995 | 1.050 |

Forecast dates start in April, July and October 2020 for the three horizons, which
gives 75, 69 and 63 dates (18, 15 and 12 since 2025).

On all dates, the indicator adds nothing to Trend at 3 months. At 6 and 9 months,
the error is about 6% lower than LOCF. The paper also finds gains at medium
horizons, but mainly on copper and aluminium, while ours are mostly on European
steel and plastics.

## Why we did not keep it

- **Not significant.** With Diebold-Mariano tests, run separately from this code,
  no material beats LOCF or Trend after correction for the 36 comparisons (lowest
  adjusted p-value: 0.17 against LOCF, 0.84 against Trend).
- **Not stable.** Since 2025, the gain almost disappears at 3 and 6 months, and
  FRED is worse than LOCF at 9 months.
- **Not general.** Compared with LOCF, the gain comes mostly from European steel and
  plastics price indices. There is no reliable gain on LME metals.
- **Optimistic.** We also tried new orders (`A31SNO`) on the same dates and kept
  the better of the two indicators.
- **Favourable timing.** The price of the forecast month is assumed known, as in
  the submission. We did not test this model with publication lags.

With ten years of monthly data and overlapping periods, this is not enough to
replace a simple benchmark.
