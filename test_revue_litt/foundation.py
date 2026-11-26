"""Same backtest as the FRED model, with TabICL instead of the ridge."""

import numpy as np
import pandas as pd
from backtest import MINIMUM_ROWS, MODELS, build_table

from schneider_electric_forecasting._04_features import TARGETS, monthly_history
from schneider_electric_forecasting.config import FORECAST_HORIZONS

FEATURES = MODELS["FRED"]


def forecast(table, origin, horizon, model):
    """Return the LOCF and TabICL forecasts, using only data known at the origin."""
    if origin not in table.index:
        return {}
    known = table.loc[:origin]
    target = known.log_price.shift(-horizon) - known.log_price
    train = known.assign(target=target).dropna()
    latest = known.tail(1)
    if len(train) < MINIMUM_ROWS or latest.isna().any(axis=None):
        return {}
    last_price = np.exp(latest.log_price.item())

    model.fit(train[FEATURES].to_numpy(), train.target.to_numpy())
    output = model.predict(
        latest[FEATURES].to_numpy(),
        output_type=["median", "quantiles"],
        alphas=[0.1, 0.9],
    )
    p10, p90 = np.ravel(output["quantiles"])
    p50 = np.ravel(output["median"])[0]
    return {
        "LOCF": (last_price, np.nan, np.nan),
        "TabICL": tuple(last_price * np.exp([p50, p10, p90])),
    }


def backtest(observations, capacity, origins, model, targets=TARGETS):
    """Repeat the forecasts for every material, past month and horizon."""
    rows = []
    for series_id in targets:
        prices = monthly_history(observations, series_id)
        table = build_table(prices, capacity)
        for origin in origins:
            for horizon in FORECAST_HORIZONS:
                actual = prices.get(origin + pd.DateOffset(months=horizon), np.nan)
                results = forecast(table, origin, horizon, model)
                for name, (p50, p10, p90) in results.items():
                    rows.append(
                        {
                            "series_id": series_id,
                            "origin": origin,
                            "horizon": horizon,
                            "model": name,
                            "prediction": p50,
                            "p10": p10,
                            "p90": p90,
                            "actual": actual,
                        }
                    )
    return pd.DataFrame(rows)


def relative_error(predictions):
    """Return the TabICL mean absolute error divided by the LOCF error."""
    realised = predictions.dropna(subset=["actual"])
    errors = (realised.prediction - realised.actual).abs()
    keys = [realised.series_id, realised.horizon, realised.model]
    mae = errors.groupby(keys).mean().unstack("model")
    return mae.TabICL / mae.LOCF


def coverage(predictions):
    """Return the share of real prices below, inside and above the P10-P90 range."""
    bands = predictions.dropna(subset=["actual", "p10"])
    shares = pd.DataFrame(
        {
            "below_p10": bands.actual < bands.p10,
            "inside": bands.actual.between(bands.p10, bands.p90),
            "above_p90": bands.actual > bands.p90,
        }
    )
    return shares.groupby(bands.horizon).mean()
