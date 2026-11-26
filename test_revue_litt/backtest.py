"""Compare LOCF, the price trend, and the trend with the FRED indicator."""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from schneider_electric_forecasting._04_features import TARGETS, monthly_history
from schneider_electric_forecasting.config import FORECAST_HORIZONS

TREND = ["return_1", "return_3"]
MODELS = {"Trend": TREND, "FRED": [*TREND, "capacity", "capacity_change"]}
MINIMUM_ROWS = 36


def build_table(prices, capacity):
    """Join the recent changes of one material with the FRED indicator."""
    log_price = np.log(prices)
    table = pd.DataFrame(
        {
            "log_price": log_price,
            "return_1": log_price.diff(1),
            "return_3": log_price.diff(3),
        }
    )
    return table.join(capacity)


def forecast(table, origin, horizon):
    """Forecast one price with each model, using only data known at the origin."""
    if origin not in table.index:
        return {}
    known = table.loc[:origin]
    target = known.log_price.shift(-horizon) - known.log_price
    train = known.assign(target=target).dropna()
    latest = known.tail(1)
    if len(train) < MINIMUM_ROWS or latest.isna().any(axis=None):
        return {}
    last_price = np.exp(latest.log_price.item())
    forecasts = {"LOCF": last_price}
    for name, features in MODELS.items():
        # Without an intercept, a model that finds no signal goes back to LOCF.
        model = make_pipeline(StandardScaler(), Ridge(alpha=10, fit_intercept=False))
        model.fit(train[features], train.target)
        change = model.predict(latest[features])[0]
        forecasts[name] = last_price * np.exp(change)
    return forecasts


def backtest(observations, capacity, origins, targets=TARGETS):
    """Repeat the forecasts for every material, past month and horizon."""
    rows = []
    for series_id in targets:
        prices = monthly_history(observations, series_id)
        table = build_table(prices, capacity)
        for origin in origins:
            for horizon in FORECAST_HORIZONS:
                actual = prices.get(origin + pd.DateOffset(months=horizon), np.nan)
                for model, prediction in forecast(table, origin, horizon).items():
                    rows.append(
                        {
                            "series_id": series_id,
                            "origin": origin,
                            "horizon": horizon,
                            "model": model,
                            "prediction": prediction,
                            "actual": actual,
                        }
                    )
    return pd.DataFrame(rows)


def relative_error(predictions):
    """Return each model's mean absolute error divided by the LOCF error."""
    realised = predictions.dropna(subset=["actual"])
    errors = (realised.prediction - realised.actual).abs()
    keys = [realised.series_id, realised.horizon, realised.model]
    mae = errors.groupby(keys).mean().unstack("model")
    return mae[list(MODELS)].div(mae["LOCF"], axis=0)
