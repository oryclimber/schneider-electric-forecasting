"""Transparent baseline forecasts and paired rolling-origin evaluation."""

import numpy as np
import pandas as pd

from schneider_electric_forecasting.features import (
    TARGETS,
    decision_date,
    monthly_history,
    snapshot,
)

MODELS = ("LOCF", "Seasonal Naive", "Drift", "Historical mean return")


def baseline_forecasts(history, target_month, *, minimum_observations=36):
    """Forecast the actual target, including the publication-delay gap.

    Drift is slope per calendar month. Mean return is the expanding arithmetic
    mean of consecutive monthly simple returns, compounded over the full gap.
    Seasonal naive repeats the most recent observable value in the target's
    calendar month (possibly two years earlier for a long effective horizon).
    MASE uses the training-only nonseasonal one-step naive MAE (m=1).
    """
    if not isinstance(history.index, pd.DatetimeIndex) or history.index.has_duplicates:
        raise ValueError("History requires a unique DatetimeIndex.")
    history = history.sort_index().asfreq("MS")
    observed = history.dropna()
    if len(observed) < minimum_observations:
        raise ValueError("Insufficient training observations.")
    if not np.isfinite(observed).all() or (observed <= 0).any():
        raise ValueError("History must contain positive finite values.")
    target = pd.Period(target_month, freq="M")
    last = observed.index[-1].to_period("M")
    steps = target.ordinal - last.ordinal
    if steps <= 0:
        raise ValueError("Target must follow the last observed month.")
    first = observed.index[0].to_period("M")
    seasonal_month = target - 12
    while seasonal_month > last:
        seasonal_month -= 12
    seasonal = history.get(seasonal_month.start_time, np.nan)
    mean_return = history.pct_change(fill_method=None).mean()
    span = last.ordinal - first.ordinal
    forecasts = {
        "LOCF": observed.iloc[-1],
        "Seasonal Naive": seasonal,
        "Drift": observed.iloc[-1]
        + steps * (observed.iloc[-1] - observed.iloc[0]) / span,
        "Historical mean return": observed.iloc[-1] * (1 + mean_return) ** steps,
    }
    scale = history.diff().abs().mean()
    return forecasts, float(scale), steps


def rolling_backtest(
    observations,
    catalog,
    origins,
    *,
    final_origin="2026-09",
    targets=TARGETS,
    rules=None,
    minimum_observations=36,
):
    """Score only labels available at final decision, on identical model windows."""
    final = snapshot(observations, catalog, final_origin, rules=rules)
    actuals = final.set_index(["series_id", "month"]).value
    forecasts, exclusions = [], []
    for origin in origins:
        period = pd.Period(origin, freq="M")
        if period > pd.Period(final_origin, freq="M"):
            raise ValueError("Backtest origin exceeds final experiment origin.")
        selected = snapshot(observations, catalog, origin, rules=rules)
        for identifier in targets:
            history = monthly_history(selected, identifier)
            for horizon in (3, 6, 9):
                target = (period + horizon).start_time
                key = {
                    "series_id": identifier,
                    "origin": period.start_time,
                    "horizon": horizon,
                    "target_month": target,
                }
                actual = actuals.get((identifier, target), np.nan)
                if pd.isna(actual):
                    exclusions.append(
                        {
                            **key,
                            "reason": "label_missing_or_unavailable_at_final_decision",
                        }
                    )
                    continue
                try:
                    predictions, scale, steps = baseline_forecasts(
                        history, target, minimum_observations=minimum_observations
                    )
                except ValueError as error:
                    exclusions.append({**key, "reason": str(error)})
                    continue
                if (
                    not np.isfinite(scale)
                    or scale <= 0
                    or not all(np.isfinite(value) for value in predictions.values())
                ):
                    exclusions.append(
                        {
                            **key,
                            "reason": "undefined_MASE_scale_or_missing_baseline; excluded_for_all_models",
                        }
                    )
                    continue
                for model, prediction in predictions.items():
                    forecasts.append(
                        {
                            **key,
                            "model": model,
                            "prediction": prediction,
                            "actual": actual,
                            "error": prediction - actual,
                            "mase_scale": scale,
                            "last_observed_month": history.last_valid_index(),
                            "effective_horizon": steps,
                            "availability_mode": "assumption_based",
                        }
                    )
    predictions = pd.DataFrame(forecasts)
    if predictions.empty:
        raise ValueError("No paired evaluation windows.")
    return predictions, score_predictions(predictions), pd.DataFrame(exclusions)


def score_predictions(predictions):
    rows = []
    for (identifier, horizon, model), group in predictions.groupby(
        ["series_id", "horizon", "model"]
    ):
        rows.append(
            {
                "series_id": identifier,
                "horizon": horizon,
                "model": model,
                "valid_windows": len(group),
                "MAE": group.error.abs().mean(),
                "RMSE": np.sqrt((group.error**2).mean()),
                "MASE": (group.error.abs() / group.mase_scale).mean(),
                "bias": group.error.mean(),
                "first_origin": group.origin.min(),
                "last_origin": group.origin.max(),
            }
        )
    return pd.DataFrame(rows)


def final_forecasts(
    observations, catalog, *, origin="2026-09", targets=TARGETS, rules=None
):
    selected = snapshot(observations, catalog, origin, rules=rules)
    rows = []
    for identifier in targets:
        history = monthly_history(selected, identifier)
        for horizon in (3, 6, 9):
            target = (pd.Period(origin, freq="M") + horizon).start_time
            predictions, _, steps = baseline_forecasts(history, target)
            for model, value in predictions.items():
                rows.append(
                    {
                        "series_id": identifier,
                        "origin": pd.Period(origin, freq="M").start_time,
                        "decision_date": decision_date(origin),
                        "horizon": horizon,
                        "target_month": target,
                        "last_observed_month": history.last_valid_index(),
                        "effective_horizon": steps,
                        "model": model,
                        "prediction": value,
                        "availability_mode": "assumption_based",
                    }
                )
    return pd.DataFrame(rows)
