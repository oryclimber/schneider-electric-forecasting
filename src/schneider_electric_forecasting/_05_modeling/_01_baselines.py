"""Transparent point baselines and paired rolling-origin evaluation.

All forecasts target the requested calendar month, including any gap caused by
publication delay. LOCF is the mandatory comparator; the other baselines are
seasonal naive, drift and historical mean return. Metrics stay in each series'
native unit; only MASE and relative improvements are safe to aggregate.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .._04_features import TARGETS, decision_date, monthly_history, snapshot

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
    # Steps span the publication gap as well as the requested horizon: the
    # model forecasts from the last *observable* month, not from the decision
    # month, because the decision month may not be published yet.
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
    """Score all baselines on identical rolling-origin windows.

    Actuals are taken from the final snapshot and only labels estimated to be
    available at the final decision are scored. A window that cannot produce a
    forecast or a defined MASE scale is excluded for *all* models, keeping the
    comparison paired.

    Args:
        observations: Long observation table.
        catalog: Source catalog.
        origins: Period-like rolling origins to evaluate.
        final_origin: Last allowed origin; later origins raise.
        targets: Series to forecast.
        rules: Optional publication-lag assumptions.
        minimum_observations: Minimum non-null training observations.

    Returns:
        ``(predictions, scores, exclusions)``: one row per model/window, the
        aggregated metrics, and a ledger explaining every excluded window.

    Raises:
        ValueError: If an origin exceeds ``final_origin`` or no paired window
            remains.
    """
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
    """Aggregate point-forecast errors by series, horizon and model.

    ``error`` is prediction minus actual in the series' native unit. MASE
    divides mean absolute error by ``mase_scale``, the training-only one-step
    naive MAE for that origin, so MASE is comparable across series while MAE and
    RMSE are not.

    Args:
        predictions: Long table with one row per prediction. Required columns
            are ``series_id``, ``horizon``, ``model``, ``error``,
            ``mase_scale``, ``origin`` and ``target_month``.

    Returns:
        One row per ``(series_id, horizon, model)`` with ``valid_windows``,
        ``MAE``, ``RMSE``, ``MASE``, ``bias``, ``first_origin`` and
        ``last_origin``.

    Raises:
        KeyError: If a required column is absent.
        ValueError: If no predictions are supplied.
    """
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
    """Produce the four baseline forecasts for the final September origin.

    Unlike :func:`rolling_backtest`, this function returns unscored forecasts
    for the actual future target months (December 2026, March 2027, June 2027).
    The effective step count still includes each series' publication gap.

    Args:
        observations: Long observation table.
        catalog: Source catalog.
        origin: Final origin month.
        targets: Series to forecast.
        rules: Optional publication-lag assumptions.

    Returns:
        One row per series/model/horizon with origin, decision date, target
        month, last observed month, effective horizon, model and prediction.

    Raises:
        ValueError: If a series has insufficient history for the baselines.
    """
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
