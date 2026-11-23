"""Build origin-specific features and supervised labels.

This is stage 4 of the pipeline. For every rolling origin it first cuts the
observations at that origin's decision date, then computes features from the
resulting snapshot only. Features include price/return lags, rolling level and
volatility statistics, momentum, calendar terms, predeclared energy/FX drivers
and economically defined cross-material return spreads.

Two rules keep the table honest:

- **Lags are relative to the last observable month**, not the decision month.
  A series delayed by publication lag therefore carries an observation-age
  field so the model can see how stale its input is.
- **Labels are metadata, not features.** :func:`supervised_dataset` stores the
  realised target in ``target_value`` and its estimated publication date in
  ``target_available_at_estimated``. Call :func:`training_rows` before fitting
  so no label is used before it would have been published.

There is deliberately no imputation, scaling or feature selection here; those
must be fitted inside each training fold by the modelling stage.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from .._02_validation import (
    DOCUMENTED_PUBLICATION_LAG_ASSUMPTIONS,
    AvailabilityMode,
    PublicationLagAssumption,
    select_available_observations,
)

LAGS = (1, 2, 3, 6, 12)
WINDOWS = (3, 6, 12)
TARGETS = tuple(
    [f"S{i:02d}" for i in (1, 2, 3, 4, 5, 7, 8, 9, 10, 11)] + ["D01", "D02"]
)
# Explicit project scenarios, not newly verified source publication evidence.
PHASE3_LAGS = dict(DOCUMENTED_PUBLICATION_LAG_ASSUMPTIONS)
for identifier, provider, lag, evidence in (
    (
        "S06",
        "European Central Bank",
        1,
        "Provisional project scenario: monthly FX average available one full month after month end; no verified vintage evidence.",
    ),
    (
        "D01",
        "Derived (S01 / S06)",
        2,
        "Dependency bound: max(IMF price lag 2, provisional FX lag 1); no independent publication.",
    ),
    (
        "D02",
        "Derived (S02 / S06)",
        2,
        "Dependency bound: max(IMF price lag 2, provisional FX lag 1); no independent publication.",
    ),
):
    PHASE3_LAGS[identifier] = PublicationLagAssumption(
        identifier, provider, lag, evidence
    )


def decision_date(origin):
    """Month-end decision, matching the Phase 2 September 30 convention."""
    return pd.Period(origin, freq="M").end_time.date()


def snapshot(
    observations, catalog, origin, *, mode=AvailabilityMode.ASSUMPTION_BASED, rules=None
):
    """Cut every series to what was available at one rolling origin.

    This is the only entry point feature code should use to obtain history: it
    applies the publication-lag rules before any lag, return or rolling
    calculation, so later steps cannot see future values.

    Args:
        observations: Long observation table.
        catalog: Matching source catalog.
        origin: Origin month (period-like). The decision date is that month's
            end.
        mode: Availability mode; the modelling pipeline uses the documented
            assumption-based mode by default.
        rules: Optional ``series_id -> PublicationLagAssumption`` mapping.
            Defaults to :data:`PHASE3_LAGS`.

    Returns:
        Rows available at the origin, with availability metadata attached.
    """
    rules = PHASE3_LAGS if rules is None else rules
    identifiers = set(rules)
    return select_available_observations(
        observations.loc[observations.series_id.isin(identifiers)],
        catalog.loc[catalog.series_id.isin(identifiers)],
        as_of=decision_date(origin),
        forecast_origin=pd.Period(origin, freq="M").start_time.date(),
        mode=mode,
        lag_assumptions=rules,
    )


def monthly_history(selected, series_id):
    """Return one series as a complete monthly index ending at its last value.

    Args:
        selected: Snapshot returned by :func:`snapshot`.
        series_id: Series to extract.

    Returns:
        A month-start indexed Series. Interior calendar gaps remain as NaN;
        trailing missing values after the last observation are dropped. An empty
        Series is returned when the series has no eligible rows.

    Raises:
        ValueError: If dates are not unique month starts, values are non-finite,
            or a non-positive price/index is present.
    """
    data = selected.loc[selected.series_id == series_id, ["month", "value"]].copy()
    if data.empty:
        return pd.Series(dtype=float, index=pd.DatetimeIndex([], name="month"))
    data["month"] = pd.to_datetime(data.month)
    if (data.month.dt.day != 1).any() or data.month.duplicated().any():
        raise ValueError("History requires unique month-start observations.")
    values = pd.to_numeric(data.set_index("month").value, errors="raise").sort_index()
    if not np.isfinite(values.dropna()).all() or (values.dropna() <= 0).any():
        raise ValueError("Prices and indices must be finite and positive.")
    values = (
        values.dropna()
        if values.notna().sum() == 0
        else values.loc[: values.last_valid_index()]
    )
    return values.asfreq("MS") if not values.empty else values


def history_features(values):
    """Lags are relative to the last observed month, not the decision month.

    Latest value is separate. A rolling window includes that latest observation;
    incomplete calendar windows remain missing. Returns are fractional changes.
    """
    if values.empty:
        return {}
    returns = values.pct_change(fill_method=None)
    result = {"latest_price": values.iloc[-1]}
    for lag in LAGS:
        result[f"price_lag_{lag}"] = values.shift(lag).iloc[-1]
        result[f"return_lag_{lag}"] = returns.shift(lag).iloc[-1]
    result["latest_return"] = returns.iloc[-1]
    for window in WINDOWS:
        result[f"mean_{window}"] = (
            values.rolling(window, min_periods=window).mean().iloc[-1]
        )
        result[f"std_{window}"] = (
            values.rolling(window, min_periods=window).std().iloc[-1]
        )
        result[f"volatility_{window}"] = returns.rolling(
            window, min_periods=window
        ).std().iloc[-1] * np.sqrt(12)
        result[f"momentum_{window}"] = values.pct_change(window, fill_method=None).iloc[
            -1
        ]
    return result


def origin_features(selected, series_id, origin, horizon):
    """Build one feature row for a series at a rolling origin.

    The row contains the series' own history features, calendar terms, the
    predeclared driver features and the three cross-material return spreads.
    All inputs come from ``selected``, which must already be cut at the origin;
    the function checks for future observations and raises if any remain.

    Args:
        selected: Snapshot returned by :func:`snapshot` for this origin.
        series_id: Series to build the row for.
        origin: Month-end decision origin (period-like).
        horizon: One of 3, 6 or 9 months.

    Returns:
        A flat ``dict`` of feature name to value. The row also records
        ``last_observed_month`` and ``observation_age_months`` so downstream
        code can see how stale each input is.

    Raises:
        ValueError: If the horizon is unsupported or the snapshot contains
            observations after the origin month.
    """
    if horizon not in (3, 6, 9):
        raise ValueError("Forecast horizon must be 3, 6 or 9 months.")
    period = pd.Period(origin, freq="M")
    if not selected.empty and pd.to_datetime(selected.month).max() > period.end_time:
        raise ValueError("Selected snapshot contains future observations.")
    target = period + horizon
    history = monthly_history(selected, series_id)
    result = {
        "series_id": series_id,
        "origin": period.start_time,
        "horizon": horizon,
        "target_month": target.start_time,
        "availability_mode": selected.availability_mode.iloc[0]
        if not selected.empty and "availability_mode" in selected
        else "unknown",
    }
    if history.empty:
        return result
    result.update(history_features(history))
    result["last_observed_month"] = history.index[-1]
    result["observation_age_months"] = (
        period.ordinal - history.index[-1].to_period("M").ordinal
    )
    for prefix, month in (("origin", period.month), ("target", target.month)):
        result[f"{prefix}_calendar_month"] = month
        result[f"{prefix}_month_sin"] = np.sin(2 * np.pi * month / 12)
        result[f"{prefix}_month_cos"] = np.cos(2 * np.pi * month / 12)
    # All candidates are predeclared economically; no full-sample correlation selection.
    drivers = ("E01", "E05", "E22", "E13", "E20", "E21", "S06", "S03", "S04", "S05")
    for identifier in drivers:
        driver = monthly_history(selected, identifier)
        if driver.empty:
            continue
        for name, value in history_features(driver).items():
            result[f"{identifier}_{name}"] = value
        result[f"{identifier}_cutoff"] = driver.index[-1]
        result[f"{identifier}_age_months"] = (
            period.ordinal - driver.index[-1].to_period("M").ordinal
        )
    # Return spreads avoid dividing unlike units (ore, nickel and steel indices).
    for first, second, name in (
        ("S03", "S09", "nickel_steel"),
        ("S04", "S09", "ore_steel"),
        ("S05", "S10", "zinc_galvanised"),
    ):
        pair = pd.concat(
            [monthly_history(selected, first), monthly_history(selected, second)],
            axis=1,
        ).dropna()
        if len(pair) < 2:
            continue
        pair = pair.asfreq("MS").pct_change(fill_method=None)
        result[f"{name}_return_spread"] = pair.iloc[-1, 0] - pair.iloc[-1, 1]
        result[f"{name}_cutoff"] = pair.index[-1]
    return result


def supervised_dataset(observations, catalog, origins, *, targets=TARGETS, rules=None):
    """Build the supervised origin × series × horizon table.

    Each row contains features computed from that origin's snapshot plus the
    realised target and its estimated availability date. The target columns are
    metadata: they must never be passed to an estimator. Use
    :func:`training_rows` to select only labels published before a training
    cutoff.

    Args:
        observations: Long observation table.
        catalog: Source catalog.
        origins: Iterable of rolling origin months.
        targets: Series to build rows for.
        rules: Optional publication-lag assumptions.

    Returns:
        A DataFrame with one row per origin/series/horizon, including
        ``target_value`` and ``target_available_at_estimated``.

    Raises:
        ValueError: If a target has no availability rule, labels are duplicated,
            or a snapshot contains future observations.
    """
    rules = PHASE3_LAGS if rules is None else rules
    labels = observations.set_index(["series_id", "month"]).value
    if labels.index.duplicated().any():
        raise ValueError("Duplicate target observations.")
    rows = []
    for origin in origins:
        selected = snapshot(observations, catalog, origin, rules=rules)
        for identifier in targets:
            if identifier not in rules:
                raise ValueError(f"No availability rule for {identifier}.")
            base = origin_features(selected, identifier, origin, 3)
            for horizon in (3, 6, 9):
                row = dict(base)
                target = pd.Period(origin, freq="M") + horizon
                row.update(
                    horizon=horizon,
                    target_month=target.start_time,
                    target_calendar_month=target.month,
                    target_month_sin=np.sin(2 * np.pi * target.month / 12),
                    target_month_cos=np.cos(2 * np.pi * target.month / 12),
                )
                row["target_value"] = labels.get(
                    (identifier, row["target_month"]), np.nan
                )
                row["target_available_at_estimated"] = (
                    pd.Period(row["target_month"], freq="M")
                    + rules[identifier].maximum_lag_months
                    + 1
                ).start_time
                rows.append(row)
    return pd.DataFrame(rows)


def training_rows(dataset, as_of: date):
    """Return rows whose target label was published by a training cutoff.

    A row is eligible only when its origin is not later than ``as_of``, its
    target is non-null, and its estimated label-availability date is strictly
    before the cutoff. This is the guard that prevents future labels from
    entering a fit.

    Args:
        dataset: Output of :func:`supervised_dataset`.
        as_of: Training decision date.

    Returns:
        A filtered copy. ``target_value`` and the availability metadata are
        retained for scoring/auditing but must be excluded from estimator
        feature inputs.
    """
    # The +1 day makes a date-only cutoff inclusive of that whole day.
    cutoff = pd.Timestamp(as_of) + pd.Timedelta(days=1)
    return dataset.loc[
        (dataset.target_available_at_estimated < cutoff)
        & dataset.target_value.notna()
        & (dataset.origin <= pd.Timestamp(as_of))
    ].copy()
