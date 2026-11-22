"""Point-forecast models and paired rolling-origin evaluation.

This module combines the two point-forecasting stages:

- **Transparent baselines** (:func:`baseline_forecasts`, :func:`rolling_backtest`,
  :func:`score_predictions`, :func:`final_forecasts`). LOCF is the mandatory
  comparator; seasonal naive, drift and historical mean return are alternatives.
- **Supervised and univariate families** (:class:`NestedForecaster`,
  :func:`estimator`, :func:`frozen_selection`) covering ARIMA, ETS, Ridge,
  Elastic Net, XGBoost and LightGBM in price-level and log-return form.

All forecasts target the requested calendar month, so the number of steps
includes both the nominal horizon and any gap caused by publication delay. Model
selection is frozen before confirmation data are inspected; see
:func:`frozen_selection`. Preprocessing is fitted inside training folds only.
Metrics stay in each series' native unit; only MASE and relative improvements
are safe to aggregate across materials.
"""

import warnings
from functools import cache

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from .._04_features import (
    TARGETS,
    decision_date,
    monthly_history,
    snapshot,
    training_rows,
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


# ---------------------------------------------------------------------------
# Supervised and univariate model families (formerly advanced.py)
# ---------------------------------------------------------------------------

SEED = 42
FEATURES = (
    "latest_price",
    "price_lag_1",
    "price_lag_12",
    "mean_3",
    "mean_12",
    "latest_return",
    "return_lag_1",
    "momentum_3",
    "momentum_6",
    "momentum_12",
    "volatility_12",
    "target_month_sin",
    "target_month_cos",
    "E01_latest_return",
    "E05_latest_return",
    "E22_latest_return",
    "S06_latest_return",
    "S03_latest_return",
    "S04_latest_return",
    "S05_latest_return",
    "E05_volatility_12",
    "nickel_steel_return_spread",
    "ore_steel_return_spread",
    "zinc_galvanised_return_spread",
)
FAMILIES = ("ARIMA", "ETS", "Ridge", "ElasticNet", "XGBoost", "LightGBM")
CANDIDATES = ("LOCF",) + tuple(
    f"{family}_{form}" for family in FAMILIES for form in ("level", "log_return")
)
SELECTION_FREEZE = pd.Timestamp("2023-12-31")


def inner_windows(data, origin):
    """Return two publication-purged inner folds for hyperparameter tuning.

    The last four and last available origins before the outer origin are used,
    with the second-to-last as the earlier fold. Training rows for each fold are
    selected with :func:`training_rows`, so a label is never used before it was
    published.

    Args:
        data: Supervised dataset for one series.
        origin: Outer origin timestamp.

    Returns:
        A list of ``(training, validation)`` DataFrame pairs.

    Raises:
        ValueError: If exactly two folds with at least 12 training labels cannot
            be formed.
    """
    eligible = training_rows(data, decision_date(origin)).sort_values("origin")
    windows = []
    for position in (-4, -1):
        if len(eligible) < abs(position):
            continue
        validation = eligible.iloc[[position]]
        training = training_rows(data, decision_date(validation.origin.iloc[0]))
        if len(training) >= 12:
            windows.append((training, validation))
    if len(windows) != 2:
        raise ValueError(
            "Two publication-purged inner folds with >=12 labels required."
        )
    return windows


def target_values(data, form):
    """Encode the supervised target as a level or a cumulative log return.

    Args:
        data: Training/validation rows containing ``target_value`` and
            ``latest_price``.
        form: ``"level"`` predicts the price directly; ``"log_return"`` predicts
            ``log(target / latest_price)``.

    Returns:
        A NumPy array of targets in the requested space.
    """
    return (
        data.target_value.to_numpy()
        if form == "level"
        else np.log(data.target_value.to_numpy() / data.latest_price.to_numpy())
    )


def original_units(prediction, row, form):
    """Map a model prediction back to the series' original price units.

    Args:
        prediction: Value produced by the estimator in its training space.
        row: One-row DataFrame containing ``latest_price`` for the log-return
            conversion.
        form: ``"level"`` or ``"log_return"``, matching the fitted model.

    Returns:
        The predicted price as a float.
    """
    return (
        float(prediction)
        if form == "level"
        else float(row.latest_price.iloc[0] * np.exp(prediction))
    )


def estimator(family, option):
    """Build one unfitted estimator with training-only preprocessing.

    Args:
        family: ``Ridge``, ``ElasticNet``, ``XGBoost`` or ``LightGBM``. ARIMA
            and ETS are fitted separately because they consume a univariate
            series rather than the feature matrix.
        option: Predeclared option index (0 or 1) selecting the regularisation
            strength or tree depth.

    Returns:
        A scikit-learn ``Pipeline`` that median-imputes (and scales for the
        linear models) before the estimator.

    Raises:
        ValueError: If the family is unknown.
    """
    if family == "Ridge":
        model = Ridge(alpha=(10.0, 100.0)[option])
    elif family == "ElasticNet":
        model = ElasticNet(
            alpha=(0.01, 0.1)[option], l1_ratio=0.5, max_iter=10000, random_state=SEED
        )
    elif family == "XGBoost":
        from xgboost import XGBRegressor

        model = XGBRegressor(
            n_estimators=40,
            max_depth=(2, 3)[option],
            learning_rate=0.05,
            reg_lambda=5.0,
            min_child_weight=5,
            n_jobs=1,
            random_state=SEED,
            objective="reg:squarederror",
        )
    elif family == "LightGBM":
        from lightgbm import LGBMRegressor

        model = LGBMRegressor(
            n_estimators=40,
            num_leaves=(4, 7)[option],
            max_depth=3,
            min_child_samples=8,
            learning_rate=0.05,
            reg_lambda=5.0,
            n_jobs=1,
            random_state=SEED,
            verbosity=-1,
            deterministic=True,
            force_col_wise=True,
        )
    else:
        raise ValueError(f"Unknown supervised model {family}")
    steps = [SimpleImputer(strategy="median", keep_empty_features=True)]
    if family in ("Ridge", "ElasticNet"):
        steps.append(StandardScaler())
    return make_pipeline(*steps, model)


class NestedForecaster:
    """Fit and tune one model family without leaking future labels.

    The class caches each series' snapshot/history and the univariate forecast
    path, then reuses them across horizons and options. For every outer origin it:

    1. selects training rows with :func:`training_rows`;
    2. tunes two predeclared options on two publication-purged inner folds;
    3. refits the winner on the outer training rows and predicts the target.

    Every inner fold is recorded in ``audit``; failed fits go to ``failures``
    with their exception text instead of silently disappearing.

    Args:
        observations: Long observation table.
        catalog: Source catalog.
    """

    def __init__(self, observations, catalog):
        self.observations, self.catalog = observations, catalog
        self.audit = []
        self.failures = []
        self.history = cache(self.history)
        self.univariate_path = cache(self.univariate_path)

    def history(self, identifier, origin):
        return monthly_history(
            snapshot(self.observations, self.catalog, origin), identifier
        )

    def univariate_path(self, identifier, origin, family, form, option):
        history = self.history(identifier, origin)
        if len(history) < 36 or history.isna().any():
            raise ValueError("Univariate fit requires >=36 complete calendar months.")
        values = history.to_numpy()
        if form == "log_return":
            values = np.diff(np.log(values))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            if family == "ARIMA":
                order = (
                    ((1, 1, 0), (0, 1, 1))[option]
                    if form == "level"
                    else ((1, 0, 0), (0, 0, 1))[option]
                )
                fitted = ARIMA(values, order=order).fit()
                if not fitted.mle_retvals.get("converged", True):
                    raise ValueError("ARIMA did not converge.")
            else:
                fitted = ExponentialSmoothing(
                    values,
                    trend=None if option == 0 else "add",
                    damped_trend=option == 1,
                    initialization_method="estimated",
                ).fit()
                if not fitted.mle_retvals.get("success", True):
                    raise ValueError("ETS did not converge.")
            prediction = np.asarray(fitted.forecast(24))
        if form == "log_return":
            prediction = history.iloc[-1] * np.exp(np.cumsum(prediction))
        if not np.isfinite(prediction).all():
            raise ValueError("Nonfinite univariate forecast.")
        return prediction, len(caught)

    def predict(
        self,
        identifier,
        origin,
        data,
        row,
        family,
        form,
        option,
        *,
        importance=None,
        minimum_labels=18,
    ):
        training = training_rows(data, decision_date(origin))
        if len(training) < minimum_labels:
            raise ValueError("At least 18 outer training labels required.")
        if family in ("ARIMA", "ETS"):
            history = self.history(identifier, pd.Timestamp(origin))
            steps = (
                pd.Period(row.target_month.iloc[0], freq="M").ordinal
                - history.index[-1].to_period("M").ordinal
            )
            path, warning_count = self.univariate_path(
                identifier, pd.Timestamp(origin), family, form, option
            )
            return float(path[steps - 1]), warning_count
        model = estimator(family, option)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model.fit(
                training[list(FEATURES)].to_numpy(), target_values(training, form)
            )
            value = model.predict(row[list(FEATURES)].to_numpy())[0]
        if importance is not None and hasattr(
            model.steps[-1][1], "feature_importances_"
        ):
            importance.extend(
                {
                    "series_id": identifier,
                    "model": f"{family}_{form}",
                    "horizon": int(row.horizon.iloc[0]),
                    "feature": name,
                    "importance": float(value),
                }
                for name, value in zip(
                    FEATURES, model.steps[-1][1].feature_importances_, strict=True
                )
            )
        return original_units(value, row, form), len(caught)

    def tune_predict(self, identifier, data, row, candidate, *, importance=None):
        origin = row.origin.iloc[0]
        history = self.history(identifier, pd.Timestamp(origin))
        if candidate == "LOCF":
            predictions, scale, _ = baseline_forecasts(
                history, row.target_month.iloc[0]
            )
            return predictions["LOCF"], {"option": None, "mase_scale": scale}
        family, form = candidate.split("_", 1)
        windows = inner_windows(data, origin)
        options = []
        for option in (0, 1):
            errors = []
            for training, validation in windows:
                inner_origin = validation.origin.iloc[0]
                try:
                    # predict's own label gate also applies; minimum outer threshold
                    # is relaxed for the smaller designated inner folds.
                    if family in ("ARIMA", "ETS"):
                        prediction, count = self.predict(
                            identifier,
                            inner_origin,
                            data,
                            validation,
                            family,
                            form,
                            option,
                            minimum_labels=12,
                        )
                    else:
                        model = estimator(family, option)
                        with warnings.catch_warnings(record=True) as caught:
                            warnings.simplefilter("always")
                            model.fit(
                                training[list(FEATURES)].to_numpy(),
                                target_values(training, form),
                            )
                            estimate = model.predict(
                                validation[list(FEATURES)].to_numpy()
                            )[0]
                        prediction, count = (
                            original_units(estimate, validation, form),
                            len(caught),
                        )
                    error = abs(prediction - validation.target_value.iloc[0])
                    if not np.isfinite(error):
                        raise ValueError("Nonfinite inner error.")
                    errors.append(error)
                    status = "ok"
                except (ValueError, np.linalg.LinAlgError, OverflowError) as error:
                    errors.append(np.inf)
                    count, status = 0, str(error)
                self.audit.append(
                    {
                        "series_id": identifier,
                        "horizon": int(row.horizon.iloc[0]),
                        "outer_origin": origin,
                        "candidate": candidate,
                        "option": option,
                        "inner_origin": inner_origin,
                        "training_rows": len(training),
                        "training_first_origin": training.origin.min(),
                        "training_last_origin": training.origin.max(),
                        "training_last_label_release": training.target_available_at_estimated.max(),
                        "validation_label_release": validation.target_available_at_estimated.iloc[
                            0
                        ],
                        "absolute_error": errors[-1],
                        "warnings": count,
                        "status": status,
                    }
                )
            options.append(np.mean(errors))
        if not np.isfinite(options).any():
            raise ValueError("All tuning options failed.")
        chosen = int(np.argmin(options))
        prediction, warnings_count = self.predict(
            identifier, origin, data, row, family, form, chosen, importance=importance
        )
        if not np.isfinite(prediction):
            raise ValueError("Nonfinite outer prediction.")
        train = training_rows(data, decision_date(origin))
        return prediction, {
            "option": chosen,
            "inner_MAE": float(options[chosen]),
            "training_rows": len(train),
            "training_first_origin": str(train.origin.min()),
            "training_last_origin": str(train.origin.max()),
            "training_last_label_release": str(
                train.target_available_at_estimated.max()
            ),
            "warnings": warnings_count,
        }


def frozen_selection(predictions):
    """Choose a final model family using selection folds only.

    LOCF remains the default. Another family replaces it only if, on the
    pre-freeze selection partitions, it has at least a 5% MAE gain, wins at least
    60% of origins, covers at least six origins, and improves in both 2021 and
    2022. Confirmation rows are ignored by design, even if they later favour a
    different family.

    Args:
        predictions: Outer predictions with ``partition``, ``series_id``,
            ``horizon``, ``model``, ``origin`` and ``error``.

    Returns:
        One row per series/horizon with the selected model, selection gain, win
        fraction, freeze date and a human-readable reason.
    """
    rows = []
    for (identifier, horizon), block in predictions.groupby(["series_id", "horizon"]):
        selection = block.loc[block.partition == "selection"]
        if selection.empty:
            rows.append(
                {
                    "series_id": identifier,
                    "horizon": horizon,
                    "selected_model": "LOCF",
                    "reason": "No selection coverage",
                }
            )
            continue
        baseline = (
            selection.loc[selection.model == "LOCF"].set_index("origin").error.abs()
        )
        qualified = []
        for model, group in selection.groupby("model"):
            errors = group.set_index("origin").error.abs().reindex(baseline.index)
            gain = 1 - errors.mean() / baseline.mean()
            wins = (errors < baseline).mean()
            year_gains = [
                (
                    baseline.loc[baseline.index.year == year]
                    - errors.loc[errors.index.year == year]
                ).mean()
                for year in (2021, 2022)
            ]
            if (
                model != "LOCF"
                and len(errors) >= 6
                and gain >= 0.05
                and wins >= 0.6
                and all(value > 0 for value in year_gains)
            ):
                qualified.append((errors.mean(), model, gain, wins))
        best = min(qualified) if qualified else (baseline.mean(), "LOCF", 0.0, 0.0)
        rows.append(
            {
                "series_id": identifier,
                "horizon": horizon,
                "selected_model": best[1],
                "selection_improvement_pct": best[2] * 100,
                "selection_win_fraction": best[3],
                "selection_freeze": SELECTION_FREEZE,
                "reason": ">=5% MAE gain, >=60% wins, positive 2021 and 2022 gains; otherwise LOCF",
            }
        )
    return pd.DataFrame(rows)
