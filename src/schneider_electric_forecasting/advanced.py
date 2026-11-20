"""Small-sample models with publication-purged nested rolling validation."""

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

from schneider_electric_forecasting.baselines import baseline_forecasts
from schneider_electric_forecasting.features import (
    decision_date,
    monthly_history,
    snapshot,
    training_rows,
)

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
    """Validation labels known at outer origin; train labels known at inner origin."""
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
    return (
        data.target_value.to_numpy()
        if form == "level"
        else np.log(data.target_value.to_numpy() / data.latest_price.to_numpy())
    )


def original_units(prediction, row, form):
    return (
        float(prediction)
        if form == "level"
        else float(row.latest_price.iloc[0] * np.exp(prediction))
    )


def estimator(family, option):
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
    """Reuse existing snapshots and histories across models and horizons."""

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
    """Use pre-freeze selection folds only; require gain and cross-year consistency."""
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
