"""LOCF-based predictive distributions and dependence-preserving scenarios."""

import hashlib
import math
import warnings

import numpy as np
import pandas as pd
from arch import arch_model
from statsmodels.stats.diagnostic import het_arch
from statsmodels.stats.multitest import multipletests

from schneider_electric_forecasting.features import (
    PHASE3_LAGS,
    TARGETS,
    monthly_history,
    snapshot,
)

SEED = 42
SIMULATIONS = 3000
CORE_METHODS = (
    "LOCF_error_quantiles",
    "Historical_simulation",
    "Gaussian_RW",
    "Innovation_bootstrap",
    "Moving_block_bootstrap",
)
METHODS = CORE_METHODS + ("GARCH", "TS_conformal")


def generator(*keys):
    digest = int.from_bytes(
        hashlib.sha256("|".join(map(str, keys)).encode()).digest()[:8], "little"
    )
    return np.random.default_rng(np.random.SeedSequence([SEED, digest]))


def block_draws(innovations, simulations, steps, rng, block_length=3):
    """Sample contiguous blocks of rows; multivariate rows always travel together."""
    innovations = np.asarray(innovations)
    if steps < 1 or block_length < 1 or len(innovations) < block_length:
        raise ValueError("Insufficient innovations or invalid forecast/block length.")
    starts = rng.integers(
        0,
        len(innovations) - block_length + 1,
        size=(simulations, math.ceil(steps / block_length)),
    )
    indices = (starts[:, :, None] + np.arange(block_length)).reshape(simulations, -1)[
        :, :steps
    ]
    return innovations[indices]


def summarize(samples):
    samples = np.asarray(samples)
    if not np.isfinite(samples).all() or (samples <= 0).any():
        raise ValueError("Predictive prices must be finite and positive.")
    lower, median, upper = np.quantile(samples, [0.1, 0.5, 0.9])
    return {"P10": lower, "P50": median, "P90": upper, "expected_price": samples.mean()}


def conformal_quantiles(errors, latest, minimum=24):
    """Two one-sided finite-sample order statistics, NOT a time-series guarantee.

    Signed log errors permit asymmetric limits and a median distinct from LOCF.
    Temporal exchangeability is not assumed verified. The dedicated calibration
    block is fixed independently of confirmation. No full distribution/mean is
    implied by this interval-only method.
    """
    errors = np.sort(np.asarray(errors))
    n = len(errors)
    if n < minimum:
        raise ValueError(f"Insufficient conformal calibration: {n} < {minimum}")
    low, high = math.floor((n + 1) * 0.1), math.ceil((n + 1) * 0.9)
    if low < 1 or high > n:
        raise ValueError("Finite conformal endpoints unavailable.")
    return {
        "P10": latest * np.exp(errors[low - 1]),
        "P50": latest * np.exp(np.median(errors)),
        "P90": latest * np.exp(errors[high - 1]),
        "expected_price": np.nan,
    }


def residual_ledger(observations, catalog, origins):
    """Labels are stored separately and cannot enter a pool before publication."""
    prices = observations.set_index(["series_id", "month"]).value
    if prices.index.duplicated().any():
        raise ValueError("Duplicate series/month labels.")
    rows = []
    for origin in origins:
        selected = snapshot(observations, catalog, origin)
        period = pd.Period(origin, freq="M")
        for identifier in TARGETS:
            history = monthly_history(selected, identifier)
            if history.empty:
                continue
            for horizon in (3, 6, 9):
                target = (period + horizon).start_time
                actual = prices.get((identifier, target), np.nan)
                available = (
                    pd.Period(target, freq="M")
                    + PHASE3_LAGS[identifier].maximum_lag_months
                    + 1
                ).start_time
                rows.append(
                    {
                        "series_id": identifier,
                        "origin": period.start_time,
                        "horizon": horizon,
                        "target_month": target,
                        "actual": actual,
                        "latest": history.iloc[-1],
                        "last_observed_month": history.index[-1],
                        "available_at_estimated": available,
                        "log_error": np.log(actual / history.iloc[-1])
                        if pd.notna(actual) and actual > 0
                        else np.nan,
                    }
                )
    return pd.DataFrame(rows)


def known_errors(ledger, identifier, horizon, origin, *, dedicated_calibration=False):
    """No confirmation-origin errors ever enter either residual pool."""
    cutoff = (pd.Period(origin, freq="M") + 1).start_time
    selected = ledger.loc[
        (ledger.series_id == identifier)
        & (ledger.horizon == horizon)
        & (ledger.available_at_estimated < cutoff)
        & (ledger.origin < pd.Timestamp(origin))
        & (ledger.origin < pd.Timestamp("2024-01-01"))
        & ledger.log_error.notna()
    ].copy()
    if dedicated_calibration:
        selected = selected.loc[
            (selected.origin >= "2022-01-01") & (selected.origin <= "2023-12-01")
        ]
    return selected


def arch_gates(histories):
    """Exploratory per-origin BH across eligible series; never full-sample gating."""
    rows = []
    for identifier, prices in histories.items():
        returns = np.log(prices).diff().dropna()
        if len(returns) < 60:
            rows.append(
                {
                    "series_id": identifier,
                    "n_returns": len(returns),
                    "pvalue": np.nan,
                    "reason": "fewer than 60 returns",
                }
            )
        else:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                pvalue = het_arch(returns - returns.mean(), nlags=3, ddof=1)[1]
            rows.append(
                {
                    "series_id": identifier,
                    "n_returns": len(returns),
                    "pvalue": pvalue,
                    "reason": "training-only ARCH-LM",
                    "warnings": "; ".join(str(item.message) for item in caught),
                }
            )
    table = pd.DataFrame(rows)
    valid = table.pvalue.notna()
    if valid.any():
        table.loc[valid, "p_BH"] = multipletests(
            table.loc[valid, "pvalue"], method="fdr_bh"
        )[1]
    else:
        table["p_BH"] = np.nan
    table["eligible"] = table.p_BH < 0.1
    return table


def fit_garch(prices):
    returns = np.log(prices).diff().dropna()
    if len(returns) < 60:
        raise ValueError("GARCH requires 60 returns.")
    centered = 100 * (returns - returns.mean())
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fitted = arch_model(
            centered, mean="Zero", vol="GARCH", p=1, q=1, dist="normal", rescale=False
        ).fit(disp="off", show_warning=False)
    parameters = fitted.params
    omega, alpha, beta = (
        parameters["omega"],
        parameters["alpha[1]"],
        parameters["beta[1]"],
    )
    if (
        fitted.convergence_flag != 0
        or omega <= 0
        or min(alpha, beta) < 0
        or alpha + beta >= 0.995
    ):
        raise ValueError(
            "GARCH optimizer failed or persistence >=0.995; no stable stationary fit."
        )
    return {
        "omega": omega,
        "alpha": alpha,
        "beta": beta,
        "last_variance": float(fitted.conditional_volatility.iloc[-1] ** 2),
        "last_residual": float(fitted.resid.iloc[-1]),
        "warnings": len(caught),
        "volatility": pd.Series(
            fitted.conditional_volatility / 100, index=returns.index
        ),
        "observations": len(returns),
        "loglikelihood": fitted.loglikelihood,
    }


def garch_draws(fit, steps, simulations, rng):
    variance = np.full(simulations, fit["last_variance"])
    error = np.full(simulations, fit["last_residual"])
    paths = np.zeros(simulations)
    for _ in range(steps):
        variance = fit["omega"] + fit["alpha"] * error**2 + fit["beta"] * variance
        error = np.sqrt(variance) * rng.standard_normal(simulations)
        paths += error / 100
    return paths


def predictive_distribution(
    prices,
    steps,
    method,
    *,
    rng,
    errors=(),
    calibration=(),
    garch_fit=None,
    simulations=SIMULATIONS,
):
    if (
        not isinstance(prices.index, pd.DatetimeIndex)
        or prices.index.has_duplicates
        or steps < 1
    ):
        raise ValueError(
            "Unique monthly dates and a positive effective horizon required."
        )
    prices = prices.sort_index().asfreq("MS")
    if prices.isna().any() or (prices <= 0).any() or not np.isfinite(prices).all():
        raise ValueError("Complete finite positive history required; no imputation.")
    returns = np.log(prices).diff().dropna().to_numpy()
    if len(returns) < 24:
        raise ValueError("At least 24 monthly innovations required.")
    latest = prices.iloc[-1]
    if method == "TS_conformal":
        return conformal_quantiles(calibration, latest)
    if method == "LOCF_error_quantiles":
        if len(errors) < 12:
            raise ValueError(f"Insufficient known forecast errors: {len(errors)} < 12")
        changes = np.asarray(errors)
    elif method == "Historical_simulation":
        changes = np.log(prices / prices.shift(steps)).dropna().to_numpy()
        if len(changes) < 12:
            raise ValueError("Fewer than 12 historical horizon changes.")
    elif method == "Gaussian_RW":
        changes = rng.normal(0, returns.std(ddof=1), size=(simulations, steps)).sum(
            axis=1
        )
    elif method == "Innovation_bootstrap":
        changes = rng.choice(returns - returns.mean(), size=(simulations, steps)).sum(
            axis=1
        )
    elif method == "Moving_block_bootstrap":
        changes = block_draws(returns - returns.mean(), simulations, steps, rng).sum(
            axis=1
        )
    elif method == "GARCH":
        if garch_fit is None:
            raise ValueError(
                "No statistically eligible, stable training-only GARCH fit."
            )
        changes = garch_draws(garch_fit, steps, simulations, rng)
    else:
        raise ValueError(f"Unknown method {method}")
    return summarize(latest * np.exp(changes))


def quantile_scores(actual, quantiles, scale):
    lower, median, upper = (quantiles[name] for name in ("P10", "P50", "P90"))
    if (
        not 0 < lower <= median <= upper
        or not np.isfinite([actual, lower, median, upper, scale]).all()
        or scale <= 0
    ):
        raise ValueError(
            "Ordered positive quantiles and finite positive scaling required."
        )
    result = {}
    for tau, label, prediction in zip(
        (0.1, 0.5, 0.9), ("P10", "P50", "P90"), (lower, median, upper), strict=True
    ):
        error = actual - prediction
        result[f"pinball_{label}"] = max(tau * error, (tau - 1) * error)
        result[f"calibration_{label}"] = float(actual <= prediction)
    result["coverage_80"] = float(lower <= actual <= upper)
    result["width"] = upper - lower
    result["winkler"] = (
        upper - lower + 10 * max(lower - actual, 0) + 10 * max(actual - upper, 0)
    )
    result["scaled_winkler"] = result["winkler"] / scale
    result["scaled_width"] = result["width"] / scale
    result["scaled_pinball"] = (
        np.mean([result[f"pinball_{name}"] for name in ("P10", "P50", "P90")]) / scale
    )
    return result


def joint_scenarios(selected, simulations=SIMULATIONS):
    """Aligned vector block bootstrap; EUR prices derive pathwise from USD/FX.

    Uses a common observed anchor deliberately, discarding the additional July FX
    observation. These are dependence scenarios, not marginally reconciled draws
    from the selected univariate policy, and not BOM costs.
    """
    base_ids = [
        identifier for identifier in TARGETS if not identifier.startswith("D")
    ] + ["S06"]
    panel = (
        pd.concat(
            {
                identifier: monthly_history(selected, identifier)
                for identifier in base_ids
            },
            axis=1,
        )
        .dropna()
        .asfreq("MS")
    )
    if panel.isna().any().any() or len(panel) < 36:
        raise ValueError("Joint scenarios require 36 aligned complete months.")
    innovations = np.log(panel).diff().dropna()
    rng = generator("joint", panel.index[-1])
    steps = 12
    draws = block_draws(innovations.to_numpy(), simulations, steps, rng)
    levels = panel.iloc[-1].to_numpy()[None, None, :] * np.exp(draws.cumsum(axis=1))
    rows = []
    for horizon in (3, 6, 9):
        target = (pd.Period("2026-09", freq="M") + horizon).start_time
        distance = (
            pd.Period(target, freq="M").ordinal - panel.index[-1].to_period("M").ordinal
        )
        sample = levels[:, distance - 1, :]
        for column, identifier in enumerate(base_ids):
            rows.append(
                pd.DataFrame(
                    {
                        "scenario": np.arange(simulations),
                        "series_id": identifier,
                        "horizon": horizon,
                        "target_month": target,
                        "price": sample[:, column],
                        "anchor_month": panel.index[-1],
                    }
                )
            )
        for derived, numerator in (("D01", "S01"), ("D02", "S02")):
            rows.append(
                pd.DataFrame(
                    {
                        "scenario": np.arange(simulations),
                        "series_id": derived,
                        "horizon": horizon,
                        "target_month": target,
                        "price": sample[:, base_ids.index(numerator)]
                        / sample[:, base_ids.index("S06")],
                        "anchor_month": panel.index[-1],
                    }
                )
            )
    historical = innovations.corr()
    simulated = pd.DataFrame(draws.reshape(-1, len(base_ids)), columns=base_ids).corr()
    return pd.concat(rows, ignore_index=True), historical, simulated
