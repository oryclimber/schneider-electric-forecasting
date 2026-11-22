"""Predictive distributions, joint scenarios and random-walk diagnostics.

This module covers two related questions:

- **How uncertain is the forecast?** :func:`predictive_distribution` implements
  LOCF error quantiles, historical simulation, Gaussian random walk, innovation
  bootstrap, moving-block bootstrap, gated GARCH and split conformal intervals.
  It returns P10/P50/P90 and an expected price where the method defines one;
  :func:`quantile_scores` scores them with pinball loss, central coverage,
  interval width and Winkler loss.
- **Are the series forecastable, and how do they move together?**
  :func:`diagnostics` runs the random-walk test battery, and
  :func:`joint_scenarios` draws aligned multivariate blocks that preserve the
  observed cross-series dependence, deriving EUR prices pathwise from USD and
  FX.

Calibration pools never contain confirmation-origin errors. The joint scenarios
are exploratory: they use a common anchor and are not yet reconciled with the
selected marginal distributions, so their marginals can differ from the P10/P50/
P90 policy. Treat intervals as nominal central-80% limits, not guarantees.
"""

import hashlib
import math
import warnings

import numpy as np
import pandas as pd
from arch import arch_model
from arch.unitroot import VarianceRatio
from statsmodels.api import OLS, add_constant
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.stats.multitest import multipletests
from statsmodels.tsa.stattools import acf, adfuller, kpss, pacf, zivot_andrews

from .._04_features import (
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
    """Return a reproducible generator derived from the global seed and keys.

    Args:
        *keys: Values that identify the simulation (for example method, series
            and origin). Different keys give independent streams while the same
            keys reproduce the same draw.

    Returns:
        A seeded ``numpy.random.Generator``.
    """
    digest = int.from_bytes(
        hashlib.sha256("|".join(map(str, keys)).encode()).digest()[:8], "little"
    )
    return np.random.default_rng(np.random.SeedSequence([SEED, digest]))


def block_draws(innovations, simulations, steps, rng, block_length=3):
    """Sample contiguous blocks of rows; multivariate rows always travel together."""
    innovations = np.asarray(innovations)
    if steps < 1 or block_length < 1 or len(innovations) < block_length:
        raise ValueError("Insufficient innovations or invalid forecast/block length.")
    # Draw whole contiguous rows, not independent cells, so cross-series
    # dependence in each month survives the bootstrap.
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
    """Reduce simulated prices to the reported quantiles and mean.

    Args:
        samples: Array of simulated positive prices.

    Returns:
        A dict with ``P10``, ``P50``, ``P90`` and ``expected_price``.

    Raises:
        ValueError: If any sample is non-finite or non-positive.
    """
    samples = np.asarray(samples)
    if not np.isfinite(samples).all() or (samples <= 0).any():
        raise ValueError("Predictive prices must be finite and positive.")
    lower, median, upper = np.quantile(samples, [0.1, 0.5, 0.9])
    return {"P10": lower, "P50": median, "P90": upper, "expected_price": samples.mean()}


def conformal_quantiles(errors, latest, minimum=24):
    """Build split-conformal P10/P50/P90 from signed log errors.

    The endpoints are finite-sample order statistics of a dedicated calibration
    block. This is not a time-series coverage guarantee: exchangeability is not
    verified, and dependent or regime-shifted errors can invalidate nominal
    coverage. The median can differ from LOCF because signed errors are used.

    Args:
        errors: Signed log forecast errors from the calibration block only.
        latest: Latest observed price used to map errors back to price space.
        minimum: Minimum calibration errors required.

    Returns:
        A dict with ``P10``, ``P50``, ``P90`` and ``expected_price`` (NaN,
        because interval endpoints do not define a full distribution).

    Raises:
        ValueError: If there are too few calibration errors or the finite-sample
            endpoints fall outside the sample.
    """
    errors = np.sort(np.asarray(errors))
    n = len(errors)
    if n < minimum:
        raise ValueError(f"Insufficient conformal calibration: {n} < {minimum}")
    # Finite-sample one-sided order statistics: the floor/ceil indices give
    # nominal 10% and 90% tails without assuming a parametric distribution.
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
    """Build the error ledger used by all LOCF-based distribution methods.

    For every origin, series and horizon the ledger records the realised value,
    the last observable price, the signed log error and the estimated label
    publication date. Labels are stored separately from features so a later
    :func:`known_errors` call can exclude anything not yet published.

    Args:
        observations: Long observation table.
        catalog: Source catalog.
        origins: Rolling origins to include.

    Returns:
        One row per origin/series/horizon with ``actual``, ``latest``,
        ``log_error`` and availability metadata.

    Raises:
        ValueError: If the series/month index contains duplicates.
    """
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
    """Fit a training-only GARCH(1,1) and return the parameters needed to draw.

    Args:
        prices: Complete, positive monthly price history.

    Returns:
        A dict with ``omega``, ``alpha``, ``beta``, last variance/residual,
        conditional-volatility Series, observation count, log-likelihood and
        warning count. Returns are scaled by 100 for optimizer stability and the
        volatility Series is converted back.

    Raises:
        ValueError: If fewer than 60 returns are available, the optimizer fails,
            parameters are invalid, or persistence is at least 0.995 (in which
            case a stable stationary fit is not available).
    """
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
    """Simulate cumulative log returns from a fitted GARCH(1,1).

    Args:
        fit: Output of :func:`fit_garch`.
        steps: Number of monthly steps to simulate.
        simulations: Number of paths.
        rng: Seeded NumPy generator.

    Returns:
        An array of shape ``(simulations,)`` with cumulative log changes over
        ``steps`` months.
    """
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
    """Simulate a predictive price distribution for one horizon.

    The forecast is built in log space. Each method supplies a different set of
    cumulative log changes over ``steps`` months; multiplying ``latest *
    exp(change)`` gives the simulated price. Methods that are interval-only
    (``TS_conformal``) return no expected price and no simulation.

    Args:
        prices: Complete, positive, month-start history ending at the latest
            observation. Gaps and missing values are rejected, not imputed.
        steps: Effective number of months from the latest observation to the
            target, including any publication-delay gap. Must be at least 1.
        method: One of the names in :data:`METHODS`.
        rng: Seeded NumPy generator, normally from :func:`generator`.
        errors: Observed log forecast errors, required by
            ``LOCF_error_quantiles``.
        calibration: Dedicated signed log errors, required by ``TS_conformal``.
        garch_fit: Output of :func:`fit_garch`, required by ``GARCH``.
        simulations: Number of simulated paths for simulation-based methods.

    Returns:
        A dict with ``P10``, ``P50``, ``P90`` and ``expected_price``. For
        conformal intervals ``expected_price`` is NaN because the method defines
        only the two interval endpoints and the median.

    Raises:
        ValueError: If the history is incomplete/non-positive, the method is
            unknown, or the method's required inputs are missing.
    """
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
    """Score one predictive interval and its median against the realised value.

    Args:
        actual: Realised price, in the series' native unit.
        quantiles: Mapping with ordered ``P10``, ``P50`` and ``P90`` values.
        scale: Positive training scale used to make losses comparable across
            series (normally the one-step naive MAE).

    Returns:
        A dict with per-quantile pinball loss and calibration indicators,
        ``coverage_80``, interval ``width``, Winkler loss, and scaled Winkler,
        width and pinball scores.

    Raises:
        ValueError: If quantiles are not ordered/positive or the scale is not
            finite and positive.
    """
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
    """Draw aligned multi-material scenarios with pathwise FX conversion.

    Returns are bootstrapped in contiguous three-month blocks across all base
    series at once, so cross-series dependence is preserved within each block.
    EUR copper/aluminium are then derived from the same draw as USD price divided
    by USD-per-EUR; they are not simulated independently.

    Args:
        selected: Snapshot cut at the final origin, containing complete aligned
            histories for the target series and ``S06``.
        simulations: Number of joint paths.

    Returns:
        ``(scenarios, historical_correlation, simulated_correlation)``. The
        first is long by scenario/series/horizon; the other two are correlation
        matrices used to check that dependence survived the bootstrap.

    Notes:
        The function intentionally uses a common observed anchor and discards
        the extra July FX observation. The resulting marginals are therefore
        **not** reconciled with the selected univariate P10/P50/P90 policy; use
        it for dependence analysis, not as a substitute for the marginal
        forecasts or as a BOM cost.

    Raises:
        ValueError: If fewer than 36 complete aligned months are available.
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


# ---------------------------------------------------------------------------
# Random-walk and dependence diagnostics (formerly random_walk.py)
# ---------------------------------------------------------------------------


def diagnostics(histories):
    """Run the random-walk and dependence test battery on supplied histories.

    For each series this tests log levels and log returns with ADF and KPSS,
    Ljung-Box autocorrelation, robust variance ratios, HAC drift, ARCH effects,
    predeclared mean breaks (2020, 2022) and, when enough history exists,
    Zivot-Andrews. ACF/PACF are returned separately.

    Args:
        histories: Mapping of ``series_id`` to a complete, positive, monthly
            price Series.

    Returns:
        ``(tests, correlations)``. ``tests`` has one row per series/test with
        its family, explicit null, statistic, raw p-value, Holm global and BH
        family-adjusted p-values, and a status/warning field. ``correlations``
        contains ACF/PACF by lag.

    Raises:
        ValueError: If a history is shorter than 36 months, has gaps, or
            contains non-positive values.

    Notes:
        A non-rejection is not proof of a random walk; a rejection is evidence
        against one specific null. Multiplicity is controlled globally with Holm
        and within test families with Benjamini-Hochberg.
    """
    rows, correlations = [], []
    for identifier, prices in histories.items():
        prices = prices.asfreq("MS")
        if len(prices) < 36:
            raise ValueError(
                "At least 36 monthly prices required for the diagnostic battery."
            )
        if prices.isna().any() or (prices <= 0).any():
            raise ValueError("Diagnostics require complete positive monthly histories.")
        levels = np.log(prices)
        returns = levels.diff().dropna()

        def record(test, family, null, function, n, identifier=identifier):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                try:
                    statistic, pvalue, detail = function()
                    status = "ok"
                except (ValueError, np.linalg.LinAlgError) as error:
                    statistic, pvalue, detail, status = np.nan, np.nan, "", str(error)
            rows.append(
                {
                    "series_id": identifier,
                    "test": test,
                    "family": family,
                    "null": null,
                    "statistic": statistic,
                    "pvalue": pvalue,
                    "observations": n,
                    "detail": detail,
                    "status": status,
                    "warnings": "; ".join(str(item.message) for item in caught),
                }
            )

        for label, values in (
            ("log_level", levels),
            ("log_return", returns),
            ("level_pre2022", levels.loc[:"2021-12"]),
            ("level_post2022", levels.loc["2022-01":]),
        ):
            if len(values) < 24:
                continue

            def adf(values=values):
                result = adfuller(values, regression="c", autolag="AIC")
                return result[0], result[1], f"AIC selected lags={result[2]}"

            def stationarity(values=values):
                result = kpss(values, regression="c", nlags="auto")
                return (
                    result[0],
                    result[1],
                    f"bandwidth={result[2]}; tabulated p-values may be bounds",
                )

            record(f"ADF_{label}", "ADF", "unit root (with constant)", adf, len(values))
            record(
                f"KPSS_{label}", "KPSS", "level stationarity", stationarity, len(values)
            )
        for lag in (6, 12):

            def ljung(lag=lag, returns=returns):
                result = acorr_ljungbox(returns, lags=[lag], return_df=True).iloc[0]
                return result.lb_stat, result.lb_pvalue, "log-return autocorrelation"

            record(
                f"Ljung_Box_{lag}",
                "Ljung_Box",
                f"no autocorrelation through lag {lag}",
                ljung,
                len(returns),
            )
        for lag in (2, 3, 6, 12):

            def variance_ratio(lag=lag, levels=levels):
                result = VarianceRatio(
                    levels,
                    lags=lag,
                    trend="c",
                    robust=True,
                    overlap=True,
                    debiased=True,
                )
                return (
                    result.stat,
                    result.pvalue,
                    f"VR={result.vr:.6f}; heteroskedasticity robust, overlapping, debiased",
                )

            record(
                f"VR_{lag}",
                "Variance_Ratio",
                "variance ratio equals one; permits drift",
                variance_ratio,
                len(levels),
            )

        def drift(returns=returns):
            result = OLS(returns.to_numpy(), np.ones((len(returns), 1))).fit(
                cov_type="HAC", cov_kwds={"maxlags": 6}
            )
            return (
                result.tvalues[0],
                result.pvalues[0],
                f"mean log return={result.params[0]:.8f}; HAC bandwidth=6",
            )

        record(
            "Drift_HAC", "Drift", "zero mean monthly log return", drift, len(returns)
        )

        def arch(returns=returns):
            result = het_arch(returns - returns.mean(), nlags=3, ddof=1)
            return result[0], result[1], "ARCH-LM, 3 lags, demeaned log returns"

        record("ARCH_LM", "ARCH", "no ARCH effects through lag 3", arch, len(returns))
        for boundary in ("2020-01-01", "2022-01-01"):
            dummy = (returns.index >= boundary).astype(int)
            if min(dummy.sum(), len(dummy) - dummy.sum()) < 24:
                continue

            def break_mean(dummy=dummy, returns=returns):
                result = OLS(returns.to_numpy(), add_constant(dummy)).fit(
                    cov_type="HAC", cov_kwds={"maxlags": 6}
                )
                return (
                    result.tvalues[1],
                    result.pvalues[1],
                    "predeclared break date; HAC mean-shift test, not proof of causality",
                )

            record(
                f"Mean_break_{boundary[:4]}",
                "Break",
                "equal mean log returns before and after date",
                break_mean,
                len(returns),
            )
        if len(levels) >= 60:

            def endogenous_break(levels=levels):
                result = zivot_andrews(levels, maxlag=6, regression="ct", autolag="AIC")
                return (
                    result[0],
                    result[1],
                    f"one endogenous intercept/trend break: {levels.index[result[4]]:%Y-%m}; trim=15%",
                )

            record(
                "Zivot_Andrews",
                "Break",
                "unit root under one-break test",
                endogenous_break,
                len(levels),
            )
        aa, pp = (
            acf(returns, nlags=12, fft=False),
            pacf(returns, nlags=12, method="ywm"),
        )
        correlations.extend(
            {
                "series_id": identifier,
                "lag": lag,
                "ACF": aa[lag],
                "PACF": pp[lag],
                "observations": len(returns),
            }
            for lag in range(13)
        )
    results = pd.DataFrame(rows)
    valid = results.pvalue.notna()
    results.loc[valid, "p_holm_global"] = multipletests(
        results.loc[valid, "pvalue"], method="holm"
    )[1]
    for _, group in results.loc[valid].groupby("family"):
        results.loc[group.index, "p_BH_family_exploratory"] = multipletests(
            group.pvalue, method="fdr_bh"
        )[1]
    results["reject_Holm_5pct"] = results.p_holm_global < 0.05
    return results, pd.DataFrame(correlations)
