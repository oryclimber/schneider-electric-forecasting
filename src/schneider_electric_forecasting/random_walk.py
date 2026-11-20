"""Random-walk diagnostics with explicit nulls and multiplicity controls."""

import warnings

import numpy as np
import pandas as pd
from arch.unitroot import VarianceRatio
from statsmodels.api import OLS, add_constant
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.stats.multitest import multipletests
from statsmodels.tsa.stattools import acf, adfuller, kpss, pacf, zivot_andrews


def diagnostics(histories):
    """Use only supplied histories; results are evidence against specific nulls."""
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
