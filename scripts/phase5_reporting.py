"""Probabilistic diagnostics, calibration figures and risk-readiness reports."""

import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def table(frame):
    data = frame.copy()
    for column in data.select_dtypes(include="number"):
        data[column] = data[column].map(lambda value: f"{value:.4f}")
    return (
        "| "
        + " | ".join(data.columns)
        + " |\n| "
        + " | ".join(["---"] * len(data.columns))
        + " |\n"
        + "\n".join(
            "| " + " | ".join(map(str, row)) + " |"
            for row in data.itertuples(index=False, name=None)
        )
    )


def build_report(out):
    tests = pd.read_csv(out / "random_walk_tests.csv")
    macro = pd.read_csv(out / "scale_independent_macro.csv")
    choices = pd.read_csv(out / "frozen_probabilistic_policy.csv")
    final = pd.read_csv(
        out / "final_selected_probabilistic_forecasts.csv",
        parse_dates=["target_month", "last_observed_month"],
    )
    forecasts = pd.read_csv(
        out / "all_origin_distributions.csv", parse_dates=["origin", "target_month"]
    )
    policy_scores = pd.read_csv(out / "frozen_policy_performance.csv")
    matched = pd.read_csv(out / "challenger_matched_performance.csv")
    matched_macro = (
        matched.groupby(["comparison_set", "partition", "method"])
        .agg(
            coverage_80=("coverage_80", "mean"),
            scaled_winkler=("scaled_winkler", "mean"),
            scaled_pinball=("scaled_pinball", "mean"),
            scaled_width=("scaled_width", "mean"),
            material_horizon_cells=("series_id", "size"),
            total_origin_windows=("windows", "sum"),
        )
        .reset_index()
    )
    matched_macro.to_csv(out / "challenger_matched_macro.csv", index=False)
    audit = pd.read_csv(out / "distribution_audit.csv")
    colors = ["#3DCD58", "#0C3027", "#176B45", "#A2B8A5", "#D09380", "#86B2C6"]
    plt.rcParams.update(
        {
            "figure.facecolor": "#F4F1E8",
            "axes.facecolor": "#F4F1E8",
            "text.color": colors[1],
            "axes.labelcolor": colors[1],
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    def save(figure, name):
        figure.tight_layout()
        for extension in ("png", "svg"):
            figure.savefig(out / f"{name}.{extension}", dpi=300, bbox_inches="tight")
        plt.close(figure)

    for identifier, sample in final.groupby("series_id"):
        sample = sample.sort_values("horizon")
        fig, ax = plt.subplots(figsize=(11, 6))
        ax.fill_between(
            sample.target_month,
            sample.P10,
            sample.P90,
            color=colors[3],
            alpha=0.5,
            label="Central 80% marginal interval",
        )
        for name, shade in (("P10", colors[5]), ("P50", colors[2]), ("P90", colors[4])):
            ax.plot(
                sample.target_month, sample[name], marker="o", color=shade, label=name
            )
        ax.scatter(
            sample.last_observed_month.iloc[0],
            sample.latest_observation.iloc[0],
            color=colors[1],
            label="Latest eligible observation",
        )
        ax.axhline(
            sample.latest_observation.iloc[0],
            color=colors[0],
            linestyle="--",
            label="LOCF point benchmark",
        )
        ax.set(
            title=f"{identifier}: assumption-based September 2026 uncertainty",
            ylabel=sample.unit.iloc[0],
            xlabel="Target month (connecting lines are illustrative, not intermediate forecasts)",
        )
        ax.legend(ncols=2)
        save(fig, f"01_fan_{identifier}")
    fig, axes = plt.subplots(4, 3, figsize=(15, 13))
    for ax, (identifier, sample) in zip(
        axes.flat, final.groupby("series_id"), strict=True
    ):
        sample = sample.sort_values("horizon")
        ax.errorbar(
            sample.horizon,
            sample.P50,
            yerr=[sample.P50 - sample.P10, sample.P90 - sample.P50],
            fmt="o",
            capsize=5,
            color=colors[2],
        )
        ax.set(
            title=f"{identifier} ({sample.unit.iloc[0]})",
            xticks=[3, 6, 9],
            xlabel="Decision horizon (months)",
        )
    fig.suptitle(
        "Selected P10 / P50 / P90: actual observation-to-target gaps are 6 / 9 / 12 months"
    )
    save(fig, "02_quantiles_by_material_horizon")
    confirmation = macro.query("partition == 'confirmation'").sort_values(
        "scaled_winkler"
    )
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.barh(confirmation.method, confirmation.coverage_80, color=colors[2])
    ax.axvline(0.8, color=colors[4], linestyle="--")
    ax.set(
        title="Confirmation coverage: five core methods on identical windows",
        xlabel="Equal-material/horizon mean central 80% coverage",
        xlim=(0, 1),
    )
    save(fig, "03_prediction_interval_coverage")
    fig, ax = plt.subplots(figsize=(10, 7))
    for color, record in zip(colors, confirmation.itertuples(), strict=False):
        ax.plot(
            [0.1, 0.5, 0.9],
            [record.calibration_P10, record.calibration_P50, record.calibration_P90],
            marker="o",
            color=color,
            label=record.method,
        )
    ax.plot([0, 1], [0, 1], linestyle="--", color=colors[1])
    ax.legend()
    ax.set(
        title="Confirmation quantile calibration: nominal versus realized",
        xlabel="Nominal quantile",
        ylabel="Fraction actual ≤ predicted quantile",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    save(fig, "04_quantile_calibration")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].barh(confirmation.method, confirmation.scaled_pinball, color=colors[2])
    axes[0].set(
        title="Mean P10/P50/P90 pinball loss",
        xlabel="Training one-step-naive scaled loss (lower is better)",
    )
    axes[1].scatter(
        confirmation.scaled_width, confirmation.coverage_80, color=colors[2]
    )
    for record in confirmation.itertuples():
        labels = {
            "Gaussian_RW": ("Gaussian RW", 8),
            "Innovation_bootstrap": ("Innovation bootstrap", -12),
            "Moving_block_bootstrap": ("Block bootstrap", -10),
            "Historical_simulation": ("Historical simulation", 8),
            "LOCF_error_quantiles": ("Error quantiles", 8),
        }
        short_label, offset = labels[record.method]
        axes[1].annotate(
            short_label,
            (record.scaled_width, record.coverage_80),
            xytext=(6, offset),
            textcoords="offset points",
            fontsize=8,
        )
    axes[1].margins(x=0.25, y=0.15)
    axes[1].axhline(0.8, color=colors[4], linestyle="--")
    axes[1].set(
        title="Calibration versus sharpness",
        xlabel="Scaled interval width",
        ylabel="Central 80% coverage",
    )
    save(fig, "05_pinball_and_sharpness")
    volatility = pd.read_csv(
        out / "historical_conditional_volatility.csv", parse_dates=["month"]
    )
    fig, axes = plt.subplots(4, 3, figsize=(15, 13))
    for ax, (identifier, sample) in zip(
        axes.flat, volatility.groupby("series_id"), strict=True
    ):
        ax.plot(
            sample.month,
            sample.historical_12m_annualized_log_vol,
            color=colors[1],
            label="Trailing 12-month",
        )
        ax.plot(
            sample.month,
            sample.GARCH_annualized_log_vol,
            color=colors[2],
            label="GARCH filtered",
        )
        if "GARCH_forecast_annualized_log_vol" in sample:
            ax.plot(
                sample.month,
                sample.GARCH_forecast_annualized_log_vol,
                color=colors[4],
                linestyle="--",
                label="GARCH forecast",
            )
        ax.set(title=identifier, ylabel="Annualized log volatility")
        if sample.GARCH_annualized_log_vol.isna().all():
            ax.text(
                0.02,
                0.95,
                "GARCH unavailable: gate/fit failed",
                transform=ax.transAxes,
                va="top",
                fontsize=8,
            )
    axes.flat[0].legend(fontsize=8)
    fig.suptitle(
        "Historical versus statistically gated conditional volatility; no ineligible GARCH values fabricated"
    )
    save(fig, "06_historical_conditional_volatility")
    diagnostic_map = tests.pivot(
        index="series_id", columns="test", values="reject_Holm_5pct"
    ).astype(float)
    fig, ax = plt.subplots(figsize=(16, 7))
    image = ax.imshow(diagnostic_map, aspect="auto", vmin=0, vmax=1, cmap="Greens")
    ax.set(
        xticks=range(len(diagnostic_map.columns)),
        xticklabels=diagnostic_map.columns,
        yticks=range(len(diagnostic_map)),
        yticklabels=diagnostic_map.index,
        title="Random-walk diagnostics: rejection of EACH TEST'S null after global Holm correction",
    )
    ax.tick_params(axis="x", rotation=65)
    fig.colorbar(image, ax=ax, ticks=[0, 1], shrink=0.6)
    save(fig, "07_random_walk_diagnostic_summary")
    correlations = pd.read_csv(out / "return_acf_pacf.csv")
    fig, axes = plt.subplots(4, 3, figsize=(15, 12))
    for ax, (identifier, sample) in zip(
        axes.flat, correlations.groupby("series_id"), strict=True
    ):
        ax.plot(sample.lag, sample.ACF, color=colors[2], label="ACF")
        ax.plot(sample.lag, sample.PACF, color=colors[4], label="PACF")
        ax.set(title=identifier, xlabel="Monthly lag", ylim=(-1, 1))
    axes.flat[0].legend()
    fig.suptitle(
        "Log-return ACF / PACF: descriptive, not simultaneous significance tests"
    )
    save(fig, "08_return_acf_pacf")
    historical = pd.read_csv(
        out / "joint_historical_innovation_correlation.csv", index_col=0
    )
    simulated = pd.read_csv(
        out / "joint_simulated_innovation_correlation.csv", index_col=0
    )
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, panel, label in zip(
        axes,
        (historical, simulated),
        ("Historical aligned innovations", "Shared-block scenario innovations"),
        strict=True,
    ):
        image = ax.imshow(panel, vmin=-1, vmax=1, cmap="BrBG")
        ax.set(
            title=label,
            xticks=range(len(panel)),
            xticklabels=panel.columns,
            yticks=range(len(panel)),
            yticklabels=panel.index,
        )
        ax.tick_params(axis="x", rotation=90)
    fig.colorbar(image, ax=axes[-1], shrink=0.6)
    save(fig, "09_joint_innovation_dependence")
    # Dependence-aware descriptive coverage intervals only when two temporal blocks fit.
    from schneider_electric_forecasting.probabilistic import block_draws, generator

    uncertainty = []
    observed = forecasts.query("partition == 'confirmation'")
    for (identifier, horizon, method), sample in observed.groupby(
        ["series_id", "horizon", "method"]
    ):
        sample = sample.sort_values("origin")
        block = int(sample.effective_horizon.max())
        low, high = np.nan, np.nan
        if len(sample) >= 2 * block:
            draws = block_draws(
                sample.coverage_80.to_numpy(),
                2000,
                len(sample),
                generator("coverage", identifier, horizon, method),
                block_length=block,
            )
            low, high = np.quantile(draws.mean(axis=1), [0.025, 0.975])
            status = "descriptive block-bootstrap; very few effective blocks"
        else:
            status = "insufficient temporal blocks for meaningful coverage interval"
        uncertainty.append(
            {
                "series_id": identifier,
                "horizon": horizon,
                "method": method,
                "windows": len(sample),
                "overlap_block_length": block,
                "disjoint_origin_upper_count": math.ceil(len(sample) / block),
                "coverage": sample.coverage_80.mean(),
                "coverage_lower_95": low,
                "coverage_upper_95": high,
                "status": status,
            }
        )
    pd.DataFrame(uncertainty).to_csv(
        out / "coverage_dependence_uncertainty.csv", index=False
    )
    summary = (
        tests.groupby("test")
        .agg(series=("series_id", "size"), Holm_rejections=("reject_Holm_5pct", "sum"))
        .reset_index()
    )
    diagnostics_report = f"""# Random-walk diagnostic report

Phase 2 completed validated ADF/KPSS and ACF/PACF, but not the full requested random-walk battery. Phase 5 adds Ljung–Box, heteroskedasticity-robust overlapping/debiased variance-ratio tests, HAC drift tests, ARCH-LM and break sensitivity. All inputs are the histories available at the September decision under the existing lag assumptions, ending June 2026. Prices are positive source prices or indices; diagnostics use log levels and log returns.

{table(summary)}

The primary multiplicity control is Holm across all {len(tests)} tests, valid without an independence assumption. BH adjustments within predefined test families are separately labeled exploratory. ADF null is a unit root; KPSS null is level stationarity. These nulls differ, and return-stationarity rejection/acceptance cannot be read as a universal random-walk verdict. KPSS tabulated p-values are bounds when warnings indicate so. Ljung–Box uses lags 6 and 12; robust variance ratios use 2/3/6/12, allowing drift and heteroskedasticity. Drift uses a constant-only log-return regression with HAC bandwidth 6. ARCH-LM uses three lags on demeaned returns.

Break sensitivity includes ADF/KPSS on pre/post-2022 subperiods, predeclared 2020/2022 mean-return shifts with HAC inference, and a Zivot–Andrews one-break unit-root test. Its endogenous break search and nonstandard null are handled by the validated implementation, not an ordinary t-test. No breakpoint is retroactively used to optimize probabilistic confirmation forecasts. Subperiod tests have less power and parameter stability is not established by a non-rejection.

ACF/PACF are descriptive; no simultaneous significance claim is made from nominal lag bands. Heteroskedasticity-robust VR inference and HAC tests are still asymptotic with only about ten years of monthly data. Multiple testing, correlated/derived series, revisions and structural shifts limit conclusions. Failure to reject any null is NOT proof of a random walk, independence, constant volatility, zero drift or exchangeability. A random walk can permit drift and changing conditional variance; the Gaussian iid method is a stronger and separately evaluated assumption.

Validated variance-ratio implementation: [arch](https://arch.readthedocs.io/en/latest/_modules/arch/unitroot/unitroot.html). Full statistics, nulls, warnings, p-values and adjusted p-values are in random_walk_tests.csv.
"""
    (out / "random_walk_diagnostic_report.md").write_text(diagnostics_report)
    policy_confirmation = policy_scores.query("partition == 'confirmation'")
    final_display = final[
        [
            "series_id",
            "horizon",
            "method",
            "P10",
            "P50",
            "P90",
            "expected_price",
            "latest_observation",
            "unit",
        ]
    ]
    conformal_status = (
        audit.loc[audit.method == "TS_conformal"]
        .groupby(["partition", "status"])
        .size()
        .to_string()
    )
    garch_status = (
        audit.loc[audit.method == "GARCH"]
        .groupby(["partition", "status"])
        .size()
        .to_string()
    )
    median_difference = (final.P50 / final.latest_observation - 1).abs().max() * 100
    report = f"""# Phase 5: probabilistic forecasting and uncertainty quantification

## Policy and preserved evidence

This is a new LOCF-based probabilistic policy, not the frozen Phase 4 point policy. Phase 4 artifact hashes are saved and verified unchanged. No advanced mean model is forced into this phase. Latest eligible RM observations are June 2026, so December 2026 / March 2027 / June 2027 targets require 6 / 9 / 12 monthly steps from the last observation, despite decision horizons of 3 / 6 / 9 months. Forecasts remain assumption-based revised-snapshot estimates, not verified historical-vintage predictions.

## Selection, calibration and confirmation

Monthly 2020–2021 origins select among the five core methods using minimum mean training-scaled Winkler score, on identical successful windows (at least 18). The policy freezes December 31, 2022 after all selection labels are released. Dedicated 2022–2023 origins provide the 24 signed LOCF log-error calibration observations for conformal intervals. Their labels are available by the first January 2025 confirmation decision. Confirmation starts January 2025 and is restricted to labels available at the final September 2026 decision. Neither confirmation outcomes nor the 2024 gap's forecast errors enter method selection or either error-calibration pool. History-based innovation/volatility parameters update at each origin using only then-observable returns, as they would operationally; this is not retrospective model selection.

All source-specific publication lags, the provisional FX scenario and derived EUR dependency bounds are inherited. Residual distributions are gated by each target's own estimated publication timestamp. No future parameter estimation, future residual sampling or full-sample volatility gating is performed. Core methods share identical evaluation keys; intermittent GARCH and conformal comparisons have separately matched subsets. all_available_performance_UNPAIRED.csv is descriptive and must not be used to rank methods on different windows.

## Definitions and expectations

LOCF_error_quantiles multiplies the latest price by empirical signed historical LOCF forecast-error relatives at the same decision horizon (minimum 12 known errors). Historical_simulation uses realized past changes over the full effective gap, without inventing independent observations from overlapping changes. Gaussian_RW simulates zero-mean log innovations with training-only sample volatility. Innovation_bootstrap independently resamples demeaned log innovations. Moving_block_bootstrap resamples contiguous three-month blocks of those centered innovations. The latter preserves short serial dependence, not all long-memory or regime structure. Historical/error methods retain empirical shifts and asymmetry, while centered simulations impose zero log drift. Thus methods need not share predictive medians or means.

P10/P50/P90 and expected prices come from each empirical/simulated price distribution. Gaussian and GARCH expected prices include the nonlinear exponentiation effect. Empirical means are finite-sample scenario estimates, not guaranteed unbiased expectations. The largest selected median-versus-LOCF discrepancy is {median_difference:.2f}%; LOCF is a point benchmark and is NOT automatically the predictive median. Conformal is interval-only: P50 is the signed calibration median and expected price is intentionally unavailable, since endpoints do not define a full distribution.

## Actual paired confirmation results

{table(confirmation[["method", "coverage_80", "calibration_P10", "calibration_P50", "calibration_P90", "scaled_pinball", "scaled_winkler", "scaled_width"]])}

The frozen policy has equal-material/horizon mean coverage {policy_confirmation.coverage_80.mean():.4f}, scaled pinball {policy_confirmation.scaled_pinball.mean():.4f}, scaled Winkler {policy_confirmation.scaled_winkler.mean():.4f}, and scaled interval width {policy_confirmation.scaled_width.mean():.4f}. Selected method counts: {choices.selected_method.value_counts().to_dict()}. These are nominal central 80% prediction intervals, not retrospective historical percentiles relabeled as calibrated forecasts. Realized coverage may be poor; confirmation results do not change the frozen choices.

For the frozen policy, realized fractions below P10/P50/P90 are {policy_confirmation.calibration_P10.mean():.4f}/{policy_confirmation.calibration_P50.mean():.4f}/{policy_confirmation.calibration_P90.mean():.4f}, versus intended 0.1/0.5/0.9. Therefore near-80% average central coverage does NOT establish tail or median calibration. The worst cell covers only {policy_confirmation.coverage_80.min():.4f} of outcomes. Undercoverage on copper and broad lower-tail ranges matter for offers; aggregation hides this imbalance.

Matched GARCH and conformal confirmation summaries (equal weight to eligible material/horizon cells; compare methods ONLY within the same comparison set):

{table(matched_macro.query("partition == 'confirmation'"))}

Raw pinball, width and Winkler scores remain in source units. Macro summaries average per-material/horizon scores with equal weight and use each origin's one-step naive training MAE as a scale; they do not pool raw EUR/t, USD/t and index-point losses. Winkler uses alpha=0.2; quantile calibration reports fraction actual≤quantile, ideally 0.1/0.5/0.9. Narrower intervals are not preferred when misses increase proper scores.

Overlapping multi-step errors are dependent. Confirmation windows span only about one or two non-overlapping long-horizon blocks. coverage_dependence_uncertainty.csv uses effective-horizon moving blocks for descriptive coverage intervals only when two blocks fit; otherwise intervals are explicitly unavailable. No independent-Bernoulli confidence bounds or significance claims are made. Simulation count 3,000 improves Monte Carlo resolution, not economic sample size. Selection and calibration errors also overlap, so 24 nominal calibration observations are not 24 independent long-horizon errors.

The frozen per-cell method selection has a worse confirmation scaled Winkler score than all five fixed core-method portfolios in this run. This reflects selection instability with a short historical sample, not a reason to swap to a confirmation winner after the fact. The historical confirmation dates were already inspected during Phase 4; this is a new chronological Phase 5 analysis, not a fresh blind external holdout.

## GARCH and conformal limitations

GARCH is only attempted with at least 60 returns and a training-only ARCH-LM(3) BH q<0.1 across eligible series at that origin. A stable stationary GARCH(1,1), zero centered-log mean, Gaussian innovations and persistence<0.995 are required; failed optimizers and near-integrated fits remain unavailable. Per-origin exploratory FDR does not control all repeated gates across time or prove future volatility predictability. GARCH is a conditional-volatility challenger, not a selected core mean model. ARCH tests here assume a constant conditional mean after demeaning; residual serial correlation, especially in PPIs, can mimic ARCH evidence under mean misspecification. The gate does not prove that a Gaussian zero-mean GARCH density is correctly specified. A future study should test mean-whitened residuals and conditional-density diagnostics before operational use. Filtered and forward conditional volatility are separately plotted. Exact statuses:

```
{garch_status}
```

Time-series conformal uses two one-sided finite-sample signed-error order statistics, not a claimed exact time-series coverage theorem. It needs at least 24 eligible dedicated calibration errors. The 24-point crisis-era calibration block is small and dependent; exchangeability and regime stability are unverified. Asymmetric signed errors allow an asymmetric interval around the LOCF benchmark. Interval-only expected price is not fabricated. Exact statuses:

```
{conformal_status}
```

See challenger_matched_performance.csv for comparisons on precisely matched GARCH/conformal dates, and distribution_audit.csv for insufficient samples and numerical failures. GARCH and conformal were not eligible for selection from the earlier core-method period and are not silently substituted into the frozen policy.

## Final selected probabilistic forecasts

{table(final_display)}

Both latest observation month/value and availability assumptions accompany every forecast. P10/P90 are the central 80% limits. No future actuals or guaranteed risk bounds are fabricated.

## Joint material and FX risk

Joint scenarios resample aligned historical innovation vectors in shared three-month blocks. They preserve contemporaneous material/FX dependence and short within-block lag dependence. All series deliberately use the common June anchor, discarding the extra eligible July FX information; the scenarios are not conditioned on that extra observation. EUR copper/aluminium are calculated pathwise as simulated USD prices divided by simulated USD-per-EUR, preserving the currency identity. No independent marginal resampling is used. Historical and simulated innovation correlation matrices accompany the scenarios.

These uncentered empirical joint scenarios are a dependence stress-analysis facility, NOT draws reconciled to every selected univariate marginal. They include historical mean shifts and do not constitute a jointly calibrated final forecasting engine. Tail dependence outside the sample and extreme supply shocks remain unmodeled. A future aggregation must use coherent joint scenarios and validated units/quantities: quantile(sum of weighted prices) is generally NOT sum of individual weighted quantiles. No BOM sum, material weight, monetary exposure or cost P90 is estimated here.

## Readiness

The repository is ready for controlled uncertainty analysis and further out-of-sample calibration, not reliable fixed-price-offer guarantees or complete BOM costing. Resolve missing brass mappings, unknown material composition, BOM mass inconsistencies, validated procurement units/currencies, approved monetary weights, regional/grade premiums and contract pass-through. Obtain publication vintages and validate joint/marginal calibration before translating price scenarios into exposure. Coverages and broad intervals in this report must be reviewed explicitly rather than selecting a better-looking confirmation method. No BOM cost forecasting or pricing optimization was performed.

Reproduce with scripts/execute_eda.py --phase5. All confidential inputs and outputs remain Git-ignored. GARCH simulation follows [arch's documented recursion](https://arch.readthedocs.io/en/stable/univariate/forecasting.html). Conformal time-series assumptions follow the caution that [exchangeability and distribution shifts affect coverage](https://mapie.readthedocs.io/en/stable/content/exchangeability-testing/).
"""
    (out / "probabilistic_forecasting_report.md").write_text(report)
    return report
