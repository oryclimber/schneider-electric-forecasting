"""Build the complete EDA report from the configured source workbooks.

This module orchestrates the table and figure helpers, writes the private
outputs under ``data/eda_presentation/`` and applies the September origin
availability rules. It does not fit models or impute missing values.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .. import _01_ingestion as ingestion
from .. import config
from .._02_validation import (
    DOCUMENTED_PUBLICATION_LAG_ASSUMPTIONS,
    AvailabilityMode,
    select_available_observations,
    summarize_availability_cutoffs,
    validate_bom_material_mappings,
    validate_observation_snapshot,
)
from ._01_tables import (
    annualized_volatility,
    autocorrelation_diagnostics,
    correlation_matrix,
    detect_iqr_outliers,
    historical_coverage,
    lagged_correlations,
    monthly_returns,
    rolling_metrics,
    seasonality_summary,
    stationarity_tests,
    summary_statistics,
)
from ._02_plots import plot_correlation_matrix, plot_series


def run_eda_report(output_directory=None):
    """Rebuild the complete local EDA report from the configured workbooks.

    Loads the RM and energy long panels, validates them, derives catalog
    metadata and coverage, applies the September origin availability rules, then
    writes descriptive tables and nine figure pairs.

    Args:
        output_directory: Destination directory. Defaults to
            ``config.DATA_DIR / "eda_presentation"``. It is created if needed.

    Returns:
        A dict with ``tables`` (name to DataFrame), the enriched ``catalog``,
        the authoritative ``bom`` and the ``output_directory`` path.

    Side effects:
        Writes ``quality_checks.json``, ``data_catalog.csv``,
        ``canonical_bom.csv``, origin/availability tables, every analysis CSV
        and figures ``01``-``09`` as 300-DPI PNG and SVG. No source workbook is
        modified and no missing value is imputed.

    Notes:
        Full-sample and origin-only tables have separate names. Historical
        horizon percentiles are descriptive and must not be presented as
        predictive intervals.
    """

    out = Path(output_directory or config.DATA_DIR / "eda_presentation")
    out.mkdir(parents=True, exist_ok=True)
    rm = ingestion.load_rm_monthly_long()
    energy = ingestion.load_energy_monthly_long()
    catalogs = [
        ingestion.load_rm_source_catalog(),
        ingestion.load_energy_source_catalog(),
    ]

    quality = {}
    for name, sample, metadata in zip(
        ("raw_material", "energy"), (rm, energy), catalogs, strict=True
    ):
        report = validate_observation_snapshot(sample, metadata)
        quality[name] = {
            "valid": report.is_valid,
            "issues": [vars(issue) for issue in report.issues],
        }
        if not report.is_valid:
            raise ValueError(f"Invalid {name} snapshot: {report.errors}")
    (out / "quality_checks.json").write_text(json.dumps(quality, indent=2))
    all_data = pd.concat([rm, energy], ignore_index=True)
    catalog = pd.concat(catalogs, ignore_index=True)
    catalog["frequency"] = catalog["frequency"].fillna("M")
    energy_rows = catalog.series_id.str.startswith("E")
    inferred_currency = (
        catalog.unit.astype(str)
        .str.extract(r"^(USD|EUR|JPY)", expand=False)
        .fillna("n/a")
    )
    catalog.loc[energy_rows, "currency"] = inferred_currency.loc[energy_rows]
    catalog["classification"] = catalog.proxy_type
    catalog.loc[energy_rows, "classification"] = "Direct"
    catalog.loc[
        energy_rows & catalog.unit.str.contains("Index", case=False, na=False),
        "classification",
    ] = "Index proxy"
    catalog.loc[catalog.series_id.isin(["E20", "E21"]), "classification"] = (
        "Quantity driver"
    )
    catalog.loc[catalog.series_id.isin(["E23", "E24"]), "classification"] = "Conversion"
    catalog.loc[catalog.series_id.str.startswith("ED"), "classification"] = "Derived"
    catalog["source_file"] = np.where(
        energy_rows,
        str(config.CANONICAL_DATASET_PATHS["energy_workbook"]),
        str(config.CANONICAL_DATASET_PATHS["rm_price_workbook"]),
    )
    catalog["source_sheet"] = "Monthly_Long / Source_Catalog"
    catalog["definition"] = catalog.series_name
    coverage = historical_coverage(all_data)
    catalog = catalog.drop(
        columns=["first_month", "last_month", "n_months"], errors="ignore"
    ).merge(coverage, on="series_id", how="left", validate="one_to_one")
    catalog["publication_lag_months_assumed"] = catalog.series_id.map(
        {
            key: rule.maximum_lag_months
            for key, rule in DOCUMENTED_PUBLICATION_LAG_ASSUMPTIONS.items()
        }
    )
    catalog["verified_publication_timestamps"] = False
    catalog["publication_lag_evidence"] = catalog.series_id.map(
        {
            key: rule.evidence
            for key, rule in DOCUMENTED_PUBLICATION_LAG_ASSUMPTIONS.items()
        }
    ).fillna("No documented per-series lag; excluded from estimated origin inputs")
    catalog["availability"] = np.where(
        catalog.observed_count.fillna(0) > 0, "observed_snapshot", "unavailable"
    )
    catalog["limitations"] = (
        "No historical vintages; lag estimates are assumptions; proxies are not grade-specific prices."
    )
    catalog.to_csv(out / "data_catalog.csv", index=False)
    bom = ingestion.load_authoritative_bom()

    bom_report = validate_bom_material_mappings(bom, catalogs[0])
    quality["bom_mappings"] = {
        "valid": bom_report.is_valid,
        "issues": [vars(issue) for issue in bom_report.issues],
    }
    (out / "quality_checks.json").write_text(json.dumps(quality, indent=2))
    bom.to_csv(out / "canonical_bom.csv", index=False)
    ingestion.load_bom_metadata().to_csv(out / "bom_summary_metadata.csv", index=False)

    # September origin is interpreted as month end, matching the existing notebook.
    # Retrospective November retrieval is never treated as historical availability.
    as_of = config.FORECAST_DECISION_DATE
    supported = all_data.series_id.isin(DOCUMENTED_PUBLICATION_LAG_ASSUMPTIONS)
    origin = select_available_observations(
        all_data.loc[supported],
        catalog.loc[catalog.series_id.isin(DOCUMENTED_PUBLICATION_LAG_ASSUMPTIONS)],
        as_of=as_of,
        forecast_origin=config.FORECAST_ORIGIN,
        mode=AvailabilityMode.ASSUMPTION_BASED,
    )
    origin.to_csv(out / "origin_inputs_assumption_based.csv", index=False)
    origin.groupby("series_id").month.max().to_csv(out / "origin_cutoffs.csv")
    summarize_availability_cutoffs(origin).to_csv(
        out / "origin_availability_evidence.csv", index=False
    )
    # E23 duplicates S06 economically; use S06 once in combined diagnostics.
    analysis = all_data.loc[all_data.series_id != "E23"].copy()
    tables = {}
    for scope, sample in (("full_sample", analysis), ("origin_assumed", origin)):
        returns = monthly_returns(sample, value_column="value")
        tables[f"{scope}_returns"] = returns
        tables[f"{scope}_summary"] = summary_statistics(
            sample, value_column="value", group_by="series_id"
        )
        tables[f"{scope}_volatility"] = annualized_volatility(
            returns, value_column="value_pct_change", group_by="series_id"
        )
        tables[f"{scope}_rolling"] = rolling_metrics(
            sample, value_column="value", group_by="series_id"
        )
        tables[f"{scope}_correlations"] = correlation_matrix(
            returns, value_column="value_pct_change"
        )
        panel = (
            returns.pivot(index="month", columns="series_id", values="value_pct_change")
            .notna()
            .astype(int)
        )
        tables[f"{scope}_correlation_counts"] = panel.T @ panel
        wide = sample.pivot(index="month", columns="series_id", values="value").asfreq(
            "MS"
        )
        quantiles = []
        for horizon in config.FORECAST_HORIZONS:
            changes = (wide / wide.shift(horizon) - 1) * 100
            q = changes.quantile([0.1, 0.5, 0.9]).T.rename(
                columns={0.1: "P10", 0.5: "P50", 0.9: "P90"}
            )
            q["horizon_months"] = horizon
            q["observations"] = changes.count()
            quantiles.append(q.reset_index())
        tables[f"{scope}_historical_horizon_quantiles"] = pd.concat(
            quantiles, ignore_index=True
        )
    returns = tables["full_sample_returns"]
    tables["coverage"] = coverage
    tables["outliers"] = detect_iqr_outliers(
        returns, value_column="value_pct_change", group_by="series_id"
    )
    tables["seasonality"] = seasonality_summary(
        returns, value_column="value_pct_change", group_by="series_id"
    )
    tables["stationarity_levels"] = stationarity_tests(
        analysis, value_column="value", group_by="series_id"
    )
    tables["stationarity_returns"] = stationarity_tests(
        returns, value_column="value_pct_change", group_by="series_id"
    )
    # Leading return NaNs are boundary observations, not internal gaps.
    tables["acf_pacf"] = autocorrelation_diagnostics(
        returns.dropna(subset=["value_pct_change"]),
        value_column="value_pct_change",
        group_by="series_id",
    )
    tables["lagged_correlations"] = pd.concat(
        [
            lagged_correlations(
                returns,
                value_column="value_pct_change",
                target_series_id=target,
                explanatory_series_ids=("E01", "E05", "E22", "S06"),
            )
            for target in ("S01", "S02", "S09", "S11")
        ],
        ignore_index=True,
    )
    for name, frame in tables.items():
        frame.to_csv(
            out / f"{name}.csv",
            index=name.endswith(("correlations", "correlation_counts")),
        )

    colors = ["#3DCD58", "#0C3027", "#176B45", "#A2B8A5", "#D09380", "#86B2C6"]
    material_labels = catalogs[0].set_index("series_id").material.to_dict()

    def label(series_id):
        return f"{material_labels.get(series_id, series_id)} · {series_id}"

    plt.rcParams.update(
        {
            "figure.facecolor": "#F4F1E8",
            "axes.facecolor": "#F4F1E8",
            "text.color": "#0C3027",
            "axes.labelcolor": "#0C3027",
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    def save(figure, name):
        figure.tight_layout()
        for extension in ("png", "svg"):
            figure.savefig(out / f"{name}.{extension}", dpi=300, bbox_inches="tight")
        plt.close(figure)

    selected = ["S01", "S02", "S03", "S04", "S09", "S11"]
    wide = ingestion.load_rm_monthly_wide().set_index("month")
    fig, ax = plt.subplots(figsize=(13, 7))
    ax.imshow(
        wide.drop(columns=["year", "month_num"]).notna().T,
        aspect="auto",
        cmap="Greens",
        vmin=0,
        vmax=1,
    )
    ax.set_yticks(range(23), labels=wide.drop(columns=["year", "month_num"]).columns)
    ax.set_xticks(range(0, len(wide), 12), labels=wide.index.strftime("%Y")[::12])
    ax.set_title("Observed price coverage | blanks remain missing; L01–L08 unavailable")
    save(fig, "01_data_availability_heatmap")
    common = wide[selected].dropna().index.min()
    fig, ax = plt.subplots(figsize=(13, 6))
    for series, color in zip(selected, colors, strict=True):
        ax.plot(
            wide.index,
            wide[series] / wide.loc[common, series] * 100,
            label=label(series),
            color=color,
        )
    ax.set(
        title=f"Historical trends | {common:%b %Y} = 100; full sample",
        ylabel="Rebased index",
    )
    ax.legend(ncols=3)
    save(fig, "02_indexed_historical_material_prices")
    vol = (
        tables["origin_assumed_volatility"]
        .query("series_id in @selected")
        .sort_values("annualized_volatility")
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(vol.series_id, vol.annualized_volatility, color=colors[0])
    ax.set_yticks(
        range(len(vol)), labels=[label(series_id) for series_id in vol.series_id]
    )
    ax.set(
        title="Annualized return volatility | estimated September-origin inputs",
        xlabel="Monthly sample standard deviation × √12 (percentage points)",
    )
    save(fig, "03_pre_origin_annualized_volatility_ranking")
    roll = tables["full_sample_rolling"].query("series_id in @selected")
    fig = plot_series(roll, value_column="value_rolling_volatility_12m")
    for line, color in zip(fig.axes[0].lines, colors, strict=True):
        line.set_color(color)
        line.set_label(label(line.get_label()))
    fig.axes[0].legend(ncols=2)
    fig.axes[0].set_title("Twelve-month rolling volatility | full sample")
    save(fig, "04_12_month_rolling_volatility")
    corr_ids = selected + ["E01", "E05", "E22", "S06"]
    fig = plot_correlation_matrix(
        tables["full_sample_correlations"].loc[corr_ids, corr_ids]
    )
    from matplotlib.colors import LinearSegmentedColormap

    fig.axes[0].images[0].set_cmap(
        LinearSegmentedColormap.from_list(
            "schneider_association", ["#86B2C6", "#F4F1E8", "#176B45"]
        )
    )
    fig.axes[0].set_title("Monthly return correlations | full sample; descriptive")
    save(fig, "05_monthly_return_correlation_heatmap")
    q = tables["full_sample_historical_horizon_quantiles"].query(
        "series_id in @selected"
    )
    fig, axes = plt.subplots(1, 3, figsize=(16, 6), sharey=True)
    for ax, horizon in zip(axes, config.FORECAST_HORIZONS, strict=True):
        panel = q.loc[q.horizon_months == horizon].set_index("series_id").loc[selected]
        ax.hlines(range(len(panel)), panel.P10, panel.P90, color=colors[2], linewidth=4)
        ax.scatter(panel.P50, range(len(panel)), color=colors[4], zorder=3)
        ax.set(
            title=f"{horizon}-month historical changes",
            xlabel="Percent change",
            yticks=range(len(panel)),
            yticklabels=[label(series_id) for series_id in selected],
        )
        ax.axvline(0, color=colors[3])
    fig.suptitle(
        "Historical P10 / P50 / P90 | overlapping samples; not predictive intervals"
    )
    save(fig, "06_historical_price_risk_by_horizon")
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    for ax, series in zip(axes.flat, selected, strict=True):
        values = returns.loc[returns.series_id == series, "value_pct_change"].dropna()
        ax.hist(values, bins=20, color=colors[2])
        ax.set(title=series, xlabel="Monthly return (%)")
    fig.suptitle("Monthly return distributions | outliers retained")
    save(fig, "07_distributions")
    fig, ax = plt.subplots(figsize=(12, 6))
    for series, color in zip(selected, colors, strict=True):
        panel = tables["seasonality"].query("series_id == @series")
        ax.plot(panel.calendar_month, panel["mean"], label=series, color=color)
    ax.set(
        title="Average monthly returns by calendar month | descriptive seasonality",
        xlabel="Calendar month",
        ylabel="Mean return (%)",
    )
    ax.legend(ncols=3)
    save(fig, "08_seasonality")
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    for ax, series in zip(axes.flat, selected, strict=True):
        panel = tables["acf_pacf"].query("series_id == @series")
        ax.plot(panel.lag, panel.acf, label="ACF", color=colors[2])
        ax.plot(panel.lag, panel.pacf, label="PACF", color=colors[4])
        ax.set(title=series, xlabel="Lag (months)", ylim=(-1, 1))
        ax.legend()
    fig.suptitle("Return ACF and PACF | exploratory diagnostics")
    save(fig, "09_acf_pacf")
    return {"tables": tables, "catalog": catalog, "bom": bom, "output_directory": out}
