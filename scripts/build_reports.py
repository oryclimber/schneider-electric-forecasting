"""Build the local EDA, Phase 4 and Phase 5 reports.

All three builders read only private artifacts under ``data/`` and
write Markdown/CSV/figures back there. No source workbook is modified.
"""

import json
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap


def table(frame):
    """Render a DataFrame as a GitHub-flavoured Markdown table.

    Numeric columns are rounded to four decimals for readability; missing values
    become empty cells. The frame itself is not modified.

    Args:
        frame: Table to render.

    Returns:
        A Markdown table string.
    """
    frame = frame.copy()
    for column in frame.select_dtypes(include="number"):
        frame[column] = frame[column].map(lambda value: f"{value:.4f}")
    return (
        "| "
        + " | ".join(frame.columns)
        + " |\n| "
        + " | ".join(["---"] * len(frame.columns))
        + " |\n"
        + "\n".join(
            "| " + " | ".join(map(str, row)) + " |"
            for row in frame.fillna("").itertuples(index=False, name=None)
        )
    )


def build_eda_report():
    """Write the current-data and canonical-migration report.

    The report degrades gracefully when the optional pre-v1 archive is
    absent; it then records the unavailable comparison instead of
    failing or fabricating prior results.
    """
    from schneider_electric_forecasting import _01_ingestion as ingestion
    from schneider_electric_forecasting import config

    out = config.DATA_DIR / "eda_presentation"
    old = config.DATA_DIR / ".archive" / "before_v1"
    bom = ingestion.load_authoritative_bom()
    wide = ingestion.load_rm_monthly_wide().set_index("month")
    prerequisites = [old / "superseded_bom" / "RM_Prices_XHEC_Challenge.xlsx"] + [
        old / "eda_presentation" / name
        for name in (
            "pre_origin_volatility_ranking.csv",
            "historical_horizon_price_change_quantiles.csv",
            "return_correlations_and_sample_sizes.csv",
            "rolling_volatility_peaks.csv",
        )
    ]
    missing = [
        str(path.relative_to(config.DATA_DIR))
        for path in prerequisites
        if not path.is_file()
    ]
    if missing:
        # A clean checkout needs only authorized input workbooks. Historical
        # comparisons are optional evidence, never hidden executable dependencies.
        summary = (
            bom.groupby("product_ref")
            .agg(
                material_rows=("mass_kg", "size"),
                mass_sum_kg=("mass_kg", "sum"),
                share_sum=("share_pct", "sum"),
            )
            .reset_index()
        )
        report = (
            "# Current data and EDA report\n\nHistorical comparison unavailable: archived prior-period artifacts were not supplied. Current EDA completed; no previous results were invented.\n\n"
            + table(summary)
            + "\n\nMissing optional comparison inputs:\n\n"
            + "\n".join(f"- {path}" for path in missing)
            + "\n\nSee data_catalog.csv, quality_checks.json, coverage.csv and full_sample_*/origin_assumed_* tables for current results. Generated BOM values remain local. Prices, proxy indices and mass shares are distinct concepts; no complete BOM costs are estimated.\n"
        )
        (out / "migration_and_eda_report.md").write_text(report)
        print(out / "migration_and_eda_report.md")
        return
    old_bom = pd.read_excel(old / "superseded_bom" / "RM_Prices_XHEC_Challenge.xlsx")
    old_bom = old_bom.loc[old_bom.plant_code.notna()]
    keys = ["product_ref", "material_label_source"]
    changes = old_bom.merge(
        bom, on=keys, how="outer", suffixes=("_old", "_new"), indicator=True
    )
    changes.to_csv(out / "bom_changes.csv", index=False)
    ids = ["S01", "S02", "S03", "S04", "S05", "S09", "S11"]
    vol = pd.read_csv(out / "origin_assumed_volatility.csv").set_index("series_id")
    prior_vol = pd.read_csv(
        old / "eda_presentation" / "pre_origin_volatility_ranking.csv"
    ).set_index("series_id")
    comparison = vol.loc[ids, ["annualized_volatility"]].join(
        prior_vol[["annualized_volatility_pct"]]
    )
    comparison["change_percentage_points"] = (
        comparison.annualized_volatility - comparison.annualized_volatility_pct
    )
    comparison.to_csv(out / "volatility_comparison.csv")
    q = pd.read_csv(out / "full_sample_historical_horizon_quantiles.csv")
    corr = pd.read_csv(out / "full_sample_correlations.csv", index_col=0)
    prior_quantiles = pd.read_csv(
        old / "eda_presentation" / "historical_horizon_price_change_quantiles.csv"
    )
    quantile_comparison = q.merge(
        prior_quantiles, on=["series_id", "horizon_months"], suffixes=("_new", "_old")
    )
    for quantile in ("P10", "P50", "P90"):
        quantile_comparison[f"{quantile}_delta_pp"] = (
            quantile_comparison[quantile]
            - quantile_comparison[f"historical_{quantile.lower()}_pct"]
        )
    quantile_comparison.to_csv(out / "historical_quantile_comparison.csv", index=False)
    prior_correlations = pd.read_csv(
        old / "eda_presentation" / "return_correlations_and_sample_sizes.csv"
    )
    prior_correlations["updated_correlation"] = [
        corr.loc[row.series_1, row.series_2] for row in prior_correlations.itertuples()
    ]
    prior_correlations["delta"] = (
        prior_correlations.updated_correlation - prior_correlations.correlation
    )
    counts = pd.read_csv(out / "full_sample_correlation_counts.csv", index_col=0)
    prior_correlations["updated_paired_returns"] = [
        counts.loc[row.series_1, row.series_2]
        for row in prior_correlations.itertuples()
    ]
    prior_correlations.to_csv(out / "correlation_comparison.csv", index=False)
    prior_rolling = pd.read_csv(
        old / "eda_presentation" / "rolling_volatility_peaks.csv"
    )
    rolling = pd.read_csv(out / "full_sample_rolling.csv")
    peaks = rolling.groupby("series_id").value_rolling_volatility_12m.max()
    prior_rolling["updated_peak"] = prior_rolling.series_id.map(peaks)
    prior_rolling["delta_pp"] = (
        prior_rolling.updated_peak - prior_rolling.peak_12m_volatility_pct
    )
    prior_rolling.to_csv(out / "rolling_volatility_comparison.csv", index=False)
    moves = []
    for series in ids:
        observed = wide[series].dropna()
        moves.append(
            {
                "series_id": series,
                "first_month": str(observed.index.min().date()),
                "last_month": str(observed.index.max().date()),
                "first_value": observed.iloc[0],
                "last_value": observed.iloc[-1],
                "total_change_pct": (observed.iloc[-1] / observed.iloc[0] - 1) * 100,
            }
        )
    moves = pd.DataFrame(moves)
    moves.to_csv(out / "historical_material_moves.csv", index=False)
    summary = bom.groupby("product_ref").agg(
        material_rows=("mass_kg", "size"),
        share_sum=("share_pct", "sum"),
        mass_sum_kg=("mass_kg", "sum"),
    )
    cutoffs = pd.read_csv(out / "origin_cutoffs.csv")
    decisions = json.loads(
        (config.DATA_DIR / "audit" / "consolidation_decisions.json").read_text()
    )
    metadata = ingestion.load_bom_metadata()
    rm_catalog = ingestion.load_rm_source_catalog()
    rm_long = ingestion.load_rm_monthly_long()
    energy = ingestion.load_energy_monthly_long()
    measurements = wide.drop(columns=["year", "month_num"])
    observed_series = int(measurements.notna().any().sum())
    licensed_series = int(
        rm_catalog.access.astype(str).str.contains("licen", case=False).sum()
    )
    derived_rows = int(rm_long.series_id.str.startswith("D").sum())
    lines = [
        "# Canonical workbook migration and EDA comparison",
        "",
        "## Source audit and architecture",
        f"The authoritative BOM_Materials sheet has {len(bom) + len(metadata)} data rows: {len(bom)} materials and {len(metadata)} summary rows. It supplies BOM content; non-overlapping price histories and their metadata remain in the retained price workbook. Original workbook bytes are preserved.",
        f"The retained price workbook supplies observations, source catalog and dictionary. The independent energy workbook supplies {energy.series_id.nunique()} series. EUR prices D01/D02 are recomputed from source USD prices and USD-per-EUR FX, adding {derived_rows} derived rows to {len(rm_long) - derived_rows} source rows. BOM summary rows are exposed separately.",
        "Data inventory and workbook sheet/schema/missingness samples are in ../audit/. No source workbooks have been overwritten. No files were permanently deleted.",
        "",
        "## Retained sources and purpose",
        "- Canonical v1: BOM composition, original labels, masses, shares and price-series mappings.",
        "- Nested RM workbook: original price/FX observations, definitions, provenance, source catalog and dictionary. Its historical BOM sheet is superseded and never loaded.",
        "- Energy workbook: independent energy prices, import quantities, FX and derived regional comparisons. E23 duplicates S06 and is excluded from combined correlations.",
        "- Steel indexes, epoxy and LME workbooks: supplementary market/grade histories and methodology, retained for reconciliation; excluded from the core panel until their definitions and source-specific lags are harmonized.",
        "- Supplier financials and ZIP: independent supplier financial context, excluded from time-series EDA.",
        "- Benchmark PDF/XLSX: independent competitor/product context; synthetic RMU benchmark: scenario illustration only, never observed target prices. The top-level synthetic XLSX is not a valid XLSX ZIP container and is retained for investigation; the nested workbook is readable.",
        "- Source README and dictionary: retained documentation. data.zip is an unavailable cloud placeholder and remains untouched.",
        "- Prior BOMs and CSV exports: preserved in .archive/before_v1 with prior EDA; unverified cloud exports are archived without deletion, not declared exact duplicates.",
        "",
        "## Consolidation decisions",
        table(pd.DataFrame(decisions)),
        "",
        "## BOM changes",
        f"Material rows increase from {len(old_bom)} to {len(bom)}. Detailed outer-join differences are in bom_changes.csv. New labels, removed placeholder labels and revised existing quantities are explicit; mass shares are not interpreted as spend weights.",
        table(summary.reset_index()),
        "Rounded source quantities are preserved and summarized above; no client quantities are embedded in this source code. Unmapped or unidentified material rows remain unresolved. Plastic-family proxies do not establish grade-specific purchasing prices. Detailed current quantities and discrepancies belong only in the ignored generated tables.",
        "",
        "## Coverage and quality",
        f"RM wide dimensions: {len(wide)} months × {len(wide.columns) + 1} columns including month; catalog: {len(rm_catalog)} series, {observed_series} with observations and {licensed_series} licensed series. Window {wide.index.min():%Y-%m}–{wide.index.max():%Y-%m}; most price series end September 2026 and FX ends October. Raw measurement cells missing: {int(measurements.isna().sum().sum())} of {measurements.size} (including licensed cells). Energy has {len(energy)} observations across {energy.series_id.nunique()} series.",
        "Snapshot validation found no duplicate observation keys, internal missing calendar periods, non-positive/nonfinite values or catalog unit/currency/code/region disagreements. Outliers are flagged, retained and not assumed erroneous. Tail missingness and licensed absence remain explicit. Full checks are in quality_checks.json and coverage.csv.",
        "",
        "## September-origin availability",
        "Origin month remains September 2026; decision cutoff is explicitly September 30. Under the existing conservative rule, availability is the start of observation month + maximum lag + one month, and must be strictly before the cutoff. This recalculates to June 2026 for IMF/PPI series and April for import quantities, matching the prior result for documented series. These are assumptions, not verified vintages. FX and other providers have no supported origin lag and remain outside the origin panel. November retrieval cannot prove September availability. Changing the decision to September 1 would require rerunning with that cutoff.",
        table(cutoffs.groupby("month").size().rename("series_count").reset_index()),
        "",
        "## Recomputed historical movements",
        table(moves),
        "",
        "## Annualized volatility comparison",
        "Values are percentage points: sample monthly return standard deviation × √12. Same source snapshot and origin sample imply unchanged estimates; BOM revisions do not alter commodity history.",
        table(comparison.reset_index()),
        "",
        "## Correlations and horizon changes",
        f"Full-sample monthly-return correlation: copper/aluminium {corr.loc['S01', 'S02']:.4f}; Brent/plastics {corr.loc['E01', 'S11']:.4f}; TTF/aluminium PPI {corr.loc['E05', 'S08']:.4f}. Pairwise sample counts accompany the matrix. Lag scans are descriptive and use earlier explanatory values; they do not validate forecasting drivers.",
        table(q.loc[q.series_id.isin(["S01", "S02", "S09", "S11"])]),
        "Historical percentiles use overlapping realized price changes and are not predictive intervals. Full sample includes post-origin outcomes. Origin-only quantiles are exported separately.",
        "",
        "## Previous presentation findings",
        "The largest correlation difference is FX versus German power: the regenerated full-sample panel includes October for both series, while the previous presentation stopped in September. Pair counts are exported in correlation_comparison.csv. This is a sample-window change rather than a BOM effect.",
        f"Maximum absolute matched differences: annualized origin volatility {comparison.change_percentage_points.abs().max():.8f} pp; rolling-volatility peaks {prior_rolling.delta_pp.abs().max():.8f} pp; correlations {prior_correlations.delta.abs().max():.8f}; historical percentiles {quantile_comparison[[f'{q}_delta_pp' for q in ('P10', 'P50', 'P90')]].abs().max().max():.8f} pp. Detailed comparisons are exported; small changes can reflect Excel versus rounded CSV precision.",
        "Commodity history is unchanged: price changes, rolling volatility, return correlations and historical percentiles remain numerically valid when compared on the same series, units, calendar and sample. The wider regenerated diagnostics include energy, FX, derived EUR prices and all retained series, so dimensions differ. Prior copper/steel/plastics BOM exposure conclusions require revision from the new composition. The previous approximate ADF p-values and faulty PACF implementation are superseded by statsmodels ADF (AIC lag selection), KPSS (automatic bandwidth) and Yule-Walker PACF. KPSS p-values outside the tabulated range are reported as bounds. These tests remain sensitive to breaks and multiple testing.",
        "",
        "## Forecasting and pricing implications",
        "Proceed with feature definitions and baseline design using lagged monthly returns, energy drivers, explicit FX quotes, rolling volatility and training-only transformations. Use rolling-origin validation and vintages or approved lag assumptions. Keep direct commodity prices, cost drivers, PPI proxies and synthetic benchmarks distinct. Do not price product costs from PPI index levels or mass shares alone. Brass, unidentified materials, regional premiums, contract pass-through and uncertain BOM masses need resolution before complete product-cost forecasting. No forecasting models were trained.",
        "",
        "## Reproducibility and deliverables",
        "Source code remains under src/schneider_electric_forecasting; scripts audit/consolidate sources and write this report. The notebook calls the existing eda module and executes all diagnostics. Figures 01–09 are PNG at 300 DPI plus SVG. All numerical reports, data, archives, executed notebooks and charts remain in ignored data/. README documents commands. Validation results and code-quality execution are recorded separately in verification.txt.",
    ]
    (out / "migration_and_eda_report.md").write_text("\n\n".join(lines) + "\n")
    print(out / "migration_and_eda_report.md")


# ---------------------------------------------------------------------------
# Phase 4 report (formerly phase4_reporting.py)
# ---------------------------------------------------------------------------


def build_phase4_report(out):
    """Write Phase 4 Markdown, CSV summaries and diagnostic figures.

    Args:
        out: Phase 4 output directory containing the experiment CSVs.

    Side effects:
        Writes report tables and figures into ``out``. It reads frozen selection
        and confirmation scores but never changes them.
    """
    scores = pd.read_csv(out / "model_performance.csv")
    predictions = pd.read_csv(
        out / "outer_predictions.csv", parse_dates=["origin", "target_month"]
    )
    selection = pd.read_csv(out / "frozen_selection.csv")
    final = pd.read_csv(out / "final_selected_forecasts.csv")
    units = final.groupby("series_id").unit.first().to_dict()
    aggregate = pd.read_csv(out / "scale_independent_aggregate.csv")
    holdout = scores.query("partition == 'confirmation'")
    gains = holdout.query("model != 'LOCF'")
    best = (
        gains.sort_values("MAE").groupby(["series_id", "horizon"]).first().reset_index()
    )
    best.to_csv(out / "descriptive_confirmation_best_NOT_SELECTION.csv", index=False)
    policy = holdout.merge(
        selection[["series_id", "horizon", "selected_model"]],
        on=["series_id", "horizon"],
    )
    policy = policy.loc[policy.model == policy.selected_model]
    policy.to_csv(out / "frozen_policy_performance.csv", index=False)
    win_rates = pd.read_csv(out / "origin_win_rates.csv")
    regimes = pd.read_csv(out / "regime_performance.csv")
    next_steps = []
    for record in policy.itertuples():
        rates = win_rates.loc[
            (win_rates.partition == "confirmation")
            & (win_rates.series_id == record.series_id)
            & (win_rates.horizon == record.horizon)
            & (win_rates.model == record.model)
        ]
        years = regimes.loc[
            (regimes.partition == "confirmation")
            & (regimes.series_id == record.series_id)
            & (regimes.horizon == record.horizon)
            & (regimes.model == record.model)
            & (regimes.valid_windows >= 2)
        ]
        win_rate = rates["mean"].iloc[0]
        eligible = (
            record.model != "LOCF"
            and record.MAE_improvement_pct >= 5
            and win_rate >= 0.6
            and len(years) >= 2
            and years.MAE_improvement_pct.gt(0).all()
        )
        next_steps.append(
            {
                "series_id": record.series_id,
                "horizon": record.horizon,
                "frozen_model": record.model,
                "confirmation_gain_pct": record.MAE_improvement_pct,
                "confirmation_win_fraction": win_rate,
                "probabilistic_challenger": eligible,
                "next_step": "Calibrate challenger alongside mandatory LOCF"
                if eligible
                else "LOCF calibration benchmark; advanced gain not sufficiently consistent",
            }
        )
    next_steps = pd.DataFrame(next_steps)
    next_steps.to_csv(out / "probabilistic_readiness.csv", index=False)
    color = ["#3DCD58", "#0C3027", "#176B45", "#A2B8A5", "#D09380", "#86B2C6"]
    plt.rcParams.update(
        {
            "figure.facecolor": "#F4F1E8",
            "axes.facecolor": "#F4F1E8",
            "text.color": color[1],
            "axes.labelcolor": color[1],
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    cmap = LinearSegmentedColormap.from_list(
        "schneider", [color[4], "#F4F1E8", color[2]]
    )

    def save(fig, name):
        fig.tight_layout()
        for extension in ("png", "svg"):
            fig.savefig(out / f"{name}.{extension}", dpi=300, bbox_inches="tight")
        plt.close(fig)

    holdout = holdout.assign(
        cell=holdout.series_id + " h" + holdout.horizon.astype(str)
    )
    for field, name, title in (
        (
            "MASE",
            "01_performance_heatmap",
            "Confirmation MASE (one-step training scale; lower is better)",
        ),
        (
            "MAE_improvement_pct",
            "02_improvement_vs_LOCF",
            "Confirmation MAE improvement versus LOCF (%)",
        ),
    ):
        panel = holdout.pivot(index="cell", columns="model", values=field)
        fig, ax = plt.subplots(figsize=(13, 12))
        values = panel.to_numpy()
        bound = (
            max(abs(np.nanpercentile(values, 5)), abs(np.nanpercentile(values, 95)))
            if field != "MASE"
            else None
        )
        image = ax.imshow(
            values,
            aspect="auto",
            cmap=cmap if field != "MASE" else "YlGn",
            vmin=-bound if bound else None,
            vmax=bound,
        )
        ax.set(
            xticks=range(len(panel.columns)),
            xticklabels=panel.columns,
            yticks=range(len(panel)),
            yticklabels=panel.index,
            title=title,
        )
        ax.tick_params(axis="x", rotation=65)
        fig.colorbar(image, ax=ax, shrink=0.7)
        save(fig, name)
    rankings = holdout.copy()
    rankings["MAE_rank"] = rankings.groupby(["series_id", "horizon"]).MAE.rank(
        method="min"
    )
    rankings.to_csv(out / "confirmation_rankings.csv", index=False)
    panel = rankings.pivot(index="cell", columns="model", values="MAE_rank")
    fig, ax = plt.subplots(figsize=(13, 12))
    image = ax.imshow(panel, aspect="auto", cmap="YlGn", vmin=1, vmax=13)
    ax.set(
        xticks=range(len(panel.columns)),
        xticklabels=panel.columns,
        yticks=range(len(panel)),
        yticklabels=panel.index,
        title="Confirmation MAE rank by material and horizon (ties share rank)",
    )
    ax.tick_params(axis="x", rotation=65)
    fig.colorbar(image, ax=ax, shrink=0.7)
    save(fig, "03b_rankings_by_material_horizon")
    fig, ax = plt.subplots(figsize=(12, 6))
    rank = rankings.groupby("model").MAE_rank.mean().sort_values()
    ax.barh(rank.index, rank, color=color[2])
    ax.invert_yaxis()
    ax.set(
        title="Average confirmation rank across material/horizon cells",
        xlabel="Mean MAE rank (lower is better)",
    )
    save(fig, "03_model_rankings")
    for identifier, block in predictions.query("partition == 'confirmation'").groupby(
        "series_id"
    ):
        fig, axes = plt.subplots(3, 2, figsize=(14, 10))
        for row, horizon in enumerate((3, 6, 9)):
            sample = block.loc[block.horizon == horizon]
            choice = selection.query(
                "series_id == @identifier and horizon == @horizon"
            ).selected_model.iloc[0]
            actual = sample.query("model == 'LOCF'").sort_values("target_month")
            axes[row, 0].plot(
                actual.target_month, actual.actual, color=color[1], label="Actual"
            )
            for model, shade in (("LOCF", color[0]), (choice, color[4])):
                if model == "LOCF" and choice == "LOCF":
                    label = "LOCF (selected)"
                else:
                    label = model
                group = sample.loc[sample.model == model].sort_values("target_month")
                axes[row, 0].plot(
                    group.target_month, group.prediction, color=shade, label=label
                )
                axes[row, 1].plot(group.origin, group.error, color=shade, label=label)
                if choice == "LOCF":
                    break
            axes[row, 0].set_title(f"{identifier}, h={horizon}: forecasts and actuals")
            axes[row, 1].set_title("Rolling signed errors by forecast origin")
            axes[row, 0].set_ylabel(units[identifier])
            axes[row, 1].set_ylabel(units[identifier])
            axes[row, 1].axhline(0, color=color[3])
            axes[row, 0].legend()
            axes[row, 1].legend()
        fig.suptitle(
            "Confirmation folds only; model family frozen before 2024; original source units"
        )
        save(fig, f"04_forecasts_errors_{identifier}")
    importance = pd.read_csv(out / "tree_feature_importance.csv")
    for model, group in importance.groupby("model"):
        fig, ax = plt.subplots(figsize=(10, 7))
        values = group.groupby("feature").importance.mean().nlargest(15).sort_values()
        ax.barh(values.index, values, color=color[2])
        ax.set(
            title=f"{model}: mean final-fit tree importance",
            xlabel="Split gain / impurity importance (model-specific; not causal)",
        )
        save(fig, f"05_importance_{model}")
    fig, axes = plt.subplots(4, 3, figsize=(14, 13))
    for ax, (identifier, sample) in zip(
        axes.flat, final.groupby("series_id"), strict=True
    ):
        ax.plot(sample.horizon, sample.prediction, marker="o", color=color[2])
        ax.set(
            title=f"{identifier} ({sample.unit.iloc[0]})",
            xticks=[3, 6, 9],
            xlabel="Horizon from September (months)",
        )
    fig.suptitle(
        "Selected final point forecasts: assumption-based September 30, 2026 decision"
    )
    save(fig, "06_final_forecasts")
    baseline = holdout.query("model == 'LOCF'").sort_values("MASE", ascending=False)
    rows = []
    for record in policy.sort_values(["series_id", "horizon"]).itertuples():
        rows.append(
            f"| {record.series_id} | {record.horizon} | {record.selected_model} | {record.valid_windows} | {record.MAE:.3f} | {record.MASE:.3f} | {record.MAE_improvement_pct:.2f} |"
        )
    table = (
        "| Series | Horizon | Frozen choice | Windows | MAE | MASE | MAE improvement % |\n|---|---:|---|---:|---:|---:|---:|\n"
        + "\n".join(rows)
    )
    aggregate_table = (
        aggregate.query("partition == 'confirmation'")
        .sort_values("macro_MASE")
        .to_string(index=False)
    )
    comparisons = len(best)
    improved = int((best.MAE_improvement_pct > 0).sum())
    robust = int((policy.MAE_improvement_pct >= 5).sum())
    report = f"""# Phase 4: advanced forecasting and evaluation

## Executed experiment

Twelve eligible raw-material/derived series, three horizons, LOCF plus ARIMA, ETS, Ridge, Elastic Net, XGBoost and LightGBM. Each advanced family compares source-unit level targets with cumulative log-return targets. This gives 13 candidates. Supervised origins extend from January 2018 through September 2026 (3,780 rows), using the same Phase 3 feature definitions. Earlier training origins supply enough publication-purged examples for nested validation; they are not added to the confirmation evaluation grid. Original data loaders, feature engine, publication selector, baseline and metric functions are reused. Deep learning is excluded because of the limited monthly sample.

## Leakage controls and coverage

Quarterly outer selection origins cover 2021–2022. Model-family choices freeze December 31, 2023; all selection labels must have been released by then. The intervening gap accommodates the nine-month horizon and publication delay. Separate quarterly confirmation origins begin January 2024. Confirmatory outcomes never choose the final family. Inner validation uses two recent origins whose labels are known at the outer origin; each inner training set is independently purged by its own label-release cutoff. Minimum inner/outer supervised counts are 12/18. Univariate histories require 36 complete months. The latest two-fold tuning score chooses between two predeclared configurations for each candidate; all fold dates, releases, losses, warnings and options are recorded.

All candidates share the intersection of successful outer windows separately for each series/horizon. coverage.csv and fit_failures.csv report individual candidate coverage and joint exclusions. This differs deliberately from Phase 3's monthly 75/72/69 windows: Phase 4 uses a quarterly grid, a historical selection period, a purge interval and a distinct later confirmation period. No failed advanced forecast is silently replaced by LOCF. Reported LOCF metrics are recomputed on exactly the shared Phase 4 keys.

Missing feature values are median-imputed inside each inner/outer training fit. Ridge/Elastic Net standardization is fitted there too. The 24-feature economic subset was specified before inspecting Phase 4 outcomes; there is no whole-sample learned feature selection. ARIMA and ETS forecast through the entire publication-delay gap. Return formulations forecast log changes then exponentiate back to source units; no fitted full-sample bias correction is used. Negative level forecasts are not clipped.

## Actual confirmation results

At least one advanced candidate has lower descriptive confirmation MAE than LOCF in {improved}/{comparisons} cells. This is a retrospective comparison across many candidates, not the selection rule. The frozen policy improves confirmation MAE by at least 5% in {robust}/36 cells. Frozen selections: {selection.selected_model.value_counts().to_dict()}.

{table}

MAE and bias are in each series' original units and must not be pooled. MASE uses each origin's training-only one-step naive MAE (m=1); MASE above one alone does not establish failure against multistep LOCF. Scale-independent macro summaries give equal weight to material/horizon cells:

```
{aggregate_table}
```

origin_win_rates.csv records the fraction of individual origins beating LOCF. regime_performance.csv separates the 2021 recovery, 2022 energy-shock period and annual confirmation blocks. These regimes are predefined calendar descriptions, not causally identified states. Confirmation rankings and best candidates are explicitly descriptive and do not override the frozen policy.

The frozen policy has confirmation macro MASE {policy.MASE.mean():.4f}, versus {holdout.loc[holdout.model == "LOCF", "MASE"].mean():.4f} for LOCF. Its equal-cell mean relative MAE improvement is {policy.MAE_improvement_pct.mean():.2f}% (negative means worse). Only {int((policy.loc[policy.model != "LOCF", "MAE_improvement_pct"] > 0).sum())} of the six preselected advanced choices improves confirmation MAE, and none meets the probabilistic challenger robustness screen. These results favor keeping LOCF as the next-stage benchmark.

## Selection and complexity

An advanced family must show at least 5% selection MAE improvement, win at least 60% of paired selection origins, improve in both 2021 and 2022 and have at least six selection folds. Otherwise LOCF remains selected. These are predeclared practical gates, not significance tests. Eight quarterly selection folds and seven-to-nine nominal confirmation folds provide limited evidence, and horizon errors overlap. The post-selection confirmation results are an honest stress test; negative confirmation results are retained rather than using them to choose a different candidate retrospectively.

The most difficult material/horizon cells by confirmation LOCF MASE include {baseline[["series_id", "horizon", "MASE"]].head(5).to_dict("records")}. Compare these dimensionless errors alongside units, regime tables and signed bias; commodity shocks and PPI regime shifts are not eliminated by added complexity. Strong final-fit tree importance does not prove predictive causality or stability. Correlated price lags can substitute for one another, and LightGBM split importance is not numerically comparable with XGBoost gain importance.

## Final forecasts and probabilistic next steps

final_selected_forecasts.csv contains 36 frozen-policy point forecasts for December 2026, March 2027 and June 2027, generated from the September 30, 2026 decision. All 468 candidate forecasts are also exported. Last RM observations are June, implying effective distances of 6/9/12 months. Final fits retune hyperparameters using only labels available at that decision; family choices remain frozen. The final tree importances describe those fits, not holdout explanations.

LOCF should proceed as the mandatory probabilistic benchmark. Advanced candidates should proceed only as experimental challengers where the frozen policy also shows positive confirmation gains and reasonable origin/regime consistency; use material-specific confirmation and regime tables rather than a global winner. Use rolling conformal or residual-based calibration inside a new validation design, with adequate calibration observations and explicit revision/availability assumptions. No predictive intervals were fabricated in this phase.

The explicit readiness screen requires the frozen advanced model to achieve at least 5% confirmation MAE gain, win at least 60% of origins, and improve in at least two annual confirmation blocks with two or more folds each. Models meeting this screen: {next_steps.loc[next_steps.probabilistic_challenger, ["series_id", "horizon", "frozen_model"]].to_dict("records")}. probabilistic_readiness.csv records all 36 decisions. This screen guides future experimentation only and does not retrospectively change the frozen point-forecast policy.

## Tracking, reproducibility and limitations

Local MLflow SQLite metadata is in mlflow.db and artifacts in mlflow_artifacts/. The parent run records source SHA-256 hashes, package versions, seed, feature set, full parameter grids and availability assumptions. Nested series/horizon runs record each fold's selected parameters, training windows and evaluation metrics. The full tuning ledger, performance tables, importance and final forecasts are logged locally. MLflow run ID is in mlflow_run_id.txt. No external tracking server is used.

These are assumption-based evaluations of a November-retrieved, potentially revised snapshot, not verified historical-vintage results. Source-supported lags and the provisional conservative FX rule are inherited unchanged. Availability gating prevents future observation/label use under those rules but cannot undo revision leakage. Aggregate results include correlated and derived series and are not independent replications. Hyperparameter grids and feature sets are deliberately small; model rankings need further independent evidence.

Missing licensed observations, unsupported BLS release rules, unmapped brass, unknown composition and BOM mass inconsistencies remain unresolved. No BOM mappings or monetary weights were fabricated. No BOM cost forecasting or pricing optimization was attempted. All data, MLflow experiments, fitted-model derivatives, reports and executed notebooks remain local under Git-ignored data/phase4/.

Implementation follows training-only preprocessing with pipelines ([scikit-learn documentation](https://scikit-learn.org/stable/common_pitfalls)) and local SQLite experiment tracking ([MLflow documentation](https://www.mlflow.org/docs/latest/ml/tracking/tutorials/local-database/)).
"""
    (out / "phase4_report.md").write_text(report)
    return report


# ---------------------------------------------------------------------------
# Phase 5 report (formerly phase5_reporting.py)
# ---------------------------------------------------------------------------


def build_phase5_report(out):
    """Write Phase 5 calibration, coverage and risk-readiness outputs.

    Args:
        out: Phase 5 output directory containing the experiment CSVs.

    Side effects:
        Writes Markdown/CSV tables and figures into ``out``; simulation and
        forecast artifacts are read-only inputs here.
    """
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
    from schneider_electric_forecasting._05_modeling._03_probabilistic import (
        block_draws,
        generator,
    )

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

Reproduce with scripts/run_notebook.py --phase5. All confidential inputs and outputs remain Git-ignored. GARCH simulation follows [arch's documented recursion](https://arch.readthedocs.io/en/stable/univariate/forecasting.html). Conformal time-series assumptions follow the caution that [exchangeability and distribution shifts affect coverage](https://mapie.readthedocs.io/en/stable/content/exchangeability-testing/).
"""
    (out / "probabilistic_forecasting_report.md").write_text(report)
    return report


def main(argv: list[str] | None = None) -> None:
    """Command-line entry point for the local report builders.

    Args:
        argv: Optional argument list, defaulting to ``sys.argv[1:]``. ``eda``
            rebuilds the data/migration report; ``phase4`` and ``phase5`` read
            the corresponding private output directories and regenerate their
            Markdown/CSV/figures; ``all`` runs the three in order.
    """
    import argparse

    from schneider_electric_forecasting import config

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report",
        choices=("eda", "phase4", "phase5", "all"),
        default="eda",
        help="Which report set to rebuild (default: eda).",
    )
    args = parser.parse_args(argv)
    if args.report in ("eda", "all"):
        build_eda_report()
    if args.report in ("phase4", "all"):
        build_phase4_report(config.DATA_DIR / "phase4")
    if args.report in ("phase5", "all"):
        build_phase5_report(config.DATA_DIR / "phase5")


if __name__ == "__main__":
    main()
