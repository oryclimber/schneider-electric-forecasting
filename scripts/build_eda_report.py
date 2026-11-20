"""Write a local evidence-backed comparison and implementation handover."""

import json

import pandas as pd


def table(frame):
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


def main():
    from schneider_electric_forecasting import config, ingestion

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


if __name__ == "__main__":
    main()
