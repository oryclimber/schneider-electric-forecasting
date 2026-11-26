"""Run the Phase 3 feature build and paired baseline experiment.

Usage:
    uv run --locked python scripts/cli.py phase 3

Reads the authorized RM and energy workbooks and writes features, rolling
predictions, baseline scores, exclusions, final September forecasts, figures and
a Markdown report to ``data/phase3/``. LOCF is the mandatory comparator; no
advanced model is fitted and no model is selected from evaluation errors.
"""

import json

import matplotlib.pyplot as plt
import pandas as pd

from schneider_electric_forecasting import _01_ingestion as ingestion
from schneider_electric_forecasting import config
from schneider_electric_forecasting._02_validation import validate_observation_snapshot
from schneider_electric_forecasting._04_features import (
    PHASE3_LAGS,
    TARGETS,
    supervised_dataset,
    training_rows,
)
from schneider_electric_forecasting._05_modeling._01_baselines import (
    final_forecasts,
    rolling_backtest,
)


def main():
    """Run the Phase 3 feature build and paired baseline evaluation.

    Reads the authorized RM and energy workbooks, writes supervised features,
    rolling predictions, scores, exclusions, final September forecasts, figures
    and a Markdown report under ``data/phase3/``. No advanced model is trained
    and no candidate is selected from test errors.

    Returns:
        A dict with the dataset, scores, forecasts and recommendations used by
        the source notebook.
    """
    out = config.DATA_DIR / "phase3"
    out.mkdir(parents=True, exist_ok=True)
    rm, energy = ingestion.load_rm_monthly_long(), ingestion.load_energy_monthly_long()
    rm_catalog, energy_catalog = (
        ingestion.load_rm_source_catalog(),
        ingestion.load_energy_source_catalog(),
    )
    for sample, catalog in ((rm, rm_catalog), (energy, energy_catalog)):
        report = validate_observation_snapshot(sample, catalog)
        if not report.is_valid:
            raise ValueError(report.errors)
    observations = pd.concat([rm, energy], ignore_index=True)
    catalog = pd.concat([rm_catalog, energy_catalog], ignore_index=True)
    # Candidate origin grid is fixed in advance. Minimum-history eligibility is
    # assessed separately at every origin, before looking at test errors.
    origins = pd.date_range("2019-01-01", "2026-09-01", freq="MS")
    print("Building origin-specific features", flush=True)
    dataset = supervised_dataset(observations, catalog, origins)
    dataset.to_csv(out / "supervised_features_and_labels.csv", index=False)
    latest = dataset.loc[dataset.origin == pd.Timestamp("2026-09-01")]
    latest.to_csv(out / "final_origin_features.csv", index=False)
    training_rows(dataset, config.FORECAST_DECISION_DATE).to_csv(
        out / "training_labels_available_at_final_origin.csv", index=False
    )
    print("Evaluating paired baseline windows", flush=True)
    predictions, scores, exclusions = rolling_backtest(observations, catalog, origins)
    forecasts = final_forecasts(observations, catalog)
    units = rm_catalog.set_index("series_id").unit
    for frame in (predictions, scores, forecasts):
        frame["unit"] = frame.series_id.map(units)
    predictions.to_csv(out / "backtest_predictions.csv", index=False)
    scores.to_csv(out / "baseline_performance.csv", index=False)
    exclusions.to_csv(out / "excluded_windows.csv", index=False)
    forecasts.to_csv(out / "september_2026_forecasts.csv", index=False)
    manifest = {
        "decision_date": str(config.FORECAST_DECISION_DATE),
        "availability_mode": "assumption_based_snapshot_not_verified_vintages",
        "minimum_history": 36,
        "mase_seasonality": 1,
        "bias_definition": "prediction minus actual",
        "targets": TARGETS,
        "lags": {key: vars(value) for key, value in PHASE3_LAGS.items()},
        "feature_definition": "lags relative to last observable month; rolling windows include latest observation; no imputation/scaling/feature selection fitted",
        "evaluation_labels": "only target observations estimated available at final September decision",
        "excluded_targets": rm_catalog.loc[
            ~rm_catalog.series_id.isin(TARGETS), ["series_id", "access", "proxy_type"]
        ].to_dict("records"),
    }
    (out / "experiment_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str)
    )
    feature_names = [
        column
        for column in dataset
        if column not in ("target_value", "target_available_at_estimated")
    ]
    pd.DataFrame(
        {
            "column": dataset.columns,
            "role": [
                "label_metadata_never_model_input"
                if c not in feature_names
                else "feature_or_origin_metadata"
                for c in dataset
            ],
        }
    ).to_csv(out / "feature_dictionary.csv", index=False)
    colors = {
        "LOCF": "#3DCD58",
        "Seasonal Naive": "#86B2C6",
        "Drift": "#D09380",
        "Historical mean return": "#176B45",
    }
    plt.rcParams.update(
        {
            "figure.facecolor": "#F4F1E8",
            "axes.facecolor": "#F4F1E8",
            "text.color": "#0C3027",
            "axes.labelcolor": "#0C3027",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    names = rm_catalog.set_index("series_id").material
    for identifier in TARGETS:
        figure, axes = plt.subplots(3, 1, figsize=(12, 10))
        for axis, horizon in zip(axes, (3, 6, 9), strict=True):
            sample = predictions.query(
                "series_id == @identifier and horizon == @horizon"
            )
            actual = sample.drop_duplicates("target_month").sort_values("target_month")
            axis.plot(
                actual.target_month,
                actual.actual,
                color="#0C3027",
                linewidth=2,
                label="Actual",
            )
            for model, group in sample.groupby("model"):
                axis.plot(
                    group.target_month,
                    group.prediction,
                    color=colors[model],
                    label=model,
                    alpha=0.85,
                )
            axis.set(
                title=f"{horizon}-month horizon · {len(actual)} paired windows",
                ylabel=units[identifier],
            )
        axes[0].legend(ncols=3)
        figure.suptitle(
            f"{names[identifier]} ({identifier}): rolling forecasts and actuals\nEstimated availability; expanding history; retrospective revised snapshot"
        )
        figure.tight_layout()
        for extension in ("png", "svg"):
            figure.savefig(
                out / f"forecast_actual_{identifier}.{extension}",
                dpi=300,
                bbox_inches="tight",
            )
        plt.close(figure)
    recommendations = []
    for (identifier, horizon), group in scores.groupby(["series_id", "horizon"]):
        winner = group.sort_values(["MAE", "model"]).iloc[0]
        locf = group.loc[group.model == "LOCF"].iloc[0]
        recommendations.append(
            {
                "series_id": identifier,
                "material": names[identifier],
                "horizon": horizon,
                "lowest_MAE_baseline": winner.model,
                "LOCF_MAE": locf.MAE,
                "lowest_MAE": winner.MAE,
                "relative_MAE_improvement_pct": (1 - winner.MAE / locf.MAE) * 100,
                "valid_windows": winner.valid_windows,
                "recommendation": "Retain LOCF as mandatory comparator. Validate any apparent winner on fresh rolling origins; overlapping errors and revised vintages limit selection confidence. "
                + (
                    "Test FX-aware commodity returns and energy costs."
                    if identifier in ("S01", "S02", "D01", "D02")
                    else "Keep cost drivers distinct from grade prices; test break-aware lagged return features."
                ),
            }
        )
    recommendations = pd.DataFrame(recommendations)
    recommendations.to_csv(out / "material_recommendations.csv", index=False)
    recommendation_table = (
        "| Series | Horizon | Windows | Lowest MAE baseline | LOCF MAE | Lowest MAE | Improvement % |\n|---|---:|---:|---|---:|---:|---:|\n"
        + "\n".join(
            f"| {row.series_id} | {row.horizon} | {row.valid_windows} | {row.lowest_MAE_baseline} | {row.LOCF_MAE:.3f} | {row.lowest_MAE:.3f} | {row.relative_MAE_improvement_pct:.2f} |"
            for row in recommendations.itertuples()
        )
    )
    wins = recommendations.lowest_MAE_baseline.value_counts().to_string()
    (
        out / "phase3_report.md"
    ).write_text(f"""# Phase 3: features and baseline forecasting

Executed {len(dataset):,} supervised origin/series/horizon rows, {len(scores)} performance rows and {len(forecasts)} final forecasts. Twelve eligible RM/derived series, three horizons, four baselines. No advanced models trained.

## Experiment and availability

September 30, 2026 is the final decision date. Targets are December 2026, March 2027 and June 2027. Each rolling origin first invokes the existing availability selector. IMF/PPI observations are delayed conservatively; FX uses a separately identified provisional one-month-after-month-end scenario. EUR prices inherit the maximum input delay. June 2026 is the last eligible final RM month, so the effective forecast distances are 6, 9 and 12 months. Energy import quantities have an earlier April cutoff; FX ends July. S12/S13 lack supported release rules and licensed series lack observations; none are fabricated. Strict mode remains fail-closed without verified availability timestamps.

Features include latest price, 1/2/3/6/12-month price and return lags, 3/6/12-month means, sample standard deviations, annualized return volatility, momentum, origin/target seasonal sine/cosine, and predeclared energy, FX and cost-driver features. Cross-material return spreads compare nickel/steel, ore/steel and zinc/galvanised proxies at a common eligible month. Lags are relative to the last observation; cutoff and age fields document staleness. Calendar gaps remain null. There is no full-sample scaling, imputation, fitted feature selection or advanced estimator.

Supervised targets are isolated from feature columns. training_rows filters labels by their own estimated publication timestamps at each training decision. Historical feature rows can exist before their labels are known; callers must apply that gate before fitting. No future exogenous values are used.

## Baseline protocol

LOCF is mandatory. Seasonal naive repeats the latest observable target-calendar-month value, stepping back another year if necessary. Drift uses first-to-last level slope per elapsed calendar month. Historical mean return compounds the expanding arithmetic mean of consecutive monthly returns across the full last-observation-to-target gap. Minimum history is 36 actual observations. Predictions are not clipped.

All four models share exactly the same eligible evaluation keys per series/horizon. Missing labels, missing seasonal observations and undefined MASE denominators exclude the window for all models. MASE divides each error by that origin's training-only mean absolute one-month naive difference (m=1), then averages. Bias is forecast minus actual. Cross-material level errors have different units and must not be pooled. The exclusion ledger and valid-window counts are exported.

Evaluation uses only actual labels estimated available by the final September decision, rather than incorporating later realized outcomes into baseline selection. Forecasts for final target months are unscored and separate. Overlapping forecast errors are dependent; winner counts are descriptive, not significance evidence.

## Descriptive baseline winners by lowest MAE

```
{wins}
```

See baseline_performance.csv for all MAE, RMSE, MASE, bias and window counts; material_recommendations.csv for every material/horizon, LOCF comparisons and recommendations. Forecast-versus-actual plots show each material in its original units. No claim is made that complex baselines necessarily improve accuracy.

{recommendation_table}

MAE units are original source units, including index points for PPIs. MASE compares multistep errors with a one-step training scale; MASE greater than one alone does not show failure against the same-horizon LOCF benchmark. Seasonal naive ties LOCF at the nine-month requested horizon because the full delay-adjusted distance is twelve months. Tie winners are reported deterministically, not as evidence of superiority.

## Readiness and limitations

Ready for advanced-model experimentation with nested rolling validation, training-only preprocessing, feature ablations, a held-out later period and mandatory LOCF comparisons. Not ready for verified point-in-time performance claims: source observations were retrieved in November and may include revisions; lag scenarios cannot undo revision leakage. FX timing is provisional. Generic PPI and commodity drivers do not establish grade-specific purchasing prices. No synthetic targets, missing BOM mappings or monetary cost weights have been created. Brass, unknown composition and BOM mass inconsistencies remain unresolved. Do not convert these univariate forecasts into complete product-cost or pricing estimates without resolving those inputs.

The candidate feature set is large relative to the number of evaluation origins. Reduce correlated/redundant candidates inside training folds, compare compact economically defined subsets, and use regularization rather than fitting an unrestricted model to all features. The material-specific historical winners are exploratory recommendations, not selection on an untouched holdout.

All generated data, figures, forecasts and executed notebooks are local and Git-ignored under data/phase3/. Source notebook and Python modules contain no cached confidential outputs.
""")
    print(out, flush=True)
    return {
        "dataset": dataset,
        "scores": scores,
        "forecasts": forecasts,
        "recommendations": recommendations,
        "output": out,
    }


if __name__ == "__main__":
    main()
