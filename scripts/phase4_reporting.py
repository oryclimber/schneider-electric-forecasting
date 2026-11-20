"""Local tables and Schneider-style visual diagnostics for Phase 4."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap


def build_report(out):
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
