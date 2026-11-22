"""Run the nested Phase 4 experiment with local-only MLflow tracking.

Usage:
    uv run --locked python scripts/run_phase4.py

Reads the authorized workbooks, evaluates ARIMA, ETS, Ridge, Elastic Net,
XGBoost and LightGBM under nested rolling validation, freezes the selected
family before confirmation data, and writes predictions, scores, tuning audits,
figures and a local MLflow SQLite store under ``data/phase4/``. No external
tracking server is contacted.
"""

import hashlib
import json
import os

os.environ.setdefault("MLFLOW_ENABLE_ASYNC_LOGGING", "false")
os.environ.setdefault("DO_NOT_TRACK", "true")

import mlflow
import numpy as np
import pandas as pd

from schneider_electric_forecasting import _01_ingestion as ingestion
from schneider_electric_forecasting import config
from schneider_electric_forecasting._02_validation import validate_observation_snapshot
from schneider_electric_forecasting._04_features import (
    PHASE3_LAGS,
    TARGETS,
    snapshot,
    supervised_dataset,
)
from schneider_electric_forecasting._05_modeling._01_baselines import (
    baseline_forecasts,
    score_predictions,
)
from schneider_electric_forecasting._05_modeling._02_advanced import (
    CANDIDATES,
    FEATURES,
    SEED,
    SELECTION_FREEZE,
    NestedForecaster,
    estimator,
    frozen_selection,
)


def performance(predictions):
    """Score predictions per partition and add LOCF-relative MAE gains.

    Args:
        predictions: Outer predictions with ``partition``, ``series_id``,
            ``horizon``, ``model``, ``error`` and ``mase_scale``.

    Returns:
        One row per partition/series/horizon/model with the standard metrics and
        ``MAE_improvement_pct`` relative to that cell's LOCF MAE.
    """
    results = []
    for partition, sample in predictions.groupby("partition"):
        table = score_predictions(sample)
        baseline = table.loc[
            table.model == "LOCF", ["series_id", "horizon", "MAE"]
        ].rename(columns={"MAE": "LOCF_MAE"})
        table = table.merge(
            baseline, on=["series_id", "horizon"], validate="many_to_one"
        )
        table["MAE_improvement_pct"] = (1 - table.MAE / table.LOCF_MAE) * 100
        table["partition"] = partition
        results.append(table)
    return pd.concat(results, ignore_index=True)


def main():
    """Run the nested Phase 4 experiment and log it to local MLflow.

    The experiment selects model families on 2021-2022 outer folds, freezes the
    choice before 2024, then evaluates confirmation folds. Predictions, tuning
    audits, scores, feature importances, final forecasts and the MLflow SQLite
    store are written under ``data/phase4/``. Phase 3 artifacts are not modified.

    Returns:
        A dict with the output path, scores, aggregate table, frozen selection
        and final forecasts.
    """
    out = config.DATA_DIR / "phase4"
    out.mkdir(exist_ok=True, parents=True)
    observations = pd.concat(
        [ingestion.load_rm_monthly_long(), ingestion.load_energy_monthly_long()],
        ignore_index=True,
    )
    catalog = pd.concat(
        [ingestion.load_rm_source_catalog(), ingestion.load_energy_source_catalog()],
        ignore_index=True,
    )
    catalog.to_csv(out / "source_catalog_used.csv", index=False)
    quality = validate_observation_snapshot(observations, catalog)
    if not quality.is_valid:
        raise ValueError(quality.errors)
    manifest = {
        "seed": SEED,
        "features": FEATURES,
        "candidates": CANDIDATES,
        "selection_origins": "quarterly 2021-01 through 2022-10",
        "selection_freeze": str(SELECTION_FREEZE),
        "confirmation_origins": "quarterly 2024-01 through 2026-01, further limited by available labels",
        "inner_folds": "last and fourth-last labels available at outer origin; training labels gated at inner origin",
        "minimum_inner_labels": 12,
        "minimum_outer_labels": 18,
        "availability": {key: vars(rule) for key, rule in PHASE3_LAGS.items()},
        "source_sha256": {
            key: hashlib.sha256(
                config.CANONICAL_DATASET_PATHS[key].read_bytes()
            ).hexdigest()
            for key in ("authoritative_bom", "rm_price_workbook", "energy_workbook")
        },
        "supervised_grids": {
            family: [
                estimator(family, option).steps[-1][1].get_params() for option in (0, 1)
            ]
            for family in ("Ridge", "ElasticNet", "XGBoost", "LightGBM")
        },
        "univariate_grids": {
            "ARIMA_level": [[1, 1, 0], [0, 1, 1]],
            "ARIMA_log_return": [[1, 0, 0], [0, 0, 1]],
            "ETS": ["no trend", "damped additive trend"],
        },
        "formulations": "levels in source units; cumulative log returns from last observable price, back-transformed without a smearing correction",
        "selection_rule": ">=5% selection MAE gain, >=60% origin wins, positive gains in both 2021 and 2022; >=6 paired selection folds; otherwise LOCF",
        "runtime_versions": {
            name: __import__(name).__version__
            for name in (
                "numpy",
                "pandas",
                "sklearn",
                "statsmodels",
                "xgboost",
                "lightgbm",
                "mlflow",
            )
        },
    }
    (out / "experiment_config.json").write_text(
        json.dumps(manifest, indent=2, default=str)
    )
    print("Building features using existing origin-specific pipeline", flush=True)
    dataset = supervised_dataset(
        observations, catalog, pd.date_range("2018-01-01", "2026-09-01", freq="MS")
    )
    dataset.to_csv(out / "supervised_dataset.csv", index=False)
    engine = NestedForecaster(observations, catalog)
    final_snapshot = snapshot(observations, catalog, "2026-09")
    labels = final_snapshot.set_index(["series_id", "month"]).value
    origins = pd.date_range("2021-01-01", "2022-12-01", freq="3MS").append(
        pd.date_range("2024-01-01", "2026-03-01", freq="3MS")
    )
    rows, coverage, failures = [], [], []
    for identifier in TARGETS:
        for horizon in (3, 6, 9):
            data = dataset.loc[
                (dataset.series_id == identifier) & (dataset.horizon == horizon)
            ].copy()
            for origin in origins:
                row = data.loc[data.origin == origin]
                target = row.target_month.iloc[0]
                actual = labels.get((identifier, target), np.nan)
                if pd.isna(actual):
                    coverage.append(
                        {
                            "series_id": identifier,
                            "horizon": horizon,
                            "origin": origin,
                            "model": "all",
                            "success": False,
                            "paired": False,
                            "reason": "Label unavailable at final decision",
                        }
                    )
                    continue
                release = row.target_available_at_estimated.iloc[0]
                partition = "selection" if origin.year <= 2022 else "confirmation"
                if partition == "selection" and release > SELECTION_FREEZE:
                    raise ValueError("Selection label not available before freeze.")
                history = engine.history(identifier, origin)
                _, scale, _ = baseline_forecasts(history, target)
                fold = []
                for candidate in CANDIDATES:
                    try:
                        prediction, settings = engine.tune_predict(
                            identifier, data, row, candidate
                        )
                        fold.append(
                            {
                                "series_id": identifier,
                                "horizon": horizon,
                                "origin": origin,
                                "target_month": target,
                                "actual": actual,
                                "model": candidate,
                                "prediction": prediction,
                                "error": prediction - actual,
                                "mase_scale": scale,
                                "partition": partition,
                                "settings": json.dumps(settings),
                                "target_label_release": release,
                                "last_observed_month": history.index[-1],
                            }
                        )
                        success, reason = True, "ok"
                    except (ValueError, np.linalg.LinAlgError, OverflowError) as error:
                        success, reason = False, str(error)
                        failures.append(
                            {
                                "series_id": identifier,
                                "horizon": horizon,
                                "origin": origin,
                                "model": candidate,
                                "reason": reason,
                            }
                        )
                    coverage.append(
                        {
                            "series_id": identifier,
                            "horizon": horizon,
                            "origin": origin,
                            "model": candidate,
                            "success": success,
                            "paired": False,
                            "reason": reason,
                        }
                    )
                if len(fold) == len(CANDIDATES) and np.isfinite(scale) and scale > 0:
                    rows.extend(fold)
                    for record in coverage[-len(CANDIDATES) :]:
                        record["paired"] = True
            print(
                f"Nested evaluation finished: {identifier} horizon {horizon}",
                flush=True,
            )
    predictions = pd.DataFrame(rows)
    if predictions.empty:
        raise ValueError("No joint evaluation windows. Inspect numerical failures.")
    predictions.to_csv(out / "outer_predictions.csv", index=False)
    pd.DataFrame(coverage).to_csv(out / "coverage.csv", index=False)
    pd.DataFrame(
        failures, columns=["series_id", "horizon", "origin", "model", "reason"]
    ).to_csv(out / "fit_failures.csv", index=False)
    selection = frozen_selection(predictions)
    scores = performance(predictions)
    # Macro averages give equal weight to each material/horizon; no pooled raw MAE.
    aggregate = (
        scores.groupby(["partition", "model"])
        .agg(
            macro_MASE=("MASE", "mean"),
            macro_MAE_improvement_pct=("MAE_improvement_pct", "mean"),
            series_horizon_cells=("series_id", "size"),
            min_windows=("valid_windows", "min"),
            max_windows=("valid_windows", "max"),
        )
        .reset_index()
    )
    scores.to_csv(out / "model_performance.csv", index=False)
    aggregate.to_csv(out / "scale_independent_aggregate.csv", index=False)
    selection.to_csv(out / "frozen_selection.csv", index=False)
    # Regimes are fixed calendar blocks, not inferred from test prices.
    regimes = predictions.copy()
    regimes["regime"] = regimes.origin.dt.year.map(
        {
            2021: "2021_recovery",
            2022: "2022_energy_shock",
            2024: "2024",
            2025: "2025",
            2026: "2026_partial",
        }
    )
    regime_scores = []
    for regime, sample in regimes.groupby("regime"):
        block = performance(sample)
        block["regime"] = regime
        regime_scores.append(block)
    pd.concat(regime_scores).to_csv(out / "regime_performance.csv", index=False)
    paired_baseline = predictions.query("model == 'LOCF'")[
        ["series_id", "horizon", "origin", "error"]
    ].rename(columns={"error": "LOCF_error"})
    consistency = predictions.merge(
        paired_baseline, on=["series_id", "horizon", "origin"]
    )
    consistency["beats_LOCF"] = consistency.error.abs() < consistency.LOCF_error.abs()
    consistency.groupby(["partition", "series_id", "horizon", "model"]).beats_LOCF.agg(
        ["mean", "count"]
    ).to_csv(out / "origin_win_rates.csv")
    selected_holdout = predictions.loc[predictions.partition == "confirmation"].merge(
        selection[["series_id", "horizon", "selected_model"]],
        on=["series_id", "horizon"],
    )
    selected_holdout = selected_holdout.loc[
        selected_holdout.model == selected_holdout.selected_model
    ]
    selected_holdout.to_csv(out / "selected_policy_confirmation.csv", index=False)
    final_rows, importance = [], []
    for identifier in TARGETS:
        for horizon in (3, 6, 9):
            data = dataset.loc[
                (dataset.series_id == identifier) & (dataset.horizon == horizon)
            ]
            row = data.loc[data.origin == pd.Timestamp("2026-09-01")]
            choice = selection.loc[
                (selection.series_id == identifier) & (selection.horizon == horizon),
                "selected_model",
            ].iloc[0]
            for candidate in CANDIDATES:
                try:
                    forecast, settings = engine.tune_predict(
                        identifier, data, row, candidate, importance=importance
                    )
                    final_rows.append(
                        {
                            "series_id": identifier,
                            "horizon": horizon,
                            "model": candidate,
                            "prediction": forecast,
                            "selected": candidate == choice,
                            "origin": "2026-09",
                            "decision_date": "2026-09-30",
                            "target_month": row.target_month.iloc[0],
                            "unit": catalog.set_index("series_id").loc[
                                identifier, "unit"
                            ],
                            "availability_mode": "assumption_based_revised_snapshot",
                            "settings": json.dumps(settings),
                        }
                    )
                except (ValueError, np.linalg.LinAlgError, OverflowError) as error:
                    raise ValueError(
                        f"Final fit failed for {identifier}/{horizon}/{candidate}: {error}"
                    ) from error
    final = pd.DataFrame(final_rows)
    final.to_csv(out / "final_candidate_forecasts.csv", index=False)
    final.loc[final.selected].to_csv(out / "final_selected_forecasts.csv", index=False)
    pd.DataFrame(importance).to_csv(out / "tree_feature_importance.csv", index=False)
    pd.DataFrame(engine.audit).to_csv(out / "nested_tuning_audit.csv", index=False)
    # Local SQLite and local artifacts only; no external tracking server.
    mlflow.set_tracking_uri(f"sqlite:///{out / 'mlflow.db'}")
    experiment = mlflow.get_experiment_by_name("Schneider_Phase4")
    experiment_id = (
        experiment.experiment_id
        if experiment
        else mlflow.create_experiment(
            "Schneider_Phase4",
            artifact_location=(out / "mlflow_artifacts").resolve().as_uri(),
        )
    )
    with mlflow.start_run(
        experiment_id=experiment_id, run_name="nested_rolling_phase4"
    ) as run:
        mlflow.log_params(
            {
                "seed": SEED,
                "feature_count": len(FEATURES),
                "selection_freeze": str(SELECTION_FREEZE),
                "availability": "assumption_based",
                "outer_cadence_months": 3,
                "inner_folds": 2,
            }
        )
        mlflow.log_artifact(str(out / "experiment_config.json"))
        for (identifier, horizon), block in predictions.groupby(
            ["series_id", "horizon"]
        ):
            with mlflow.start_run(
                experiment_id=experiment_id,
                nested=True,
                run_name=f"{identifier}_h{horizon}",
            ):
                mlflow.log_params(
                    {
                        "series_id": identifier,
                        "horizon": horizon,
                        "seed": SEED,
                        "feature_set": "compact_economic_24",
                        "selection_freeze": str(SELECTION_FREEZE),
                    }
                )
                mlflow.log_dict(
                    json.loads(block.to_json(orient="records", date_format="iso")),
                    "fold_predictions_and_hyperparameters.json",
                )
                for record in scores.loc[
                    (scores.series_id == identifier) & (scores.horizon == horizon)
                ].itertuples():
                    for metric in (
                        "MAE",
                        "RMSE",
                        "MASE",
                        "bias",
                        "MAE_improvement_pct",
                        "valid_windows",
                    ):
                        mlflow.log_metric(
                            f"{record.partition}.{record.model}.{metric}",
                            float(getattr(record, metric)),
                        )
        for path in out.glob("*.csv"):
            mlflow.log_artifact(str(path))
        (out / "mlflow_run_id.txt").write_text(run.info.run_id + "\n")
    from build_reports import build_phase4_report

    build_phase4_report(out)
    print(f"Phase 4 complete: {out}", flush=True)
    return {
        "output": out,
        "scores": scores,
        "aggregate": aggregate,
        "selection": selection,
        "final": final,
    }


if __name__ == "__main__":
    main()
