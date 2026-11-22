"""Run the separate LOCF-based probabilistic policy (Phase 5).

Usage:
    uv run --locked python scripts/run_phase5.py

Requires the frozen Phase 4 artifacts and verifies their hashes before and after
the run. Writes predictive distributions, calibration ledgers, policy scores,
the final P10/P50/P90 table, 3,000 joint material/FX paths and reports to
``data/phase5/``. Phase 4 model selection is not retrained or modified.
"""

import hashlib
import json

import numpy as np
import pandas as pd

from schneider_electric_forecasting import _01_data as ingestion
from schneider_electric_forecasting import config
from schneider_electric_forecasting._02_validation import validate_observation_snapshot
from schneider_electric_forecasting._04_features import (
    PHASE3_LAGS,
    TARGETS,
    monthly_history,
    snapshot,
)
from schneider_electric_forecasting.modeling._06_uncertainty import (
    CORE_METHODS,
    METHODS,
    SEED,
    SIMULATIONS,
    arch_gates,
    diagnostics,
    fit_garch,
    generator,
    joint_scenarios,
    known_errors,
    predictive_distribution,
    quantile_scores,
    residual_ledger,
)


def file_hashes(directory):
    """Hash every file below a directory for change detection.

    Args:
        directory: Directory to walk recursively.

    Returns:
        A mapping of relative path string to SHA-256 hex digest.
    """
    return {
        str(path.relative_to(directory)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


def evaluation_tables(predictions):
    """Aggregate probabilistic scores by partition, series, horizon and method.

    Args:
        predictions: Long table of interval forecasts and realised values with
            pinball, coverage, width and scaled scores already computed.

    Returns:
        One row per grouping cell with mean metrics and the number of windows.
    """
    metrics = [
        "pinball_P10",
        "pinball_P50",
        "pinball_P90",
        "coverage_80",
        "calibration_P10",
        "calibration_P50",
        "calibration_P90",
        "width",
        "winkler",
        "scaled_winkler",
        "scaled_width",
        "scaled_pinball",
    ]
    tables = []
    for partition, block in predictions.groupby("partition"):
        table = (
            block.groupby(["series_id", "horizon", "method"])[metrics]
            .mean()
            .reset_index()
        )
        counts = (
            block.groupby(["series_id", "horizon", "method"])
            .size()
            .rename("windows")
            .reset_index()
        )
        table = table.merge(counts, on=["series_id", "horizon", "method"])
        table["partition"] = partition
        tables.append(table)
    return pd.concat(tables, ignore_index=True)


def main():
    """Run the Phase 5 probabilistic policy and joint scenarios.

    Requires the frozen Phase 4 artifacts and verifies their hashes before and
    after the run. Writes distributions, calibration ledgers, policy scores, the
    final P10/P50/P90 table, 3,000 joint material/FX paths and reports under
    ``data/phase5/``. It does not retrain or alter the Phase 4 point policy.

    Returns:
        A dict with the output path, scores, macro table, frozen choices, final
        forecasts and random-walk diagnostics.

    Raises:
        FileNotFoundError: If the required Phase 4 evidence is missing.
        ValueError: If Phase 4 artifacts change during the run or a material/
            horizon cell has no selected method.
    """
    out = config.DATA_DIR / "phase5"
    out.mkdir(parents=True, exist_ok=True)
    # Frozen Phase 4 ledger is hashed, never opened for updates or policy selection.
    phase4_directory = config.DATA_DIR / "phase4"
    required_phase4 = (
        "experiment_config.json",
        "frozen_selection.csv",
        "final_selected_forecasts.csv",
    )
    if any(not (phase4_directory / name).is_file() for name in required_phase4):
        raise FileNotFoundError(
            "Phase 4 evidence is missing. Run Phase 4 or restore its approved local artifacts before Phase 5."
        )
    phase4_before = file_hashes(phase4_directory)
    (out / "phase4_frozen_hashes.json").write_text(json.dumps(phase4_before, indent=2))
    prices = pd.concat(
        [ingestion.load_rm_monthly_long(), ingestion.load_energy_monthly_long()],
        ignore_index=True,
    )
    catalog = pd.concat(
        [ingestion.load_rm_source_catalog(), ingestion.load_energy_source_catalog()],
        ignore_index=True,
    )
    validation = validate_observation_snapshot(prices, catalog)
    if not validation.is_valid:
        raise ValueError(validation.errors)
    catalog.to_csv(out / "source_catalog_used.csv", index=False)
    final_snapshot = snapshot(prices, catalog, "2026-09")
    histories = {
        identifier: monthly_history(final_snapshot, identifier)
        for identifier in TARGETS
    }
    test_results, correlations = diagnostics(histories)
    test_results.to_csv(out / "random_walk_tests.csv", index=False)
    correlations.to_csv(out / "return_acf_pacf.csv", index=False)
    print("Expanded random-walk diagnostics complete", flush=True)
    ledger = residual_ledger(
        prices, catalog, pd.date_range("2018-01-01", "2026-09-01", freq="MS")
    )
    ledger.to_csv(out / "locf_error_ledger.csv", index=False)
    final_labels = final_snapshot.set_index(["series_id", "month"]).value
    origins = (
        pd.date_range("2020-01-01", "2023-12-01", freq="MS")
        .append(pd.date_range("2025-01-01", "2026-03-01", freq="MS"))
        .append(pd.DatetimeIndex(["2026-09-01"]))
    )
    records, audit, gates, volatility = [], [], [], []
    units = catalog.set_index("series_id").unit.to_dict()
    for origin in origins:
        selected = snapshot(prices, catalog, origin)
        current_histories = {
            identifier: monthly_history(selected, identifier) for identifier in TARGETS
        }
        eligibility = arch_gates(current_histories)
        eligibility["origin"] = origin
        gates.append(eligibility)
        partition = (
            "selection"
            if origin.year <= 2021
            else "calibration"
            if origin.year <= 2023
            else "final"
            if origin == pd.Timestamp("2026-09-01")
            else "confirmation"
        )
        for identifier in TARGETS:
            history = current_histories[identifier]
            fit, fit_reason = None, "training-only ARCH gate not met"
            if eligibility.loc[eligibility.series_id == identifier, "eligible"].iloc[0]:
                try:
                    fit = fit_garch(history)
                    fit_reason = "eligible stable GARCH fit"
                except ValueError as error:
                    fit_reason = str(error)
            if partition == "final":
                historical_vol = np.log(history).diff().rolling(12).std() * np.sqrt(12)
                conditional = (
                    fit["volatility"] * np.sqrt(12)
                    if fit is not None
                    else pd.Series(dtype=float)
                )
                for month, value in historical_vol.items():
                    volatility.append(
                        {
                            "series_id": identifier,
                            "month": month,
                            "historical_12m_annualized_log_vol": value,
                            "GARCH_annualized_log_vol": conditional.get(month, np.nan),
                            "GARCH_status": fit_reason,
                        }
                    )
                if fit is not None:
                    variance = (
                        fit["omega"]
                        + fit["alpha"] * fit["last_residual"] ** 2
                        + fit["beta"] * fit["last_variance"]
                    )
                    for step in range(1, 13):
                        volatility.append(
                            {
                                "series_id": identifier,
                                "month": history.index[-1] + pd.DateOffset(months=step),
                                "historical_12m_annualized_log_vol": np.nan,
                                "GARCH_forecast_annualized_log_vol": np.sqrt(variance)
                                / 100
                                * np.sqrt(12),
                                "GARCH_status": fit_reason,
                            }
                        )
                        variance = (
                            fit["omega"] + (fit["alpha"] + fit["beta"]) * variance
                        )
            for horizon in (3, 6, 9):
                target = (pd.Period(origin, freq="M") + horizon).start_time
                steps = (
                    pd.Period(target, freq="M").ordinal
                    - history.index[-1].to_period("M").ordinal
                )
                actual = final_labels.get((identifier, target), np.nan)
                # During method selection, all scored labels must precede the freeze.
                release = (
                    pd.Period(target, freq="M")
                    + PHASE3_LAGS[identifier].maximum_lag_months
                    + 1
                ).start_time
                if partition == "selection" and release > pd.Timestamp("2022-12-31"):
                    raise ValueError("Selection labels unavailable by freeze.")
                if partition != "final" and pd.isna(actual):
                    audit.append(
                        {
                            "series_id": identifier,
                            "horizon": horizon,
                            "origin": origin,
                            "method": "all",
                            "partition": partition,
                            "status": "evaluation label unavailable at final September decision",
                        }
                    )
                    continue
                known = known_errors(ledger, identifier, horizon, origin)
                calibration = known_errors(
                    ledger, identifier, horizon, origin, dedicated_calibration=True
                )
                scale = history.diff().abs().mean()
                for method in METHODS:
                    key = {
                        "series_id": identifier,
                        "horizon": horizon,
                        "origin": origin,
                        "target_month": target,
                        "method": method,
                        "partition": partition,
                    }
                    details = {
                        **key,
                        "error_pool_n": len(known),
                        "calibration_n": len(calibration),
                        "max_error_release": known.available_at_estimated.max(),
                        "max_calibration_release": calibration.available_at_estimated.max(),
                        "history_end": history.index[-1],
                        "garch_status": fit_reason,
                    }
                    try:
                        if method == "GARCH" and fit is None:
                            raise ValueError(fit_reason)
                        values = predictive_distribution(
                            history,
                            steps,
                            method,
                            rng=generator(identifier, str(origin), horizon, method),
                            errors=known.log_error,
                            calibration=calibration.log_error,
                            garch_fit=fit,
                        )
                        record = {
                            **key,
                            **values,
                            "latest_observation": history.iloc[-1],
                            "last_observed_month": history.index[-1],
                            "effective_horizon": steps,
                            "actual": actual,
                            "unit": units[identifier],
                            "scale": scale,
                            "error_pool_n": len(known),
                            "calibration_n": len(calibration),
                            "availability": "assumption_based_revised_snapshot",
                            "LOCF_point_benchmark": history.iloc[-1],
                        }
                        if pd.notna(actual):
                            record.update(quantile_scores(actual, values, scale))
                        records.append(record)
                        details["status"] = "ok"
                    except (ValueError, np.linalg.LinAlgError, OverflowError) as error:
                        details["status"] = str(error)
                    audit.append(details)
        print(
            f"Probabilistic origin complete: {origin:%Y-%m} ({partition})", flush=True
        )
    forecasts = pd.DataFrame(records)
    forecasts.to_csv(out / "all_origin_distributions.csv", index=False)
    pd.DataFrame(audit).to_csv(out / "distribution_audit.csv", index=False)
    pd.concat(gates).to_csv(out / "origin_arch_eligibility.csv", index=False)
    pd.DataFrame(volatility).to_csv(
        out / "historical_conditional_volatility.csv", index=False
    )
    observed = forecasts.loc[forecasts.actual.notna()].copy()
    keys = ["series_id", "horizon", "origin"]
    # Primary five methods share common successful dates. Conformal/GARCH have
    # dedicated availability ledgers and are also compared on matched subsets.
    counts = (
        observed.loc[observed.method.isin(CORE_METHODS)].groupby(keys).method.nunique()
    )
    common = counts.loc[counts == len(CORE_METHODS)].reset_index()[keys]
    primary = observed.loc[observed.method.isin(CORE_METHODS)].merge(common, on=keys)
    scores = evaluation_tables(primary)
    scores.to_csv(out / "primary_performance.csv", index=False)
    full_scores = evaluation_tables(observed)
    full_scores.to_csv(out / "all_available_performance_UNPAIRED.csv", index=False)
    matched = []
    for challenger in ("GARCH", "TS_conformal"):
        eligible_keys = observed.loc[observed.method == challenger, keys]
        comparison = (
            observed.loc[observed.method.isin((*CORE_METHODS, challenger))]
            .merge(eligible_keys, on=keys)
            .merge(common, on=keys)
        )
        if not comparison.empty:
            table = evaluation_tables(comparison)
            table["comparison_set"] = challenger
            matched.append(table)
    if matched:
        pd.concat(matched).to_csv(
            out / "challenger_matched_performance.csv", index=False
        )
    choices = []
    for (identifier, horizon), sample in scores.query(
        "partition == 'selection'"
    ).groupby(["series_id", "horizon"]):
        if sample.windows.min() < 18:
            raise ValueError("At least 18 paired selection windows required.")
        best = sample.sort_values(["scaled_winkler", "method"]).iloc[0]
        choices.append(
            {
                "series_id": identifier,
                "horizon": horizon,
                "selected_method": best.method,
                "selection_scaled_winkler": best.scaled_winkler,
                "selection_windows": int(best.windows),
                "selection_freeze": "2022-12-31",
                "policy": "Phase5_LOCF_probabilistic; separate from Phase4_frozen_point_policy",
            }
        )
    choices = pd.DataFrame(choices)
    if len(choices) != 36:
        raise ValueError("Missing material/horizon method selection.")
    choices.to_csv(out / "frozen_probabilistic_policy.csv", index=False)
    policy = primary.merge(
        choices[["series_id", "horizon", "selected_method"]],
        on=["series_id", "horizon"],
    )
    policy = policy.loc[policy.method == policy.selected_method]
    evaluation_tables(policy).to_csv(out / "frozen_policy_performance.csv", index=False)
    macro = (
        scores.groupby(["partition", "method"])[
            [
                "coverage_80",
                "calibration_P10",
                "calibration_P50",
                "calibration_P90",
                "scaled_winkler",
                "scaled_pinball",
                "scaled_width",
            ]
        ]
        .mean()
        .reset_index()
    )
    macro.to_csv(out / "scale_independent_macro.csv", index=False)
    final = forecasts.query("partition == 'final'").merge(
        choices[["series_id", "horizon", "selected_method"]],
        on=["series_id", "horizon"],
    )
    final["selected"] = final.method == final.selected_method
    final.to_csv(out / "final_all_method_forecasts.csv", index=False)
    final.loc[final.selected].to_csv(
        out / "final_selected_probabilistic_forecasts.csv", index=False
    )
    scenarios, historical_correlation, simulated_correlation = joint_scenarios(
        final_snapshot
    )
    scenarios["unit"] = scenarios.series_id.map(units)
    scenarios.to_csv(out / "joint_material_fx_scenarios.csv", index=False)
    historical_correlation.to_csv(out / "joint_historical_innovation_correlation.csv")
    simulated_correlation.to_csv(out / "joint_simulated_innovation_correlation.csv")
    manifest = {
        "seed": SEED,
        "simulations": SIMULATIONS,
        "methods": METHODS,
        "core_selection_methods": CORE_METHODS,
        "selection": "2020-2021 forecast origins; freeze December 31 2022 after all labels released",
        "calibration": "2022-2023 forecast origins only, fixed signed LOCF log errors; minimum 24",
        "confirmation": "monthly origins January 2025-March 2026 limited to labels available September 30 2026",
        "final_origin": "2026-09-30",
        "conformal_claim": "two one-sided finite ranks; no verified time-series coverage guarantee",
        "garch_gate": "at least 60 returns; per-origin ARCH-LM(3), BH across series q<0.1; optimizer convergence, persistence<0.995",
        "innovation_mean": "Gaussian and innovation/block/GARCH simulations zero-mean log innovations; empirical horizon errors and historical simulation retain sample shifts/asymmetry",
        "block_length": 3,
        "selection_loss": "training-scaled Winkler interval score on paired selection origins",
        "availability": {key: vars(rule) for key, rule in PHASE3_LAGS.items()},
        "joint": "uncentered common vector blocks; common June anchor, extra July FX observation discarded; derived EUR prices pathwise USD/FX; not selected-policy marginal reconciliation",
        "packages": {
            name: __import__(name).__version__
            for name in ("numpy", "pandas", "statsmodels", "arch")
        },
    }
    (out / "experiment_config.json").write_text(
        json.dumps(manifest, indent=2, default=str)
    )
    if file_hashes(config.DATA_DIR / "phase4") != phase4_before:
        raise ValueError("Frozen Phase 4 artifacts changed unexpectedly.")
    from build_reports import build_phase5_report

    build_phase5_report(out)
    print(f"Phase 5 complete: {out}", flush=True)
    return {
        "output": out,
        "scores": scores,
        "macro": macro,
        "choices": choices,
        "final": final,
        "diagnostics": test_results,
    }


if __name__ == "__main__":
    main()
