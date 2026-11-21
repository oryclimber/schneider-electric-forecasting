"""Quantile scoring, temporal calibration gates and scenario identities."""

import numpy as np
import pandas as pd
import pytest
from arch.unitroot import VarianceRatio

from schneider_electric_forecasting.features import TARGETS
from schneider_electric_forecasting.probabilistic import (
    CORE_METHODS,
    arch_gates,
    block_draws,
    conformal_quantiles,
    fit_garch,
    garch_draws,
    generator,
    joint_scenarios,
    known_errors,
    predictive_distribution,
    quantile_scores,
)
from schneider_electric_forecasting.random_walk import diagnostics


@pytest.fixture
def history():
    rng = np.random.default_rng(123)
    return pd.Series(
        100 * np.exp(np.cumsum(rng.normal(0.002, 0.03, 119))),
        index=pd.date_range("2016-11-01", periods=119, freq="MS"),
    )


def test_correct_pinball_interval_and_winkler_scores():
    values = {"P10": 90.0, "P50": 100.0, "P90": 110.0}
    result = quantile_scores(120, values, 2)
    assert result["pinball_P10"] == pytest.approx(3)
    assert result["pinball_P50"] == pytest.approx(10)
    assert result["pinball_P90"] == pytest.approx(9)
    assert result["winkler"] == 120
    assert result["scaled_winkler"] == 60
    assert result["coverage_80"] == 0
    assert quantile_scores(100, values, 2)["coverage_80"] == 1
    with pytest.raises(ValueError):
        quantile_scores(100, {"P10": 110, "P50": 100, "P90": 90}, 2)


def test_conformal_finite_ranks_asymmetric_median_and_no_invented_mean():
    errors = np.linspace(0.02, 0.25, 24)
    result = conformal_quantiles(errors, 100)
    assert result["P10"] == pytest.approx(100 * np.exp(errors[1]))
    assert result["P90"] == pytest.approx(100 * np.exp(errors[22]))
    assert result["P50"] > 100
    assert pd.isna(result["expected_price"])
    with pytest.raises(ValueError, match="Insufficient"):
        conformal_quantiles(errors[:12], 100)


def test_error_pool_release_gate_calibration_split_and_no_confirmation_reuse():
    ledger = pd.DataFrame(
        {
            "series_id": ["S01"] * 4,
            "horizon": [9] * 4,
            "origin": pd.to_datetime(
                ["2021-01-01", "2022-01-01", "2023-12-01", "2025-01-01"]
            ),
            "available_at_estimated": pd.to_datetime(
                ["2022-01-01", "2023-01-01", "2024-12-01", "2026-01-01"]
            ),
            "log_error": [0.1, 0.2, 0.3, 100.0],
        }
    )
    pool = known_errors(ledger, "S01", 9, "2023-06")
    assert pool.log_error.tolist() == [0.1, 0.2]
    calibration = known_errors(ledger, "S01", 9, "2026-09", dedicated_calibration=True)
    assert calibration.log_error.tolist() == [0.2, 0.3]
    assert 100 not in known_errors(ledger, "S01", 9, "2026-09").log_error.tolist()
    ledger.loc[2:, "log_error"] = 1e9
    assert known_errors(ledger, "S01", 9, "2023-06").log_error.tolist() == [0.1, 0.2]


def test_all_core_methods_reproducible_ordered_positive_and_gap_sensitive(history):
    for method in CORE_METHODS:
        first = predictive_distribution(
            history,
            6,
            method,
            rng=generator(method),
            errors=np.linspace(-0.2, 0.3, 36),
            simulations=1000,
        )
        second = predictive_distribution(
            history,
            6,
            method,
            rng=generator(method),
            errors=np.linspace(-0.2, 0.3, 36),
            simulations=1000,
        )
        assert first == second
        assert 0 < first["P10"] <= first["P50"] <= first["P90"]
        assert first["expected_price"] > 0
    short = predictive_distribution(
        history, 6, "Gaussian_RW", rng=generator("gaussian"), simulations=10000
    )
    long = predictive_distribution(
        history, 12, "Gaussian_RW", rng=generator("gaussian"), simulations=10000
    )
    assert long["P90"] - long["P10"] > short["P90"] - short["P10"]
    assert long["expected_price"] > history.iloc[-1]


def test_missing_periods_not_imputed_and_unsupported_garch_not_fabricated(history):
    with pytest.raises(ValueError, match="no imputation"):
        predictive_distribution(
            history.drop(history.index[30]), 6, "Gaussian_RW", rng=generator(1)
        )
    with pytest.raises(ValueError, match="GARCH"):
        predictive_distribution(history, 6, "GARCH", rng=generator(1))
    with pytest.raises(ValueError, match="24"):
        predictive_distribution(history.iloc[:12], 6, "Gaussian_RW", rng=generator(1))


def test_shared_block_rows_preserve_dependencies():
    first = np.arange(100.0)
    vector = np.column_stack([first, 2 * first, -first])
    draws = block_draws(vector, 30, 12, generator("blocks"))
    np.testing.assert_allclose(draws[:, :, 1], 2 * draws[:, :, 0])
    np.testing.assert_allclose(draws[:, :, 2], -draws[:, :, 0])
    np.testing.assert_allclose(np.diff(draws[:, :3, 0], axis=1), 1)


def test_joint_currency_identity_and_common_anchor(history):
    data = []
    for position, identifier in enumerate(
        [i for i in TARGETS if not i.startswith("D")] + ["S06"]
    ):
        data.append(
            pd.DataFrame(
                {
                    "series_id": identifier,
                    "month": history.index,
                    "value": history.to_numpy() * (position + 1),
                }
            )
        )
    scenarios, historical, simulated = joint_scenarios(pd.concat(data), simulations=50)
    panel = scenarios.pivot(
        index=["scenario", "horizon"], columns="series_id", values="price"
    )
    np.testing.assert_allclose(panel.D01, panel.S01 / panel.S06)
    np.testing.assert_allclose(panel.D02, panel.S02 / panel.S06)
    assert scenarios.anchor_month.nunique() == 1
    np.testing.assert_allclose(historical, simulated, atol=1e-10)


def test_robust_variance_ratio_matches_validated_library_and_multiplicity(history):
    results, correlations = diagnostics({"S01": history})
    expected = VarianceRatio(
        np.log(history), lags=6, robust=True, overlap=True, debiased=True, trend="c"
    )
    row = results.loc[results.test == "VR_6"].iloc[0]
    assert row.statistic == pytest.approx(expected.stat)
    assert row.pvalue == pytest.approx(expected.pvalue)
    assert (results.p_holm_global >= results.pvalue).all()
    assert set(correlations.lag) == set(range(13))
    with pytest.raises(ValueError, match="36"):
        diagnostics({"short": history.iloc[:20]})


def test_garch_recursion_simulation_and_eligibility(history):
    gate = arch_gates({"short": history.iloc[:40], "full": history})
    assert not gate.loc[gate.series_id == "short", "eligible"].iloc[0]
    fit = {
        "omega": 0.1,
        "alpha": 0.1,
        "beta": 0.8,
        "last_variance": 1.0,
        "last_residual": 0.5,
    }
    draws = garch_draws(fit, 1, 50000, generator("garch"))
    assert np.var(draws) == pytest.approx((0.1 + 0.1 * 0.25 + 0.8) / 10000, rel=0.025)
    fitted = fit_garch(history)
    assert fitted["omega"] > 0
    assert fitted["alpha"] + fitted["beta"] < 0.995
