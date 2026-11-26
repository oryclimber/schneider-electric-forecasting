"""Regression checks for the standalone challenge pack builder."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from commands import build_challenge_pack as MODULE  # noqa: E402


def test_month_start_handles_series_and_index():
    series = pd.Series(pd.to_datetime(["2026-09-15", "2026-10-04"]))
    index = pd.DatetimeIndex(pd.to_datetime(["2026-09-01", "2026-10-15"]))
    assert MODULE._month_start(series).tolist() == [
        pd.Timestamp("2026-09-01"),
        pd.Timestamp("2026-10-01"),
    ]
    assert MODULE._month_start(index).tolist() == [
        pd.Timestamp("2026-09-01"),
        pd.Timestamp("2026-10-01"),
    ]


def test_forecast_is_locf_median_with_ordered_quantiles():
    rng = np.random.default_rng(7)
    months = pd.date_range("2020-01-01", periods=72, freq="MS")
    values = 100 * np.exp(np.cumsum(rng.normal(0, 0.03, size=len(months))))
    history = pd.Series(values, index=months)
    p10, p50, p90 = MODULE._forecast_series(history, 3, rng)
    assert p10 <= p50 <= p90
    assert p50 == history.iloc[-1]
    assert p10 < p90


def test_bom_classification_covers_direct_prices_and_exclusions():
    assert (
        MODULE._classify_bom(
            {"material_label_source": "Copper /ton", "material_family": "Copper"}
        )
        == "copper"
    )
    assert (
        MODULE._classify_bom(
            {"material_label_source": "Epoxy Resin EU", "material_family": "Plastic"}
        )
        == "epoxy"
    )
    assert (
        MODULE._classify_bom(
            {"material_label_source": "Brass", "material_family": "Brass"}
        )
        is None
    )


def test_offer_rule_uses_the_p90_p50_tail():
    forecast = pd.DataFrame(
        {
            "target_type": ["bom_index"] * 1,
            "target_id": ["RM6_NE_IQI_433520"],
            "horizon_m": [9],
            "target_month": ["2027-06"],
            "p10": [90.0],
            "p50": [100.0],
            "p90": [130.0],
        }
    )
    scenarios = MODULE._offer_scenarios(forecast)
    row = scenarios.iloc[0]
    expected = 0.80 * 0.50 * (130.0 - 100.0)
    assert row["recommended_price_adjustment_pct"] == expected
    assert row["quote_validity_months"] == 3
    assert row["indexation_trigger_pct"] == 5.0
