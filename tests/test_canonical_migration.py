"""Regression checks for workbook migration and statistical correctness."""

import numpy as np
import pandas as pd
import pytest
from statsmodels.tsa.stattools import adfuller, pacf

from schneider_electric_forecasting import ingestion
from schneider_electric_forecasting.eda import (
    autocorrelation_diagnostics,
    stationarity_tests,
)
from schneider_electric_forecasting.transforms import add_monthly_returns
from schneider_electric_forecasting.validation import validate_observation_snapshot


def test_bom_summary_rows_are_preserved_separately(monkeypatch):
    source = pd.DataFrame(
        {
            "product_ref": ["product", "Total mass"],
            "plant_code": ["plant", None],
            "mass_kg": [3, 3],
        }
    )
    monkeypatch.setattr(ingestion, "_read_excel", lambda *args, **kwargs: source.copy())
    assert ingestion.load_authoritative_bom().product_ref.tolist() == ["product"]
    assert ingestion.load_bom_metadata().product_ref.tolist() == ["Total mass"]


def test_workbook_eur_prices_are_recomputed_not_cached(monkeypatch):
    source = pd.DataFrame(
        {
            "month": pd.to_datetime(["2026-01-01", "2026-02-01"]),
            "S01": [100, 120],
            "S02": [40, 60],
            "S06": [2, np.nan],
            "D01": [999, 999],
            "D02": [999, 999],
        }
    )
    monkeypatch.setattr(ingestion, "_read_excel", lambda *args, **kwargs: source.copy())
    result = ingestion.load_rm_monthly_wide()
    assert result.loc[0, "D01"] == 50
    assert result.loc[0, "D02"] == 20
    assert pd.isna(result.loc[1, "D01"])


def test_monthly_return_does_not_bridge_missing_month():
    source = pd.DataFrame(
        {
            "month": pd.to_datetime(["2026-01-01", "2026-03-01"]),
            "series_id": ["A", "A"],
            "value": [100, 120],
        }
    )
    result = add_monthly_returns(source, value_column="value", group_by="series_id")
    assert result.value_pct_change.isna().all()


def test_energy_conversions_recompute_with_source_precision(monkeypatch):
    source = pd.DataFrame({f"E{number:02d}": [2.0] for number in range(1, 25)})
    source["month"] = pd.to_datetime(["2026-01-01"])
    source["ED02"] = 999.0
    monkeypatch.setattr(ingestion, "_read_excel", lambda *args, **kwargs: source.copy())
    result = ingestion.load_energy_monthly_wide()
    assert result.loc[0, "ED02"] == pytest.approx(1 / 0.293071)
    assert result.loc[0, "ED05"] == 1000
    assert result.loc[0, "ED06"] == 1


def test_stationarity_and_pacf_match_validated_library():
    values = np.random.default_rng(10).normal(size=100)
    source = pd.DataFrame(
        {"month": pd.date_range("2010-01-01", periods=100, freq="MS"), "value": values}
    )
    result = stationarity_tests(source, value_column="value")
    assert result.loc[0, "adf_pvalue"] == pytest.approx(
        adfuller(values, autolag="AIC")[1]
    )
    diagnostics = autocorrelation_diagnostics(source, value_column="value", max_lags=6)
    np.testing.assert_allclose(diagnostics.pacf, pacf(values, nlags=6, method="ywm"))
    source.loc[40, "value"] = np.nan
    assert (
        stationarity_tests(source, value_column="value").loc[0, "status"]
        == "missing_or_discontinuous_months"
    )


def test_snapshot_validation_rejects_invalid_price_and_metadata():
    source = pd.DataFrame(
        {
            "month": pd.to_datetime(["2026-01-01"]),
            "series_id": ["A"],
            "value": [-1],
            "unit": ["USD/t"],
        }
    )
    catalog = pd.DataFrame({"series_id": ["A"], "unit": ["EUR/t"]})
    report = validate_observation_snapshot(source, catalog)
    assert {issue.code for issue in report.errors} == {
        "INVALID_VALUE",
        "METADATA_MISMATCH",
    }
