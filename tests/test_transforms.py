"""Synthetic tests for non-destructive preprocessing helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from schneider_electric_forecasting.transforms import (
    TransformationError,
    add_monthly_returns,
    adjust_for_inflation,
    align_monthly_calendar,
    classify_missing_values,
    convert_units,
    convert_usd_to_eur,
    normalize_monthly_dates,
    rebase_price_index,
    separate_licensed_series,
)


def test_month_normalization_converts_timezone_to_utc_before_month() -> None:
    data = pd.DataFrame({"month": ["2026-03-01T00:30:00+02:00"]})

    normalized = normalize_monthly_dates(data)

    assert normalized.loc[0, "month"] == pd.Timestamp("2026-02-01")
    assert data.loc[0, "month"] == "2026-03-01T00:30:00+02:00"


def test_month_alignment_is_group_specific_and_does_not_fill_values() -> None:
    data = pd.DataFrame(
        {
            "month": ["2026-01-01", "2026-03-01", "2026-02-01", "2026-03-01"],
            "series_id": ["A", "A", "B", "B"],
            "value": [1.0, 3.0, 20.0, 30.0],
            "unit": ["USD/t"] * 4,
        }
    )

    aligned = align_monthly_calendar(
        data,
        group_by="series_id",
        value_columns=("value",),
    )

    assert len(aligned) == 5
    a_february = aligned.loc[
        (aligned["series_id"] == "A") & (aligned["month"] == "2026-02-01")
    ]
    assert a_february["value"].isna().all()
    assert a_february["unit"].eq("USD/t").all()
    assert data["value"].notna().all()


def test_alignment_rejects_duplicate_observation_keys_and_missing_dates() -> None:
    duplicates = pd.DataFrame(
        {
            "month": ["2026-01-01", "2026-01-10"],
            "series_id": ["A", "A"],
            "value": [1.0, 2.0],
        }
    )
    with pytest.raises(TransformationError, match="unique"):
        align_monthly_calendar(duplicates, group_by="series_id")
    malformed = pd.DataFrame({"month": ["not-a-date"], "value": [1.0]})
    with pytest.raises(TransformationError, match="invalid"):
        normalize_monthly_dates(malformed)


def test_missingness_classifies_licensed_series_without_observations() -> None:
    observations = pd.DataFrame(
        {
            "series_id": ["S01", "S01", "S02"],
            "value": [1.0, np.nan, 2.0],
        }
    )

    summary = classify_missing_values(
        observations,
        licensed_series_ids=("L01", "L02"),
        all_series_ids=("S01", "S02"),
    ).set_index("series_id")

    assert summary.loc["S01", "classification"] == "missing_observation"
    assert summary.loc["L01", "classification"] == "unavailable_licensed"
    assert summary.loc["S02", "classification"] == "observed"


def test_licensed_series_are_separated_without_changing_original() -> None:
    observations = pd.DataFrame({"series_id": ["S01", "L01"], "value": [1.0, 2.0]})
    catalog = pd.DataFrame(
        {"series_id": ["S01", "L01"], "access": ["Open", "Licensed"]}
    )

    open_data, licensed_data = separate_licensed_series(observations, catalog)

    assert open_data["series_id"].tolist() == ["S01"]
    assert licensed_data["series_id"].tolist() == ["L01"]
    assert len(observations) == 2


def test_explicit_unit_and_usd_to_eur_conversions() -> None:
    values = pd.Series([1000.0], name="copper")
    units = convert_units(values, factor=0.001, source_unit="kg", target_unit="t")
    euros = convert_usd_to_eur(values, pd.Series([1.25]))

    assert units.iloc[0] == 1.0
    assert units.attrs["unit"] == "t"
    assert euros.iloc[0] == 800.0
    assert euros.attrs["rate_definition"].startswith("USD per EUR")
    with pytest.raises(TransformationError, match="positive"):
        convert_usd_to_eur(values, 0)


def test_index_rebasing_uses_historical_base_and_masks_post_cutoff() -> None:
    data = pd.DataFrame(
        {
            "month": ["2026-01-01", "2026-02-01", "2026-03-01"],
            "series_id": ["S01"] * 3,
            "value": [50.0, 100.0, 200.0],
        }
    )

    rebased = rebase_price_index(
        data,
        value_column="value",
        date_column="month",
        base_period="2026-02",
        as_of="2026-02",
        group_by="series_id",
    )

    assert rebased["rebased_index"].iloc[:2].tolist() == [50.0, 100.0]
    assert pd.isna(rebased["rebased_index"].iloc[2])
    with pytest.raises(TransformationError, match="later than as_of"):
        rebase_price_index(
            data,
            value_column="value",
            date_column="month",
            base_period="2026-03",
            as_of="2026-02",
            group_by="series_id",
        )


def test_returns_do_not_bridge_missing_months_or_interpolate() -> None:
    data = pd.DataFrame(
        {
            "month": ["2026-01-01", "2026-02-01", "2026-03-01"],
            "series_id": ["S01"] * 3,
            "value": [100.0, np.nan, 120.0],
        }
    )

    result = add_monthly_returns(
        data, value_column="value", group_by="series_id", method="percent"
    )

    assert result["value_pct_change"].isna().all()
    with pytest.raises(TransformationError, match="strictly positive"):
        add_monthly_returns(
            pd.DataFrame({"month": ["2026-01-01"], "value": [-1.0]}),
            value_column="value",
            method="log",
        )


def test_inflation_adjustment_masks_months_after_origin() -> None:
    nominal = pd.DataFrame(
        {
            "month": ["2026-01-01", "2026-02-01", "2026-03-01"],
            "nominal": [100, 110, 120],
        }
    )
    inflation = pd.DataFrame(
        {"month": ["2026-01-01", "2026-02-01", "2026-03-01"], "index": [100, 110, 120]}
    )

    adjusted = adjust_for_inflation(
        nominal,
        inflation,
        value_column="nominal",
        inflation_column="index",
        base_period="2026-01",
        as_of="2026-02",
    )

    assert adjusted["real_value"].iloc[0] == 100
    assert adjusted["real_value"].iloc[1] == pytest.approx(100)
    assert pd.isna(adjusted["real_value"].iloc[2])
