"""Build the September 2026 challenge submission pack.

This script is deliberately separate from the Phase 3-5 research pipeline. It
uses the same authorized workbooks plus the previously dormant steel and epoxy
files to produce:

- a challenge-format forecast CSV for the 12 priceable series;
- a partial monetary BOM cost index for RM6 and RM AirSeT;
- BOM coverage and offer-rule scenario tables;
- an Excel workbook and two figures for the business presentation.

The point forecast is LOCF. P10/P90 come from centred empirical distributions
of historical cumulative log changes, so the median stays at the latest observed
price. The BOM index uses direct EUR/t prices for copper, aluminium, steel and
epoxy; plastics, brass and unidentified materials are excluded and disclosed.

Usage:
    uv run --locked python scripts/cli.py pack
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from schneider_electric_forecasting import _01_ingestion as ingestion
from schneider_electric_forecasting import config

TEAM_ID = "team_02"
ORIGIN = pd.Timestamp("2026-09-01")
TARGETS = {
    3: pd.Timestamp("2026-12-01"),
    6: pd.Timestamp("2027-03-01"),
    9: pd.Timestamp("2027-06-01"),
}
N_SIMULATIONS = 5000
SEED = 20260930
PRODUCTS = ("RM6_NE_IQI_433520", "RMAirSeT IIQ 24kV IEC")
MATERIAL_TARGETS = (
    "S01",
    "S02",
    "S03",
    "S04",
    "S05",
    "S07",
    "S08",
    "S09",
    "S10",
    "S11",
    "D01",
    "D02",
)
PRODUCT_LABELS = {
    "RM6_NE_IQI_433520": "RM6 / SM6 proxy",
    "RMAirSeT IIQ 24kV IEC": "RM AirSeT / SM AirSeT proxy",
}
TARGET_MARGINS = {"RM6_NE_IQI_433520": 0.30, "RMAirSeT IIQ 24kV IEC": 0.25}
MATERIAL_SHARES = {"RM6_NE_IQI_433520": 0.80, "RMAirSeT IIQ 24kV IEC": 0.80}
BUFFER_FRACTION = 0.50
INDEXATION_TRIGGER_PCT = 5.0
QUOTE_VALIDITY_MONTHS = 3

PRICE_KEYS = (
    "copper",
    "aluminium",
    "carbon_steel",
    "stainless_steel",
    "galvanised_steel",
    "epoxy",
)


def _month_start(values):
    """Normalise dates or labels to the first day of the reference month."""
    parsed = pd.to_datetime(values, errors="coerce")
    if isinstance(parsed, pd.Series):
        return parsed.dt.to_period("M").dt.to_timestamp()
    return parsed.to_period("M").to_timestamp()


def _load_steel_prices():
    """Return German CR, HDG and 304 stainless monthly prices in EUR/t."""
    path = config.DATA_DIR / "Copia de Europe Steel INDEXES.xlsx"
    meps = pd.read_excel(path, sheet_name="MEPS", header=4)
    meps = meps.loc[:, ["Month", "CR coil", "HDG coil"]].copy()
    meps.columns = ["month", "carbon_steel", "galvanised_steel"]
    meps["month"] = _month_start(meps["month"])

    stainless = pd.read_excel(path, sheet_name="CRU Stainless", header=4)
    stainless = stainless.loc[:, ["Month", "BASE", "Alloy"]].copy()
    stainless["stainless_steel"] = stainless["BASE"] + stainless["Alloy"]
    stainless["month"] = _month_start(stainless["Month"])
    stainless = stainless.loc[:, ["month", "stainless_steel"]]

    steel = meps.merge(stainless, on="month", how="outer")
    return steel.dropna(subset=["month"]).sort_values("month")


def _load_epoxy_prices():
    """Return European epoxy resin prices in EUR/t by month."""
    path = config.DATA_DIR / "Epoxy for student.xlsx"
    frame = pd.read_excel(path, sheet_name="Feuil1")
    frame = frame.iloc[:, [0, 6]].copy()
    frame.columns = ["month", "epoxy"]
    frame["month"] = _month_start(frame["month"])
    return frame.dropna(subset=["month"]).sort_values("month")


def _build_price_panel():
    """Build a monthly EUR/t panel for the six direct-price BOM materials."""
    rm = ingestion.load_rm_monthly_long()
    wide = (
        rm.loc[rm.series_id.isin(["D01", "D02"])]
        .pivot(index="month", columns="series_id", values="value")
        .rename(columns={"D01": "copper", "D02": "aluminium"})
    )
    wide.index = _month_start(wide.index)
    steel = _load_steel_prices().set_index("month")
    epoxy = _load_epoxy_prices().set_index("month")
    panel = wide.join(steel, how="outer").join(epoxy, how="outer")
    panel = panel.sort_index().asfreq("MS")
    # Small publication gaps are carried forward briefly; the origin month is
    # September 2026, so the August stainless value is the latest available.
    panel = panel.ffill(limit=2)
    return panel.loc["2018-01-01":"2026-09-01"]


def _classify_bom(row):
    """Map a BOM row to one of the six directly priced material keys."""
    label = str(row["material_label_source"]).lower()
    family = str(row["material_family"])
    if "epoxy" in label:
        return "epoxy"
    return {
        "Copper": "copper",
        "Aluminium": "aluminium",
        "Carbon steel": "carbon_steel",
        "Stainless steel": "stainless_steel",
        "Galvanised steel": "galvanised_steel",
    }.get(family)


def _bom_quantities():
    """Aggregate directly priced BOM mass by product and material key."""
    bom = ingestion.load_authoritative_bom().copy()
    bom = bom.loc[bom.product_ref.isin(PRODUCTS)].copy()
    bom["price_key"] = bom.apply(_classify_bom, axis=1)
    result = {}
    for product, block in bom.groupby("product_ref"):
        priced = block.loc[block.price_key.notna()]
        quantities = priced.groupby("price_key").mass_kg.sum().to_dict()
        total_mass = block.mass_kg.sum()
        priced_mass = priced.mass_kg.sum()
        result[product] = {
            "quantities": quantities,
            "total_mass_kg": float(total_mass),
            "priced_mass_kg": float(priced_mass),
            "coverage_pct": float(100 * priced_mass / total_mass),
            "excluded": sorted(
                block.loc[block.price_key.isna(), "material_label_source"].unique()
            ),
        }
    return result


def _forecast_series(history, horizon, rng):
    """LOCF median plus centred empirical P10/P90 from historical changes."""
    history = history.dropna().asfreq("MS")
    if len(history) < 36:
        raise ValueError("At least 36 observations are required.")
    changes = np.log(history.shift(-horizon) / history).dropna()
    if len(changes) < 12:
        raise ValueError("At least 12 overlapping historical changes are required.")
    centred = (changes - changes.median()).to_numpy()
    draws = rng.choice(centred, size=N_SIMULATIONS, replace=True)
    last = float(history.iloc[-1])
    prices = last * np.exp(draws)
    p10, p90 = np.quantile(prices, [0.10, 0.90])
    return float(p10), last, float(p90)


def _material_forecasts():
    """Forecast every priceable RM series and keep the source metadata."""
    rm = ingestion.load_rm_monthly_long()
    wide = rm.pivot(index="month", columns="series_id", values="value")
    wide.index = _month_start(wide.index)
    wide = wide.sort_index().asfreq("MS")
    catalog = ingestion.load_rm_source_catalog().set_index("series_id")
    rng = np.random.default_rng(SEED)
    rows = []
    for series_id in sorted(wide.columns):
        if series_id not in MATERIAL_TARGETS:
            continue
        values = wide[series_id]
        if values.notna().sum() < 36:
            continue
        for horizon, target in TARGETS.items():
            p10, p50, p90 = _forecast_series(values, horizon, rng)
            unit = str(catalog.loc[series_id, "unit"])
            region = (
                "EU"
                if series_id in ("D01", "D02")
                or "EU" in str(catalog.loc[series_id, "region"])
                else "WW"
            )
            rows.append(
                {
                    "target_type": "material",
                    "target_id": series_id,
                    "region": region,
                    "origin_month": ORIGIN.strftime("%Y-%m"),
                    "horizon_m": horizon,
                    "target_month": target.strftime("%Y-%m"),
                    "unit": unit,
                    "p10": p10,
                    "p50": p50,
                    "p90": p90,
                    "model_name": "locf_centred_empirical_v1",
                    "ai_tools": "DeepSeek Harness coding assistant; no synthetic prices",
                }
            )
    return pd.DataFrame(rows)


def _bom_index(panel, bom):
    """Build a partial monetary BOM index and its joint empirical quantiles."""
    rng = np.random.default_rng(SEED + 1)
    history, coverage, forecasts = {}, [], []
    for product in PRODUCTS:
        quantities = bom[product]["quantities"]
        keys = sorted(quantities)
        cost = sum(quantities[key] / 1000 * panel[key] for key in keys)
        base = float(cost.loc[ORIGIN])
        index = 100 * cost / base
        history[product] = index
        coverage.append(
            {
                "product_ref": product,
                "coverage_pct": bom[product]["coverage_pct"],
                "priced_mass_kg": bom[product]["priced_mass_kg"],
                "total_mass_kg": bom[product]["total_mass_kg"],
                "materials_priced": ", ".join(keys),
                "materials_excluded": ", ".join(bom[product]["excluded"]),
            }
        )
        base_prices = panel.loc[ORIGIN, keys].astype(float)
        for horizon, target in TARGETS.items():
            changes = np.log(panel[keys].shift(-horizon) / panel[keys]).dropna()
            centred = changes - changes.median()
            draws = centred.to_numpy()[rng.choice(len(centred), N_SIMULATIONS)]
            simulated = base_prices.to_numpy() * np.exp(draws)
            simulated_cost = simulated @ (
                np.array([quantities[key] for key in keys]) / 1000
            )
            simulated_index = 100 * simulated_cost / base
            p10, p90 = np.quantile(simulated_index, [0.10, 0.90])
            forecasts.append(
                {
                    "target_type": "bom_index",
                    "target_id": product,
                    "region": "EU",
                    "origin_month": ORIGIN.strftime("%Y-%m"),
                    "horizon_m": horizon,
                    "target_month": target.strftime("%Y-%m"),
                    "unit": "index (2026-09=100)",
                    "p10": float(p10),
                    "p50": 100.0,
                    "p90": float(p90),
                    "model_name": "partial_monetary_bom_v1",
                    "ai_tools": "DeepSeek Harness coding assistant; no synthetic prices",
                }
            )
    return history, pd.DataFrame(coverage), pd.DataFrame(forecasts)


def _offer_scenarios(forecast):
    """Translate BOM quantiles into a simple quote-adjustment rule."""
    rows = []
    for product in PRODUCTS:
        block = forecast.loc[forecast.target_id == product]
        for row in block.itertuples():
            p50, p90 = float(row.p50), float(row.p90)
            material_share = MATERIAL_SHARES[product]
            buffer = material_share * BUFFER_FRACTION * (p90 - p50) / 100
            adjustment = material_share * (p50 / 100 - 1) + buffer
            rows.append(
                {
                    "product_ref": product,
                    "horizon_m": row.horizon_m,
                    "target_month": row.target_month,
                    "bom_p50_index": p50,
                    "bom_p90_index": p90,
                    "material_share_assumption": material_share,
                    "risk_buffer_fraction": BUFFER_FRACTION,
                    "recommended_price_adjustment_pct": 100 * adjustment,
                    "target_gross_margin": TARGET_MARGINS[product],
                    "quote_validity_months": QUOTE_VALIDITY_MONTHS,
                    "indexation_trigger_pct": INDEXATION_TRIGGER_PCT,
                }
            )
    return pd.DataFrame(rows)


def _figures(history, forecast, out):
    """Save a BOM fan chart and a compact material forecast panel."""
    fig, axes = plt.subplots(2, 1, figsize=(11, 9), sharex=True)
    for axis, (product, colour) in zip(
        axes, zip(PRODUCTS, ("#3DCD58", "#0C3027"), strict=True), strict=True
    ):
        index = history[product].dropna()
        axis.plot(index.index, index.values, color=colour, label="History")
        block = forecast.loc[forecast.target_id == product].sort_values("horizon_m")
        block["target_month"] = pd.to_datetime(block["target_month"])
        axis.fill_between(
            block.target_month,
            block.p10,
            block.p90,
            color=colour,
            alpha=0.20,
            label="P10-P90",
        )
        axis.plot(block.target_month, block.p50, "o--", color=colour, label="P50")
        axis.axhline(100, color="#999999", linewidth=0.8, linestyle=":")
        axis.set_title(PRODUCT_LABELS[product])
        axis.set_ylabel("Index (Sep 2026 = 100)")
        axis.legend(loc="upper left", fontsize=8)
    fig.suptitle("Partial monetary BOM cost index")
    fig.tight_layout()
    fig.savefig(out / "bom_cost_index.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    materials = forecast.loc[forecast.target_type == "material"]
    selected = sorted(materials.target_id.unique())
    fig, axes = plt.subplots(3, 4, figsize=(16, 10), sharex=True)
    for axis, series_id in zip(axes.flat, selected, strict=False):
        block = materials.loc[materials.target_id == series_id].sort_values("horizon_m")
        block["target_month"] = pd.to_datetime(block["target_month"])
        axis.fill_between(
            block.target_month, block.p10, block.p90, color="#3DCD58", alpha=0.2
        )
        axis.plot(block.target_month, block.p50, "o-", color="#0C3027")
        axis.set_title(series_id, fontsize=10)
        axis.tick_params(axis="x", labelrotation=30)
    for axis in axes.flat[len(selected) :]:
        axis.axis("off")
    fig.suptitle("Material forecasts: LOCF median and empirical P10-P90")
    fig.tight_layout()
    fig.savefig(out / "material_forecasts.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


REQUIRED_COLUMNS = [
    "team_id",
    "target_type",
    "target_id",
    "region",
    "origin_month",
    "horizon_m",
    "target_month",
    "unit",
    "p10",
    "p50",
    "p90",
    "model_name",
    "ai_tools",
]


def _validate(submission, coverage, scenarios):
    """Fail closed if the generated pack breaks its own contract."""
    if list(submission.columns) != REQUIRED_COLUMNS:
        raise ValueError("Submission columns do not match the challenge schema.")
    if len(submission) != 42:
        raise ValueError("Expected 36 material rows and 6 BOM rows.")
    if submission.duplicated(["target_type", "target_id", "horizon_m"]).any():
        raise ValueError("Duplicate target/horizon rows.")
    if not (
        (submission.p10 <= submission.p50) & (submission.p50 <= submission.p90)
    ).all():
        raise ValueError("Quantiles are not ordered P10 <= P50 <= P90.")
    if set(submission.origin_month) != {"2026-09"}:
        raise ValueError("Origin must be September 2026.")
    expected_targets = {"2026-12", "2027-03", "2027-06"}
    if set(submission.target_month) != expected_targets:
        raise ValueError("Target months must be Dec 2026, Mar 2027 and Jun 2027.")
    if set(submission.target_type) != {"material", "bom_index"}:
        raise ValueError("Unexpected target types.")
    if coverage.coverage_pct.min() < 80:
        raise ValueError("BOM coverage fell below the 80% floor.")
    if (scenarios.recommended_price_adjustment_pct < 0).any():
        raise ValueError("Offer rule produced a negative adjustment.")
    bom = submission.loc[submission.target_type == "bom_index"]
    if not (bom.p50 == 100).all():
        raise ValueError("BOM index P50 must equal 100 at the origin.")
    return {
        "rows": int(len(submission)),
        "materials": int((submission.target_type == "material").sum()),
        "bom_rows": int(len(bom)),
        "min_bom_coverage_pct": float(coverage.coverage_pct.min()),
        "max_adjustment_pct": float(scenarios.recommended_price_adjustment_pct.max()),
    }


def main():
    """Build every generated challenge-pack artifact under data/submission."""
    out = config.DATA_DIR / "submission"
    out.mkdir(parents=True, exist_ok=True)
    panel = _build_price_panel()
    forecast_materials = _material_forecasts()
    bom = _bom_quantities()
    history, coverage, forecast_bom = _bom_index(panel, bom)
    forecast = pd.concat([forecast_materials, forecast_bom], ignore_index=True)
    forecast = forecast.sort_values(["target_type", "target_id", "horizon_m"])
    scenarios = _offer_scenarios(forecast_bom)

    submission = forecast[
        [
            "target_type",
            "target_id",
            "region",
            "origin_month",
            "horizon_m",
            "target_month",
            "unit",
            "p10",
            "p50",
            "p90",
            "model_name",
            "ai_tools",
        ]
    ].copy()
    submission.insert(0, "team_id", TEAM_ID)
    checks = _validate(submission, coverage, scenarios)
    submission_path = out / f"{TEAM_ID}_forecast_202609.csv"
    submission.to_csv(submission_path, index=False)
    forecast.to_csv(out / "forecast_details.csv", index=False)
    coverage.to_csv(out / "bom_coverage.csv", index=False)
    scenarios.to_csv(out / "offer_rule_scenarios.csv", index=False)

    index_history = pd.DataFrame(history).rename(
        columns={product: product for product in PRODUCTS}
    )
    index_history.index.name = "month"
    index_history.reset_index().to_csv(out / "bom_cost_index_history.csv", index=False)

    _figures(history, forecast, out)

    assumptions = pd.DataFrame(
        [
            ("origin", "September 2026; October-November excluded"),
            ("cutoff", "Challenge convention: use the latest complete month"),
            ("point_forecast", "LOCF (last observation carried forward)"),
            ("interval_method", "Centred empirical cumulative log changes"),
            ("simulations", str(N_SIMULATIONS)),
            ("bom_method", "Partial monetary cost index, 2026-09 = 100"),
            ("bom_prices", "D01/D02, German CR/HDG steel, 304 stainless, epoxy"),
            ("bom_excluded", "Plastics, brass, unidentified/other materials"),
            (
                "margin_rm6",
                f"{TARGET_MARGINS['RM6_NE_IQI_433520']:.0%} working assumption",
            ),
            (
                "margin_airset",
                f"{TARGET_MARGINS['RMAirSeT IIQ 24kV IEC']:.0%} working assumption",
            ),
            ("risk_buffer", f"{BUFFER_FRACTION:.0%} of the P90-P50 cost gap"),
            ("quote_validity", f"{QUOTE_VALIDITY_MONTHS} months"),
            (
                "indexation_trigger",
                f">{INDEXATION_TRIGGER_PCT:.0f}% material index move",
            ),
            ("ai_disclosure", "AI used for code assistance and drafting only"),
        ],
        columns=["item", "assumption"],
    )
    with pd.ExcelWriter(
        out / "schneider_challenge_pack.xlsx", engine="openpyxl"
    ) as writer:
        submission.to_excel(writer, sheet_name="Forecast_Submission", index=False)
        forecast.to_excel(writer, sheet_name="Forecast_Details", index=False)
        coverage.to_excel(writer, sheet_name="BOM_Coverage", index=False)
        index_history.reset_index().to_excel(
            writer, sheet_name="BOM_Index_History", index=False
        )
        scenarios.to_excel(writer, sheet_name="Offer_Rule", index=False)
        assumptions.to_excel(writer, sheet_name="Assumptions", index=False)

    summary = {
        "submission_rows": int(len(submission)),
        "material_rows": int(len(forecast_materials)),
        "bom_rows": int(len(forecast_bom)),
        "bom_coverage_pct": coverage.set_index("product_ref").coverage_pct.to_dict(),
        "checks": checks,
        "output": str(out),
    }
    (out / "validation.json").write_text(json.dumps(checks, indent=2) + "\n")
    (out / "build_manifest.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    main()
