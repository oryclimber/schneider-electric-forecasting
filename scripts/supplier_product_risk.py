"""Supply-base risk per material and per product, from the supplier scorecard.

Counterparty risk, kept separate from price volatility: it feeds offer terms and
the downside scenario, not the price forecast.

Method
- Ratings come from ``supplier_risk.py``; low = 1, medium = 2, high = 3.
- Each priced material has a set of suppliers (``MATERIAL_SUPPLIERS``, from
  scripts/supplier_material_map.py, which also writes the map CSV). ASSUMPTION: suppliers of the same
  material are weighted equally; no volume split exists in the data.
- Base supply-base risk of a material = equal-weighted mean of its suppliers'
  ratings. Downside = the weakest (highest-rated) supplier.
- Product risk = sum over materials of cost share x supply-base risk, using the
  index's own cost shares (BOM mass x origin price, as in ``offer_matrix.py``).
- A material with no mapped supplier is "no information", not "low risk": it is
  left out of the sum, and the covered share of priced cost is reported. The
  headline figure is the sum divided by that covered share, so missing
  materials do not pull the figure toward zero.

Usage:
    uv run --locked python scripts/supplier_product_risk.py
"""

from __future__ import annotations

import unicodedata

import numpy as np
import pandas as pd
from build_challenge_pack import (
    ORIGIN,
    PRODUCT_LABELS,
    PRODUCTS,
    _bom_quantities,
    _build_price_panel,
)
from supplier_material_map import MATERIAL_SUPPLIERS
from supplier_risk import build as build_scorecard

from schneider_electric_forecasting import config

POINTS = {"low": 1, "medium": 2, "high": 3}  # ASSUMPTION: linear scoring of ratings

def _fold(name: str) -> str:
    """Accent-stripped name, so 'Rötzer' matches however the file encodes it."""
    return unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()


def _label(score: float) -> str:
    # ASSUMPTION: equal-width bands over the 1-3 scale.
    return "low" if score < 1.67 else ("medium" if score < 2.34 else "high")


def material_risk(scorecard: pd.DataFrame) -> pd.DataFrame:
    scorecard = scorecard.rename(index=_fold)
    points = scorecard.rating.map(POINTS)
    rows = []
    for material, suppliers in MATERIAL_SUPPLIERS.items():
        vals = {s: points[s] for s in suppliers}
        rows.append(
            {
                "material": material,
                "suppliers": "; ".join(f"{s} ({scorecard.rating[s]})" for s in suppliers),
                "n_suppliers": len(vals),
                "base_risk": float(np.mean(list(vals.values()))),
                "downside_risk": float(max(vals.values())),
                "weakest_supplier": max(vals, key=vals.get),
            }
        )
    out = pd.DataFrame(rows).set_index("material")
    out["base_label"] = out.base_risk.map(_label)
    out["downside_label"] = out.downside_risk.map(_label)
    return out


def product_risk(mat: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    panel, bom = _build_price_panel(), _bom_quantities()
    detail, summary = [], []
    for product in PRODUCTS:
        qty = bom[product]["quantities"]
        cost = pd.Series({k: panel.loc[ORIGIN, k] * q / 1000 for k, q in qty.items()})
        share = cost / cost.sum()
        mapped = share.index.intersection(mat.index)
        covered = float(share[mapped].sum())
        for k in share.index:
            detail.append(
                {
                    "product": PRODUCT_LABELS[product],
                    "material": k,
                    "cost_share_pct": 100 * share[k],
                    "base_risk": mat.base_risk.get(k, np.nan),
                    "downside_risk": mat.downside_risk.get(k, np.nan),
                    "status": "mapped" if k in mat.index else "no information",
                }
            )
        base = float((share[mapped] * mat.base_risk[mapped]).sum())
        down = float((share[mapped] * mat.downside_risk[mapped]).sum())
        summary.append(
            {
                "product": PRODUCT_LABELS[product],
                "priced_cost_covered_pct": 100 * covered,
                "priced_mass_coverage_pct": bom[product]["coverage_pct"],
                "base_risk_sum": base,
                "base_risk": base / covered,
                "base_label": _label(base / covered),
                "downside_risk_sum": down,
                "downside_risk": down / covered,
                "downside_label": _label(down / covered),
                "share_high_risk_downside_pct": 100
                * float(share[mapped][mat.downside_risk[mapped] >= 2.34].sum()),
                "unknown_pct": 100 * (1 - covered),
            }
        )
    return pd.DataFrame(detail), pd.DataFrame(summary)


def main():
    out = config.DATA_DIR / "submission"
    mat = material_risk(build_scorecard())
    detail, summary = product_risk(mat)
    mat.to_csv(out / "supply_base_risk_by_material.csv")
    detail.to_csv(out / "supply_base_risk_product_detail.csv", index=False)
    summary.to_csv(out / "supply_base_risk_by_product.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30, "display.float_format", "{:.2f}".format):
        print(mat.to_string()); print(); print(detail.to_string(index=False)); print(); print(summary.T.to_string())


if __name__ == "__main__":
    main()
