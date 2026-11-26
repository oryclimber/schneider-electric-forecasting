"""Offer terms and stress scenarios adjusted for supplier counterparty risk.

Sits on top of ``offer_matrix.py`` (price volatility) and
``supplier_product_risk.py`` (supply-base risk). Price levers are not changed;
the supplier levers are added beside them.

Levers (every parameter below is a working assumption)
- Supplier buffer: extra price buffer on top of the price-volatility buffer.
  buffer = s x sum_j w_j x (risk_j - 1)/2 x SUPPLIER_BUFFER_MAX x horizon/9,
  with s the material share of price, w_j the cost share of material j and
  risk_j the base supply-base risk (1-3). SUPPLIER_BUFFER_MAX is the buffer, in
  % of material cost, for an all-high-risk supply base over 9 months.
- Validity cap: horizon x (1 - VALIDITY_CUT x (risk - 1)/2), whole months. The
  final validity is the shorter of this and the price-based validity.
- Supplier indexation flag: the material with the largest exposure to its
  weakest supplier (cost share x that supplier's volume share), restricted to
  materials whose weakest supplier is rated high. It is compared with the
  material already indexed on price volatility.
- Stress cases on the P90 downside: a fragile supplier raises its price by
  SUPPLIER_RAISE, or fails and its volume is re-sourced at SUPPLIER_FAIL above
  market. Only that supplier's volume (equal split) is hit.

Usage:
    uv run --locked python scripts/supplier_offer_terms.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from build_challenge_pack import MATERIAL_SHARES, PRODUCT_LABELS
from offer_matrix import build as build_offer_matrix
from supplier_product_risk import build_scorecard, material_risk, product_risk

from schneider_electric_forecasting import config

# ASSUMPTION (all five): working values chosen for scenario sizing, not estimated
# from data. Replace with procurement quotes / re-sourcing estimates when available.
SUPPLIER_BUFFER_MAX = (
    0.05  # ASSUMPTION: buffer, % of material cost, all-high base, 9 months
)
VALIDITY_CUT = 0.50  # ASSUMPTION: max share of validity cut for an all-high supply base
SUPPLIER_RAISE = 0.15  # ASSUMPTION: fragile supplier raises its price by 15%
SUPPLIER_FAIL = 0.30  # ASSUMPTION: failed supplier's volume re-sourced 30% above market
HIGH = (
    2.34  # ASSUMPTION: risk score from which a material counts as "high" (see _label)
)


def build():
    mat = material_risk(build_scorecard())
    detail, summary = product_risk(mat)
    rec, scen = build_offer_matrix()
    label_to_product = {v: k for k, v in PRODUCT_LABELS.items()}

    exposure, terms, stress = {}, [], []
    for label, block in detail.groupby("product"):
        s = MATERIAL_SHARES[label_to_product[label]]
        block = block.set_index("material")
        w = block.cost_share_pct / 100
        n = mat.n_suppliers.reindex(block.index)
        weak_high = block.downside_risk >= HIGH
        expo = (w / n).where(weak_high, 0.0)  # cost share on the weakest supplier
        exposure[label] = expo
        flag = expo.idxmax()
        base_risk = float(summary.set_index("product").loc[label, "base_risk"])
        for _, r in rec[rec["product"] == label].iterrows():
            h = int(r.offer_horizon_m)
            buffer = (
                s
                * float((w * (block.base_risk - 1) / 2).sum())
                * SUPPLIER_BUFFER_MAX
                * h
                / 9
            )
            cap = max(1, int(np.floor(h * (1 - VALIDITY_CUT * (base_risk - 1) / 2))))
            terms.append(
                {
                    "product": label,
                    "offer_horizon_m": h,
                    "price_quote_adj_pct": r.quote_adj_pct,
                    "supplier_buffer_pct": 100 * buffer,
                    "total_quote_adj_pct": r.quote_adj_pct + 100 * buffer,
                    "supplier_buffer_pct_of_margin": 100
                    * buffer
                    / (r.target_margin_pct / 100),
                    "price_validity_m": int(r.validity_m),
                    "supplier_validity_cap_m": cap,
                    "final_validity_m": min(int(r.validity_m), cap),
                    "price_indexed_material": r.indexed_material,
                    "supplier_flag_material": flag,
                    "weakest_supplier": mat.weakest_supplier[flag],
                    "flag_cost_exposure_pct": 100 * expo[flag],
                    "add_supplier_clause": flag != r.indexed_material,
                }
            )
            down = scen[
                (scen["product"] == label)
                & (scen.offer_horizon_m == h)
                & (scen.scenario == "downside")
            ].iloc[0]
            for case, shock in (
                ("supplier_raises_price", SUPPLIER_RAISE),
                ("supplier_fails", SUPPLIER_FAIL),
            ):
                hit = s * float(expo[flag]) * shock
                stress.append(
                    {
                        "product": label,
                        "offer_horizon_m": h,
                        "case": case,
                        "supplier": mat.weakest_supplier[flag],
                        "material": flag,
                        "supplier_cost_hit_pct_of_price": 100 * hit,
                        "downside_margin_pct": down.margin_after_pct,
                        "downside_margin_with_buffer_pct": down.margin_after_pct
                        + 100 * buffer,
                        "stress_margin_pct": down.margin_after_pct
                        + 100 * (buffer - hit),
                        "margin_change_vs_target_pp": down.margin_after_pct
                        + 100 * (buffer - hit)
                        - r.target_margin_pct,
                    }
                )
    return pd.DataFrame(terms), pd.DataFrame(stress)


def main():
    out = config.DATA_DIR / "submission"
    terms, stress = build()
    terms.to_csv(out / "supplier_offer_terms.csv", index=False)
    stress.to_csv(out / "supplier_stress_scenarios.csv", index=False)
    with pd.option_context(
        "display.width",
        250,
        "display.max_columns",
        30,
        "display.float_format",
        "{:.2f}".format,
    ):
        print(terms.to_string(index=False))
        print()
        print(stress.to_string(index=False))


if __name__ == "__main__":
    main()
