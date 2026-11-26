"""Competitive risk view: our supplier-adjusted quote against competitor prices.

Joins the three existing streams, reading their CSV outputs in data/submission/:
- ``offer_matrix_scenarios.csv``  (price-volatility quote, downside cost, clause)
- ``supplier_offer_terms.csv``    (supplier buffer, validity cap, supplier clause flag)
- ``supplier_stress_scenarios.csv`` (supplier price rise / failure cost hit)
- ``competitor_price_comparison.csv`` (synthetic 2026 P50 per brand)

One row per product x horizon:
1. Our quote = base x (1 + total_quote_adj_pct), i.e. price-volatility adjustment
   plus the supplier buffer.
2. Gap to the comparable rivals, with and without the supplier buffer, so the
   buffer's own cost in competitiveness is visible (``buffer_cost_in_gap_pp``).
3. Gross margin if we match the reference competitor price under the downside
   cost case, and under the downside plus each supplier stress case:
   GM = (P_comp - COGS) / P_comp   (business_problem_statement.md, section 2).
4. A status and a suggested lever (keep the buffer, move risk into an
   indexation clause / shorter validity, or accept thinner margin).

Cost units: offer_matrix.py works in fractions of the base price, with
margin_after = target + quote_adj + clause_adj - cost_impact. Hence downside
revenue = 1 + quote + clause, and COGS = revenue - margin_after. The supplier
buffer is extra revenue; a supplier hit is extra COGS.

ASSUMPTIONS / LIMITATIONS
- Competitor prices are the SYNTHETIC 2026 P50s and are held flat over the
  horizon (no competitor trend data exists).
- Competitors exist for RM6 only. The AirSeT rows keep the supplier terms but
  have no competitor columns (no validated AirSeT competitor mapping).
- Comparable rivals are Siemens, ABB and Eaton. Ormazabal and SGC are excluded as
  low-cost outliers (SGC's should-cost total is implausible).
- Reference competitor = median of the comparable rivals. TIGHT_BAND is a
  judgment cut-off. All output is directional scenario analysis, not pricing policy.

Usage:
    uv run --locked python scripts/competitive_risk_view.py
"""

from __future__ import annotations

import pandas as pd

from schneider_electric_forecasting import config

OUT = config.DATA_DIR / "submission"
COMPARABLE = ["Siemens", "ABB", "Eaton"]  # ASSUMPTION: see module docstring
TIGHT_BAND = 0.03  # ASSUMPTION: quote within 3% above reference counts as "tight"
COMPETITOR_PRODUCT = "RM6"
# Sensitivity grid for offer_matrix.TOLERANCE_MARGIN_FRACTION (base case 0.10).
TOLERANCE_GRID = [0.10, 0.15, 0.25, 0.50, 0.75, 1.00]


def _status(gap_with: float, gap_without: float) -> str:
    if gap_with <= 0:
        return "competitive"
    if gap_without <= 0:
        return "buffer_makes_uncompetitive"
    return "uncompetitive_before_buffer" if gap_with > TIGHT_BAND else "tight"


def _lever(status: str, add_clause: bool, margin_matched: float, target: float) -> str:
    if status == "competitive":
        return "keep full supplier buffer"
    if status in ("buffer_makes_uncompetitive", "tight"):
        base = "move supplier risk into an indexation clause / shorter validity instead of price"
        return base + (" (clause flagged)" if add_clause else "")
    if margin_matched >= target:
        return "match competitor: target margin still holds"
    return "price above rivals or accept thinner margin; buffer is not the cause"


def build() -> pd.DataFrame:
    scen = pd.read_csv(OUT / "offer_matrix_scenarios.csv")
    terms = pd.read_csv(OUT / "supplier_offer_terms.csv")
    stress = pd.read_csv(OUT / "supplier_stress_scenarios.csv")
    comp = pd.read_csv(OUT / "competitor_price_comparison.csv").set_index("brand")
    rec = pd.read_csv(OUT / "offer_matrix_recommendation.csv")

    ours = float(comp.loc["Schneider", "synthetic_p50_2026"])
    ref = float(comp.loc[COMPARABLE, "synthetic_p50_2026"].median())
    nearest = float(comp.loc[COMPARABLE, "synthetic_p50_2026"].min())
    keys = ["product", "offer_horizon_m"]
    target = rec.set_index(keys).target_margin_pct

    down = scen[scen.scenario == "downside"].set_index(keys)
    base = scen[scen.scenario == "baseline"].set_index(keys)
    rows = []
    for _, t in terms.iterrows():
        k = (t["product"], int(t.offer_horizon_m))
        has_comp = t["product"].startswith(COMPETITOR_PRODUCT)
        tm = float(target[k])
        d = down.loc[k]
        # downside revenue/COGS in fractions of base price (see docstring)
        rev_d = 1 + (d.quote_adj_pct + d.clause_adj_pct) / 100
        cogs_d = rev_d - d.margin_after_pct / 100
        row = {
            "product": t["product"],
            "offer_horizon_m": k[1],
            "target_margin_pct": tm,
            "price_quote_adj_pct": t.price_quote_adj_pct,
            "supplier_buffer_pct": t.supplier_buffer_pct,
            "total_quote_adj_pct": t.total_quote_adj_pct,
            "final_validity_m": t.final_validity_m,
            "add_supplier_clause": t.add_supplier_clause,
            "supplier_flag_material": t.supplier_flag_material,
            "weakest_supplier": t.weakest_supplier,
            "baseline_margin_pct": base.loc[k].margin_after_pct,
        }
        if has_comp:
            quote = ours * (1 + t.total_quote_adj_pct / 100)
            quote_nobuf = ours * (1 + t.price_quote_adj_pct / 100)
            gap = quote / ref - 1
            gap_nb = quote_nobuf / ref - 1
            p_ref = ref / ours  # competitor price in base-price units
            row.update(
                {
                    "our_quote": quote,
                    "reference_competitor_price": ref,
                    "lowest_comparable_price": nearest,
                    "gap_to_reference_pct": 100 * gap,
                    "gap_without_buffer_pct": 100 * gap_nb,
                    "buffer_cost_in_gap_pp": 100 * (gap - gap_nb),
                    "gap_to_lowest_comparable_pct": 100 * (quote / nearest - 1),
                    "gm_at_competitor_downside_pct": 100 * (p_ref - cogs_d) / p_ref,
                }
            )
            for case in ("supplier_raises_price", "supplier_fails"):
                hit = (
                    stress[
                        (stress["product"] == k[0])
                        & (stress.offer_horizon_m == k[1])
                        & (stress.case == case)
                    ]
                    .iloc[0]
                    .supplier_cost_hit_pct_of_price
                    / 100
                )
                row[f"gm_at_competitor_{case}_pct"] = (
                    100 * (p_ref - cogs_d - hit) / p_ref
                )
            row["status"] = _status(gap, gap_nb)
            row["suggested_lever"] = _lever(
                row["status"],
                bool(t.add_supplier_clause),
                row["gm_at_competitor_downside_pct"],
                tm,
            )
        else:
            row["status"] = "no comparable competitor"
            row["suggested_lever"] = (
                "supplier terms only; add competitor mapping for AirSeT"
            )
        rows.append(row)
    return pd.DataFrame(rows)


def sensitivity() -> pd.DataFrame:
    """Re-run the offer matrix at other loss tolerances; RM6 only (has competitors).

    The tolerance (share of target margin an offer may lose at P90) sets both the
    price-based validity and the indexation trigger, hence the quote adjustment.
    The supplier buffer and supplier validity cap do not depend on it, so they are
    taken from supplier_offer_terms.csv.
    """
    import offer_matrix

    terms = pd.read_csv(OUT / "supplier_offer_terms.csv").set_index(
        ["product", "offer_horizon_m"]
    )
    comp = pd.read_csv(OUT / "competitor_price_comparison.csv").set_index("brand")
    ours = float(comp.loc["Schneider", "synthetic_p50_2026"])
    ref = float(comp.loc[COMPARABLE, "synthetic_p50_2026"].median())
    original = offer_matrix.TOLERANCE_MARGIN_FRACTION
    rows = []
    try:
        for frac in TOLERANCE_GRID:
            offer_matrix.TOLERANCE_MARGIN_FRACTION = frac
            rec, scen = offer_matrix.build()
            down = scen[scen.scenario == "downside"].set_index(
                ["product", "offer_horizon_m"]
            )
            for _, r in rec[
                rec["product"].str.startswith(COMPETITOR_PRODUCT)
            ].iterrows():
                k = (r["product"], int(r.offer_horizon_m))
                t, d = terms.loc[k], down.loc[k]
                total_adj = r.quote_adj_pct + t.supplier_buffer_pct
                rev = 1 + (d.quote_adj_pct + d.clause_adj_pct) / 100
                cogs = rev - d.margin_after_pct / 100
                p_ref = ref / ours
                rows.append(
                    {
                        "tolerance_fraction_of_margin": frac,
                        "tolerance_pp_of_price": r.target_margin_pct * frac,
                        "offer_horizon_m": k[1],
                        "price_validity_m": int(r.validity_m),
                        "final_validity_m": min(
                            int(r.validity_m), int(t.supplier_validity_cap_m)
                        ),
                        "quote_adj_pct": r.quote_adj_pct,
                        "total_quote_adj_pct": total_adj,
                        "gap_to_reference_pct": 100
                        * (ours * (1 + total_adj / 100) / ref - 1),
                        "gm_at_competitor_downside_pct": 100 * (p_ref - cogs) / p_ref,
                    }
                )
    finally:
        offer_matrix.TOLERANCE_MARGIN_FRACTION = original
    return pd.DataFrame(rows)


def main() -> None:
    df = build()
    df.round(3).to_csv(OUT / "competitive_risk_view.csv", index=False)
    sens = sensitivity()
    sens.round(3).to_csv(OUT / "competitive_risk_sensitivity.csv", index=False)
    print("\nSensitivity to the loss tolerance (RM6; base case is 0.10):")
    with pd.option_context(
        "display.width", 300, "display.float_format", "{:.2f}".format
    ):
        print(sens.to_string(index=False))
    print()
    print(
        "LIMITATION: competitor prices are synthetic 2026 P50s held flat; RM6 only; "
        "directional scenario analysis (see module docstring)."
    )
    cols = [
        "product",
        "offer_horizon_m",
        "total_quote_adj_pct",
        "supplier_buffer_pct",
        "our_quote",
        "gap_to_reference_pct",
        "buffer_cost_in_gap_pp",
        "gm_at_competitor_downside_pct",
        "gm_at_competitor_supplier_fails_pct",
        "final_validity_m",
        "status",
        "suggested_lever",
    ]
    with pd.option_context(
        "display.width",
        300,
        "display.max_columns",
        30,
        "display.max_colwidth",
        60,
        "display.float_format",
        "{:.2f}".format,
    ):
        print(df.reindex(columns=cols).to_string(index=False))


if __name__ == "__main__":
    main()
