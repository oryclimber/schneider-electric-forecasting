"""Offer matrix: price, risk buffer, validity and indexation by horizon/scenario.

Builds, for each RM product and each 3/6/9-month offer, the four levers under
upside / baseline / downside raw-material cases. It reuses the price panel and
BOM quantities from ``build_challenge_pack.py`` (imported, not modified).

Scenarios (Schneider is the buyer): upside = P10 BOM cost (materials cheaper),
baseline = P50, downside = P90 (materials dearer).

Levers (s = material share of price, C = simulated BOM cost index, 100 = now):
- Price: base price + expected cost increase = s * (E[C]/100 - 1)
- Risk buffer: BUFFER_FRACTION * s * (P90 - P50)/100, grows with the horizon.
  With an indexation clause the buffer is sized on the residual risk only.
- Validity: longest window <= horizon whose P90 price impact stays within the
  product's tolerance (TOLERANCE_MARGIN_FRACTION x target margin).
- Indexation: on the material with the largest share of BOM cost variance at
  the horizon. If its price moves more than T from the quote base, the price
  is adjusted by the full pass-through of that move. T is the material move
  whose price impact equals the tolerance.

All shares, margins, tolerance and buffer fraction are working assumptions.

Usage:
    uv run --locked python scripts/offer_matrix.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from build_challenge_pack import (
    BUFFER_FRACTION,
    MATERIAL_SHARES,
    N_SIMULATIONS,
    ORIGIN,
    PRODUCT_LABELS,
    PRODUCTS,
    SEED,
    TARGET_MARGINS,
    TARGETS,
    _bom_quantities,
    _build_price_panel,
)

from schneider_electric_forecasting import config

TOLERANCE_MARGIN_FRACTION = 0.10  # share of target margin an offer may lose
SCENARIOS = {"downside": 0.90, "baseline": 0.50, "upside": 0.10}


def _simulate(panel, keys, quantities, months, rng):
    """Joint empirical simulation of material prices and BOM cost by material."""
    changes = np.log(panel[keys].shift(-months) / panel[keys]).dropna()
    centred = changes - changes.median()
    draws = centred.to_numpy()[rng.choice(len(centred), N_SIMULATIONS)]
    base = panel.loc[ORIGIN, keys].astype(float).to_numpy()
    qty = np.array([quantities[k] for k in keys]) / 1000
    prices = base * np.exp(draws)
    return prices / base - 1, prices * qty, base * qty


def _validity(panel, keys, quantities, share, horizon, tolerance, rng):
    """Longest whole-month window whose P90 price impact is within tolerance."""
    best = 1
    for months in range(1, horizon + 1):
        _, cost, base_cost = _simulate(panel, keys, quantities, months, rng)
        index = 100 * cost.sum(axis=1) / base_cost.sum()
        if share * (np.quantile(index, 0.90) - 100) / 100 <= tolerance:
            best = months
    return best


def build():
    """Return (recommendation table, scenario table, hypothesis table)."""
    panel = _build_price_panel()
    bom = _bom_quantities()
    rec_rows, scen_rows = [], []
    for product in PRODUCTS:
        s, margin = MATERIAL_SHARES[product], TARGET_MARGINS[product]
        tolerance = TOLERANCE_MARGIN_FRACTION * margin
        quantities = bom[product]["quantities"]
        keys = sorted(quantities)
        rng = np.random.default_rng(SEED + 11)
        for horizon in TARGETS:
            move, cost, base_cost = _simulate(panel, keys, quantities, horizon, rng)
            total = cost.sum(axis=1)
            index = 100 * total / base_cost.sum()
            weights = base_cost / base_cost.sum()
            # Euler risk contribution of each material to BOM cost variance.
            contrib = np.array(
                [np.cov(cost[:, j], total)[0, 1] for j in range(len(keys))]
            ) / np.var(total, ddof=1)
            j = int(np.argmax(contrib))
            ref = keys[j]
            trigger = tolerance / (s * weights[j])  # material move, fraction
            # Residual risk: indexed material is passed through once it breaches.
            breach = np.abs(move[:, j]) > trigger
            residual = cost.copy()
            residual[breach, j] = base_cost[j]
            res_index = 100 * residual.sum(axis=1) / base_cost.sum()

            p90 = np.quantile(index, 0.90)
            expected = s * (index.mean() / 100 - 1)
            gross_buffer = BUFFER_FRACTION * s * (p90 - 100) / 100
            net_buffer = (
                BUFFER_FRACTION * s * (np.quantile(res_index, 0.90) - 100) / 100
            )
            validity = _validity(panel, keys, quantities, s, horizon, tolerance, rng)
            rec_rows.append(
                {
                    "product": PRODUCT_LABELS[product],
                    "offer_horizon_m": horizon,
                    "target_margin_pct": 100 * margin,
                    "base_plus_expected_pct": 100 * expected,
                    "p90_price_impact_pct": 100 * s * (p90 - 100) / 100,
                    "gross_buffer_pct": 100 * gross_buffer,
                    "net_buffer_pct": 100 * net_buffer,
                    "quote_adj_pct": 100 * (expected + net_buffer),
                    "validity_m": validity,
                    "indexed_material": ref,
                    "ref_risk_share_pct": 100 * contrib[j],
                    "second_material": keys[int(np.argsort(contrib)[-2])],
                    "trigger_pct": 100 * trigger,
                    "p_trigger_hit_pct": 100 * breach.mean(),
                    "buffer_pct_of_margin": 100 * net_buffer / margin,
                    "p90_impact_pct_of_margin": 100 * s * (p90 - 100) / 100 / margin,
                }
            )
            for scenario, q in SCENARIOS.items():
                i = float(np.quantile(index, q)) if scenario != "baseline" else 100.0
                cost_impact = s * (i / 100 - 1)
                ref_move = (
                    float(np.quantile(move[:, j], q)) if scenario != "baseline" else 0.0
                )
                clause_hit = abs(ref_move) > trigger
                clause_adj = s * weights[j] * ref_move if clause_hit else 0.0
                quote = expected + net_buffer
                margin_after = margin + quote + clause_adj - cost_impact
                scen_rows.append(
                    {
                        "product": PRODUCT_LABELS[product],
                        "offer_horizon_m": horizon,
                        "scenario": scenario,
                        "bom_index": i,
                        "cost_impact_pct": 100 * cost_impact,
                        "quote_adj_pct": 100 * quote,
                        "indexed_material": ref,
                        "ref_move_pct": 100 * ref_move,
                        "clause_triggered": clause_hit,
                        "clause_adj_pct": 100 * clause_adj,
                        "margin_after_pct": 100 * margin_after,
                        "margin_change_pp": 100 * (margin_after - margin),
                    }
                )
    return pd.DataFrame(rec_rows), pd.DataFrame(scen_rows)


def main():
    """Write the CSVs under data/submission and print both tables."""
    out = config.DATA_DIR / "submission"
    out.mkdir(parents=True, exist_ok=True)
    rec, scen = build()
    rec.to_csv(out / "offer_matrix_recommendation.csv", index=False)
    scen.to_csv(out / "offer_matrix_scenarios.csv", index=False)
    with pd.option_context(
        "display.width", 250, "display.float_format", "{:.2f}".format
    ):
        print(rec.to_string(index=False))
        print()
        print(scen.to_string(index=False))


if __name__ == "__main__":
    main()
