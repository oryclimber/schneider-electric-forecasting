"""Compare our proposed RM6 quote with competitors using the functional-modules benchmark.

Two lenses:
1. Price: our quote (Schneider 2026 P50 x (1 + quote_adj_pct)) vs the synthetic
   RMU benchmark P50s and the only observed list price (Siemens, webshop).
2. Cost: should-cost per brand from the RM6 NE-IQI functional-modules workbook,
   with the price each brand would need to earn our 30% target margin.

Caveats are written into the output: the RMU benchmark is synthetic (and its
workbook is DRM-protected, so P50s are taken from documentation/strategy_pack.md);
the should-cost workbook is provisional and SGC's total is implausible.

LIMITATION: should-cost totals are used as published, as an assumption. All part masses in the workbook are
"inferred, total held to the mass budget", and a few lines dominate each brand's
material cost (Schneider fasteners and packaging, Siemens/Ormazabal silver in
fuse-links, ABB copper busbars). See documentation/strategy_pack.md section 3.
Treat them as directional. Only the synthetic-P50 price comparison is independent of should-cost.

Usage:
    uv run --locked python scripts/cli.py competitor
"""

from __future__ import annotations

import pandas as pd

from schneider_electric_forecasting import config

FM = config.DATA_DIR / (
    "benchmark-modules-fonctionnels-rm6-ne-iqi-3-fonctions-400a-appareillage-"
    "isolement-et-coupure-dans-le-gaz-sf6.xlsx"
)
# 2026 P50, synthetic (documentation/strategy_pack.md section 3)
SYNTHETIC_P50 = {
    "Siemens": 13762,
    "ABB": 12273,
    "Schneider": 12233,
    "Eaton": 11560,
    "Ormazabal": 9652,
    "SGC": 9154,
}
OUT = config.DATA_DIR / "submission"
TARGET_MARGIN = 0.30


def _should_cost() -> pd.DataFrame:
    raw = pd.read_excel(FM, sheet_name="Benchmark", header=None)
    brands = ["Schneider", "Ormazabal", "SGC", "ABB", "Siemens", "Eaton"]
    keys = ["total", "material", "process", "labour", "overhead", "List price"]
    rows = raw.set_index(0).loc[keys, 1:6]
    rows.columns = brands
    out = rows.T.apply(pd.to_numeric, errors="coerce")
    return out.rename(columns={"List price": "list_price"})


def main() -> None:
    rec = pd.read_csv(OUT / "offer_matrix_recommendation.csv")
    adj = (
        rec[rec["product"].str.startswith("RM6")]
        .set_index("offer_horizon_m")
        .quote_adj_pct
    )
    sc = _should_cost()
    ours = SYNTHETIC_P50["Schneider"]

    df = sc.copy()
    df["synthetic_p50_2026"] = pd.Series(SYNTHETIC_P50)
    df["vs_our_base_pct"] = 100 * (df.synthetic_p50_2026 / ours - 1)
    for h, a in adj.items():
        df[f"vs_our_quote_{h}m_pct"] = 100 * (
            df.synthetic_p50_2026 / (ours * (1 + a / 100)) - 1
        )
    df["price_to_should_cost_x"] = df.synthetic_p50_2026 / df.total
    df["price_at_30pct_margin"] = df.total / (1 - TARGET_MARGIN)
    for c in ["material", "process", "labour", "overhead"]:
        df[f"{c}_pct_of_cost"] = 100 * df[c] / df.total
        df[f"{c}_vs_schneider_pct"] = 100 * (df[c] / df.loc["Schneider", c] - 1)
    df["total_vs_schneider_pct"] = 100 * (df.total / df.loc["Schneider", "total"] - 1)
    # like-for-like "should-price": each brand's cost at our target margin, plus our quote adjustment
    for h, a in adj.items():
        df[f"should_price_{h}m"] = df.price_at_30pct_margin * (1 + a / 100)
    df.index.name = "brand"
    df.round(2).to_csv(OUT / "competitor_price_comparison.csv")
    print(
        "LIMITATION: should-cost columns rest on inferred part masses and are directional "
        "(see documentation/strategy_pack.md section 3)."
    )
    pd.set_option("display.width", 250)
    print(df.round(1).to_string())
    print(
        "\nOur quote (RM6, adj %):",
        adj.round(2).to_dict(),
        {h: round(ours * (1 + a / 100)) for h, a in adj.items()},
    )


if __name__ == "__main__":
    main()
