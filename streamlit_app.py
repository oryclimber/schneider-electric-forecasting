"""Interactive decision tool for the September 2026 Schneider challenge pack.

Run with:
    uv run --group app streamlit run streamlit_app.py

The app is a presentation companion, not a pricing policy. It reads the generated
CSV/XLSX artifacts under ``data/submission`` and never retrains a model or writes
to a source workbook. Build the pack first with:

    uv run --locked python scripts/build_challenge_pack.py
"""

from __future__ import annotations

from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "submission"

PRODUCT_LABELS = {
    "RM6_NE_IQI_433520": "RM6 / SM6 proxy",
    "RMAirSeT IIQ 24kV IEC": "RM AirSeT / SM AirSeT proxy",
}
DEFAULT_MARGIN = {
    "RM6_NE_IQI_433520": 0.30,
    "RMAirSeT IIQ 24kV IEC": 0.25,
}
DEFAULT_QUOTE = {
    "RM6_NE_IQI_433520": 12_233,
    "RMAirSeT IIQ 24kV IEC": 12_233,
}
# Current pack decision: 80% of the selling price is exposed material cost.
DEFAULT_MATERIAL_SHARE = 0.80
DEFAULT_BUFFER = 0.50
COMPETITOR_REFERENCE = 12_273  # RM6 median of Siemens, ABB and Eaton (synthetic)

st.set_page_config(
    page_title="Schneider challenge — decision tool",
    page_icon="⚡",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def _read(name: str, **kwargs) -> pd.DataFrame | None:
    """Read an optional pack table, returning None when it is absent."""
    path = DATA / name
    return pd.read_csv(path, **kwargs) if path.exists() else None


@st.cache_data(show_spinner=False)
def load_data():
    """Load the mandatory and optional pack tables once per session."""
    required = {
        "forecast": "team_02_forecast_202609.csv",
        "history": "bom_cost_index_history.csv",
        "coverage": "bom_coverage.csv",
    }
    missing = [name for name in required.values() if not (DATA / name).exists()]
    if missing:
        st.error(
            "The pack is missing: "
            + ", ".join(missing)
            + ". Run `uv run --locked python scripts/build_challenge_pack.py` first."
        )
        st.stop()
    return {
        "forecast": _read(
            required["forecast"],
        ),
        "history": _read(required["history"], parse_dates=["month"]),
        "coverage": _read(required["coverage"]),
        "rules": _read("offer_rule_scenarios.csv"),
        "recommendation": _read("offer_matrix_recommendation.csv"),
        "scenarios": _read("offer_matrix_scenarios.csv"),
        "competitor": _read("competitor_price_comparison.csv"),
    }


def style():
    """Apply the light Schneider-inspired theme used in the deck."""
    st.markdown(
        """
        <style>
        .block-container {padding-top: 1.4rem; padding-bottom: 2rem;}
        h1, h2, h3 {color: #0C3027;}
        div[data-testid="stMetricValue"] {color: #0C3027;}
        div[data-testid="stMetricDelta"] {font-size: 0.85rem;}
        .stTabs [data-baseweb="tab"] {font-weight: 600;}
        .callout {background:#E8F3EC; border-left:5px solid #3DCD58;
                  padding:0.9rem 1rem; border-radius:4px;}
        .warn {background:#FFF4E5; border-left:5px solid #D09380;
               padding:0.9rem 1rem; border-radius:4px;}
        </style>
        """,
        unsafe_allow_html=True,
    )


def bom_chart(history, forecast, product):
    """Build the BOM history and forecast chart for one product."""
    hist = history[["month", product]].rename(columns={product: "value"})
    hist["series"] = "History"
    fc = forecast.loc[
        (forecast.target_id == product) & (forecast.target_type == "bom_index")
    ].copy()
    fc["target_month"] = pd.to_datetime(fc["target_month"] + "-01")
    band = (
        alt.Chart(fc)
        .mark_area(opacity=0.18, color="#3DCD58")
        .encode(
            x=alt.X("target_month:T", title="Month"),
            y=alt.Y("p10:Q", title="Index (Sep 2026 = 100)"),
            y2="p90:Q",
        )
    )
    line = (
        alt.Chart(hist)
        .mark_line(color="#0C3027", strokeWidth=2)
        .encode(x=alt.X("month:T", title="Month"), y="value:Q")
    )
    points = (
        alt.Chart(fc)
        .mark_point(color="#3DCD58", filled=True, size=70)
        .encode(x="target_month:T", y="p50:Q")
    )
    rule = (
        alt.Chart(pd.DataFrame({"y": [100]}))
        .mark_rule(color="#999999", strokeDash=[4, 4])
        .encode(y="y:Q")
    )
    return (line + band + points + rule).properties(height=340)


def material_chart(forecast, series_id):
    """Build the material forecast chart for one series."""
    block = forecast.loc[
        (forecast.target_type == "material") & (forecast.target_id == series_id)
    ].copy()
    block["target_month"] = pd.to_datetime(block["target_month"] + "-01")
    band = (
        alt.Chart(block)
        .mark_area(opacity=0.18, color="#3DCD58")
        .encode(x="target_month:T", y="p10:Q", y2="p90:Q")
    )
    line = (
        alt.Chart(block)
        .mark_line(point=True, color="#0C3027")
        .encode(
            x=alt.X("target_month:T", title="Target month"),
            y=alt.Y("p50:Q", title=block.unit.iloc[0]),
        )
    )
    return (band + line).properties(height=300)


def _recommendation_row(data, product: str, horizon: int):
    """Return the offer-matrix recommendation for a product/horizon, if present."""
    rec = data["recommendation"]
    if rec is None:
        return None
    label = PRODUCT_LABELS[product]
    block = rec.loc[(rec["product"] == label) & (rec["offer_horizon_m"] == horizon)]
    return None if block.empty else block.iloc[0]


def _recommended_terms(data, product: str, horizon: int):
    """Return (validity, trigger, indexed material, net adjustment)."""
    row = _recommendation_row(data, product, horizon)
    if row is None:
        return "1 month", "—", "—", None
    return (
        f"{int(row.validity_m)} month",
        f"{row.trigger_pct:.1f}%",
        str(row.indexed_material).replace("_", " "),
        float(row.quote_adj_pct) / 100,
    )


def sidebar(data):
    """Render the assumption controls and return their values."""
    with st.sidebar:
        st.header("Assumptions")
        product = st.selectbox(
            "Product",
            list(PRODUCT_LABELS),
            format_func=lambda key: PRODUCT_LABELS[key],
        )
        horizon = st.select_slider("Horizon (months)", options=[3, 6, 9], value=9)
        base_quote = st.number_input(
            "Base quote (EUR)",
            min_value=1_000,
            value=DEFAULT_QUOTE[product],
            step=500,
            help="Illustrative list price. Replace with the live offer value.",
        )
        material_share = st.slider(
            "Material share of price",
            min_value=0.10,
            max_value=0.95,
            value=DEFAULT_MATERIAL_SHARE,
            step=0.05,
            help="Working assumption. This is not the material share of should-cost.",
        )
        buffer_fraction = st.slider(
            "Risk-buffer fraction of P90−P50",
            min_value=0.0,
            max_value=1.0,
            value=DEFAULT_BUFFER,
            step=0.05,
            help="50% is the base case; 100% is the conservative case.",
        )
        target_margin = st.slider(
            "Target gross margin",
            min_value=0.05,
            max_value=0.60,
            value=DEFAULT_MARGIN[product],
            step=0.01,
        )
        competitor = st.number_input(
            "Competitor price (EUR)",
            min_value=0,
            value=COMPETITOR_REFERENCE if product.startswith("RM6") else 0,
            step=500,
            help=(
                "Synthetic RM6 reference (median of Siemens, ABB, Eaton). "
                "Set 0 to hide the comparison; there is no AirSeT benchmark."
            ),
        )
        st.divider()
        st.caption(
            "Working assumptions, not Schneider policy. The BOM index is partial; "
            "P90 is empirical, not a guaranteed ceiling."
        )
    return {
        "product": product,
        "horizon": horizon,
        "base_quote": base_quote,
        "material_share": material_share,
        "buffer_fraction": buffer_fraction,
        "target_margin": target_margin,
        "competitor": competitor,
    }


def executive_tab(data):
    """Render the one-screen executive summary that mirrors slide 15."""
    st.subheader("Executive summary")
    st.markdown(
        "**Fixed-price ETO offers are exposed to raw-material moves. "
        "We built the missing link from price forecast to quote.**"
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("LOCF retained", "30 / 36", "frozen selections")
    c2.metric("9m copper above P90", "7 / 9", "observed origins")
    c3.metric("AirSeT premium", "19% → 5%", "since 2021")
    c4.metric("BOM mass covered", "92.7% / 85.9%", "RM6 / AirSeT")

    left, right = st.columns([1.05, 1])
    with left:
        st.markdown(
            """
            **What we built**

            - A 42-row forecast pack: 12 materials plus 2 product cost indices at
              3, 6 and 9 months.
            - A partial monetary BOM index (September 2026 = 100) that prices
              copper, aluminium, steel and epoxy.
            - An offer rule with a risk buffer, validity and indexation trigger.
            - A Streamlit calculator and a strategy for voluntary SF6-free adoption.
            """
        )
        st.markdown(
            """
            **What the evidence says**

            - Complexity did not beat LOCF: 30 of 36 frozen choices stayed with
              the naive benchmark.
            - Average interval coverage looks acceptable, but copper's upper tail
              is under-covered at nine months.
            - Copper is 54% of RM6's priced material cost; the AirSeT premium has
              narrowed, but a nine-month fixed price remains commercially fragile.
            """
        )
    with right:
        st.markdown(
            """
            **Recommendation**

            1. Lead with indexation, not a long fixed-price validity.
            2. Use product-specific buffers: +3.9% to +14.7% across horizons.
            3. Target US, Middle East and APAC ex-China data centres.
            4. Pilot two lighthouse accounts; close the four evidence gaps.
            """
        )
        st.markdown(
            '<div class="callout"><b>Bottom line:</b> the forecast is not the '
            "answer by itself. The pack links it to a cost index, a quote rule "
            "and a market strategy. The next step is a controlled pilot, not "
            "more modelling.</div>",
            unsafe_allow_html=True,
        )
    st.caption(
        "Working assumptions: 80% material share of price, 50% of the P90−P50 gap, "
        "1-month base validity. Competitor prices are synthetic; the BOM index is partial."
    )


def pricing_tab(data, inputs):
    """Render the quote calculator and the offer-matrix recommendation."""
    product = inputs["product"]
    horizon = inputs["horizon"]
    base_quote = inputs["base_quote"]
    share = inputs["material_share"]
    buffer_fraction = inputs["buffer_fraction"]
    target_margin = inputs["target_margin"]

    bom = data["forecast"].loc[data["forecast"].target_type == "bom_index"]
    selected = bom.loc[(bom.target_id == product) & (bom.horizon_m == horizon)].iloc[0]
    validity, trigger, indexed, net_adjustment = _recommended_terms(
        data, product, horizon
    )

    gross_adjustment = share * (selected.p50 / 100 - 1) + (
        share * buffer_fraction * (selected.p90 - selected.p50) / 100
    )
    # At the pack defaults, show the published offer-matrix result (net of the
    # indexation clause). If the user changes a slider, show the transparent
    # gross what-if instead of silently mixing a custom share with a fixed net.
    custom = not (
        abs(share - DEFAULT_MATERIAL_SHARE) < 1e-9
        and abs(buffer_fraction - DEFAULT_BUFFER) < 1e-9
    )
    adjustment = (
        net_adjustment
        if (not custom and net_adjustment is not None)
        else gross_adjustment
    )
    quote = base_quote * (1 + adjustment)
    risk_buffer = (
        base_quote * share * buffer_fraction * (selected.p90 - selected.p50) / 100
    )

    st.subheader("Pricing & quote rule")
    st.caption(
        "The recommended adjustment comes from the offer matrix (indexation "
        "included). The sliders let you test a different assumption."
    )
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("BOM P90", f"{selected.p90:.1f}", f"{selected.p90 - 100:+.1f}")
    c2.metric("Gross buffer", f"{gross_adjustment:+.1%}", "before clause")
    c3.metric(
        "Scenario adjustment" if custom else "Net adjustment",
        f"{adjustment:+.1%}",
        "gross what-if" if custom else "after clause",
    )
    c4.metric("Recommended quote", f"€{quote:,.0f}", f"€{quote - base_quote:+,.0f}")
    c5.metric("Risk buffer", f"€{risk_buffer:,.0f}", "P90−P50 share")
    st.caption(
        f"Base validity **{validity}**, indexed on **{indexed}**, trigger **{trigger}**. "
        "The 1-month validity is the model output under a 10%-of-margin tolerance. "
        "At the pack defaults the quote uses the net adjustment; changing a slider "
        "shows the gross what-if before the clause."
    )

    if inputs["competitor"] > 0:
        gap = (quote - inputs["competitor"]) / inputs["competitor"]
        st.info(
            f"At these assumptions the quote is **{gap:+.1%}** versus the "
            f"competitor price of €{inputs['competitor']:,.0f}. "
            "This is the raw-material adjustment only; the deck's "
            "+4%/+7%/+14% gap also includes the 0.5–1.4pp supplier buffer. "
            "The RM6 reference is synthetic; do not treat it as a realised quote."
        )

    rows = []
    for h in (3, 6, 9):
        row = bom.loc[(bom.target_id == product) & (bom.horizon_m == h)].iloc[0]
        _, _, _, net = _recommended_terms(data, product, h)
        gross = share * (row.p50 / 100 - 1) + (
            share * buffer_fraction * (row.p90 - row.p50) / 100
        )
        adj = net if (not custom and net is not None) else gross
        rows.append(
            {
                "Horizon": h,
                "Target": row.target_month,
                "BOM P50": round(row.p50, 1),
                "BOM P90": round(row.p90, 1),
                "Gross buffer": gross,
                "Net adjustment": adj,
                "Quote": base_quote * (1 + adj),
            }
        )
    table = pd.DataFrame(rows)
    st.dataframe(
        table.style.format(
            {
                "BOM P50": "{:.1f}",
                "BOM P90": "{:.1f}",
                "Gross buffer": "{:+.1%}",
                "Net adjustment": "{:+.1%}",
                "Quote": "€{:,.0f}",
            }
        ),
        hide_index=True,
        width="stretch",
    )

    if data["scenarios"] is not None:
        label = PRODUCT_LABELS[product]
        scen = data["scenarios"].loc[
            (data["scenarios"]["product"] == label)
            & (data["scenarios"]["offer_horizon_m"] == horizon)
        ]
        if not scen.empty:
            st.markdown("**Margin after the clause, by raw-material scenario**")
            st.dataframe(
                scen[
                    [
                        "scenario",
                        "bom_index",
                        "cost_impact_pct",
                        "clause_triggered",
                        "margin_after_pct",
                    ]
                ].rename(
                    columns={
                        "scenario": "Scenario",
                        "bom_index": "BOM index",
                        "cost_impact_pct": "Cost impact %",
                        "clause_triggered": "Clause triggered",
                        "margin_after_pct": "Margin after %",
                    }
                ),
                hide_index=True,
                width="stretch",
            )

    st.download_button(
        "Download this scenario",
        table.to_csv(index=False).encode("utf-8"),
        file_name=f"quote_scenario_{product}_{horizon}.csv",
        mime="text/csv",
    )
    st.caption(
        f"Target gross margin: {target_margin:.0%}. The BOM index is material "
        "cost only, not full COGS; margin is shown for context."
    )


def cost_tab(data, product):
    """Render the BOM cost and material forecast charts."""
    forecast = data["forecast"]
    st.subheader("Cost index and material forecasts")
    st.altair_chart(bom_chart(data["history"], forecast, product), width="stretch")
    bom = forecast.loc[
        (forecast.target_type == "bom_index") & (forecast.target_id == product)
    ]
    st.dataframe(
        bom[["horizon_m", "target_month", "p10", "p50", "p90", "unit"]],
        hide_index=True,
        width="stretch",
    )

    st.divider()
    materials = forecast.loc[forecast.target_type == "material"]
    series = st.selectbox("Material series", sorted(materials.target_id.unique()))
    st.altair_chart(material_chart(forecast, series), width="stretch")
    st.dataframe(
        materials.loc[materials.target_id == series][
            ["horizon_m", "target_month", "p10", "p50", "p90", "unit"]
        ],
        hide_index=True,
        width="stretch",
    )
    st.caption(
        "P50 is the last observed price (LOCF); P10/P90 are empirical historical "
        "ranges, not calibrated guarantees."
    )


def strategy_tab():
    """Render the SF6-free strategy that mirrors the market slides."""
    st.subheader("SF6-free strategy: where and how to win voluntary adoption")
    st.caption(
        "Hypothesis-led. The available data contain no market-size, customer-"
        "preference or realised-margin evidence; validate before scaling."
    )
    where = pd.DataFrame(
        [
            {
                "Priority": "Core 1",
                "Market": "United States",
                "Why": "Large data-centre pipeline; no EU-style SF6 ban; ESG procurement.",
                "Check": "State rules, utility acceptance, local competitor prices.",
            },
            {
                "Priority": "Core 2",
                "Market": "Middle East (UAE / Saudi Arabia)",
                "Why": "Data-centre growth; European vendor relationships; voluntary switch.",
                "Check": "Pipeline, local-content rules, competitor presence.",
            },
            {
                "Priority": "Core 3",
                "Market": "APAC ex-China (India / Southeast Asia)",
                "Why": "Growth and price sensitivity; commercial rather than regulatory case.",
                "Check": "Country regulation, price competitiveness, service network.",
            },
            {
                "Priority": "Reference",
                "Market": "EU / UK",
                "Why": "Regulation-led baseline.",
                "Check": "Do not use as the voluntary-switch case.",
            },
            {
                "Priority": "Watch",
                "Market": "China",
                "Why": "Large but different competition and industrial policy.",
                "Check": "Separate partner/localisation case.",
            },
        ]
    )
    st.dataframe(where, hide_index=True, width="stretch")

    left, right = st.columns(2)
    with left:
        st.markdown(
            """
            **Why customers switch**

            - **TCO:** avoid SF6 handling, leak monitoring, recovery and disposal.
            - **ESG:** SF6 has a very high global-warming potential; avoiding it
              supports customer reporting and procurement criteria.
            - **Regulatory optionality:** already compliant if local rules tighten.
            - **Raw-material outlook:** the AirSeT premium has narrowed; the index
              shows whether it is widening again.
            """
        )
    with right:
        st.markdown(
            """
            **How to quote**

            - Value-based pricing: justify the premium on TCO and risk, not a
              discount.
            - Index the main cost driver: copper for RM6, carbon steel for AirSeT.
            - Keep a short base validity and a documented trigger.
            - Add a supplier clause for AirSeT copper exposure.
            - Do not match a nine-month fixed price below the margin floor.
            """
        )

    st.markdown("**Roadmap**")
    st.markdown(
        """
        - **0–3 months:** confirm the product mapping, pick two lighthouse accounts,
          run the rule on live quotes, confirm steel licences and material share.
        - **3–12 months:** build two or three TCO/ESG reference cases, train sales,
          track win rate, discount depth and margin at the competitor price.
        - **12+ months:** extend the index to more materials and regions,
          recalibrate the supplier overlay and the buffer with realised outcomes.
        """
    )
    st.markdown(
        '<div class="warn"><b>Evidence gaps:</b> no market-size or willingness-to-pay '
        "data, no validated SM6/SM AirSeT cost mapping, no realised quote/margin "
        "data, synthetic competitor prices and a provisional RM6 teardown.</div>",
        unsafe_allow_html=True,
    )


def evidence_tab(data):
    """Render coverage, caveats and downloads."""
    st.subheader("Coverage, caveats and downloads")
    st.dataframe(data["coverage"], hide_index=True, width="stretch")
    st.markdown(
        """
        **What this tool does not do**

        - It does not forecast full COGS; the BOM index is partial.
        - It does not prove competitor prices are comparable.
        - It does not establish realised margin erosion.
        - It does not replace approval by sales, procurement or finance.
        - It does not make the P90 a guaranteed ceiling.
        """
    )
    st.download_button(
        "Download the full submission CSV",
        (DATA / "team_02_forecast_202609.csv").read_bytes(),
        file_name="team_02_forecast_202609.csv",
        mime="text/csv",
    )
    with st.expander("Method in one paragraph"):
        st.write(
            "The point forecast is last observation carried forward (LOCF). "
            "P10/P90 come from centred empirical distributions of historical "
            "cumulative log changes. The BOM index is a partial monetary cost "
            "index with September 2026 = 100, covering copper, aluminium, steel "
            "and epoxy. It covers 92.7% of RM6 mass and 85.9% of AirSeT mass. "
            "The offer rule uses an 80% material share and a 50% tail buffer; "
            "the offer matrix adds a 1-month validity and the indexation trigger. "
            "See documentation/submission_method.md and documentation/offer_rule.md."
        )
    st.caption(
        "Private generated data lives under data/submission and must not be "
        "committed. See documentation/deliverables_guide.md for the handover."
    )


def main():
    """Render the app."""
    style()
    st.title("Raw-material risk → ETO offer")
    st.caption("September 2026 challenge pack · Team 2 · decision support, not policy")
    data = load_data()
    inputs = sidebar(data)

    tabs = st.tabs(
        [
            "Executive summary",
            "Pricing & quote",
            "Cost & forecasts",
            "SF6-free strategy",
            "Evidence & caveats",
        ]
    )
    with tabs[0]:
        executive_tab(data)
    with tabs[1]:
        pricing_tab(data, inputs)
    with tabs[2]:
        cost_tab(data, inputs["product"])
    with tabs[3]:
        strategy_tab()
    with tabs[4]:
        evidence_tab(data)


if __name__ == "__main__":
    main()
