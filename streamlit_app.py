"""Interactive decision tool for the September 2026 challenge pack.

Run with:
    uv run --group app streamlit run streamlit_app.py

The app reads the generated CSV/XLSX artifacts under ``data/submission``. It
does not retrain models or modify any source workbook.
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
DEFAULT_MATERIAL_SHARE = {
    "RM6_NE_IQI_433520": 0.80,
    "RMAirSeT IIQ 24kV IEC": 0.80,
}

st.set_page_config(
    page_title="Schneider challenge — decision tool",
    page_icon="⚡",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def load_data():
    """Load the generated pack tables."""
    forecast = pd.read_csv(DATA / "team_02_forecast_202609.csv")
    history = pd.read_csv(DATA / "bom_cost_index_history.csv", parse_dates=["month"])
    coverage = pd.read_csv(DATA / "bom_coverage.csv")
    scenarios = pd.read_csv(DATA / "offer_rule_scenarios.csv")
    return forecast, history, coverage, scenarios


def style():
    """Apply a light Schneider-inspired theme."""
    st.markdown(
        """
        <style>
        .block-container {padding-top: 1.6rem;}
        h1, h2, h3 {color: #0C3027;}
        div[data-testid="stMetricValue"] {color: #0C3027;}
        .stTabs [data-baseweb="tab"] {font-weight: 600;}
        </style>
        """,
        unsafe_allow_html=True,
    )


def bom_chart(history, forecast, product):
    """Build an interactive BOM history and forecast chart."""
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
    return (line + band + points + rule).properties(height=360)


def material_chart(forecast, series_id):
    """Build an interactive material forecast chart."""
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
    return (band + line).properties(height=320)


def main():
    """Render the app."""
    style()
    st.title("Raw-material risk → ETO offer")
    st.caption("September 2026 challenge pack · Team 2 · decision support, not policy")
    if not (DATA / "team_02_forecast_202609.csv").exists():
        st.error(
            "The pack is missing. Run `python scripts/build_challenge_pack.py` first."
        )
        st.stop()

    forecast, history, coverage, scenarios = load_data()
    latest = forecast.loc[forecast.target_type == "material"]
    bom = forecast.loc[forecast.target_type == "bom_index"]

    with st.sidebar:
        st.header("Assumptions")
        product = st.selectbox(
            "Product",
            list(PRODUCT_LABELS),
            format_func=lambda key: PRODUCT_LABELS[key],
        )
        horizon = st.select_slider("Horizon (months)", options=[3, 6, 9], value=9)
        base_quote = st.number_input(
            "Base quote (EUR)", min_value=1000, value=10_000, step=500
        )
        material_share = st.slider(
            "Material share of price",
            min_value=0.10,
            max_value=0.90,
            value=DEFAULT_MATERIAL_SHARE[product],
            step=0.05,
        )
        buffer_fraction = st.slider(
            "Risk-buffer fraction of P90-P50",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
        )
        target_margin = st.slider(
            "Target gross margin",
            min_value=0.05,
            max_value=0.60,
            value=DEFAULT_MARGIN[product],
            step=0.01,
        )
        competitor = st.number_input(
            "Competitor price (optional, EUR)",
            min_value=0,
            value=0,
            step=500,
            help="Leave at zero to hide the competitive comparison.",
        )
        st.divider()
        st.caption(
            "The BOM index is partial. Plastics, brass and unidentified rows are excluded."
        )

    selected = bom.loc[(bom.target_id == product) & (bom.horizon_m == horizon)].iloc[0]
    adjustment = material_share * (selected.p50 / 100 - 1) + (
        material_share * buffer_fraction * (selected.p90 - selected.p50) / 100
    )
    quote = base_quote * (1 + adjustment)
    risk_buffer = (
        base_quote
        * material_share
        * buffer_fraction
        * (selected.p90 - selected.p50)
        / 100
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("BOM P90 index", f"{selected.p90:.1f}", f"{selected.p90 - 100:+.1f}")
    col2.metric("Price adjustment", f"{adjustment:+.1%}")
    col3.metric("Recommended quote", f"€{quote:,.0f}", f"€{quote - base_quote:+,.0f}")
    col4.metric("Risk buffer", f"€{risk_buffer:,.0f}")
    if competitor > 0:
        gap = (quote - competitor) / competitor
        st.info(
            f"The recommended quote is **{gap:+.1%}** versus the competitor price "
            f"of €{competitor:,.0f}."
        )

    tabs = st.tabs(
        ["BOM index", "Material forecasts", "Quote rule", "Coverage & caveats"]
    )
    with tabs[0]:
        st.subheader(f"{PRODUCT_LABELS[product]} — cost index")
        st.altair_chart(bom_chart(history, forecast, product), width="stretch")
        st.dataframe(
            bom.loc[bom.target_id == product][
                ["horizon_m", "target_month", "p10", "p50", "p90"]
            ],
            hide_index=True,
            width="stretch",
        )
    with tabs[1]:
        series = st.selectbox("Material", sorted(latest.target_id.unique()))
        st.altair_chart(material_chart(forecast, series), width="stretch")
        st.dataframe(
            latest.loc[latest.target_id == series][
                ["horizon_m", "target_month", "p10", "p50", "p90", "unit"]
            ],
            hide_index=True,
            width="stretch",
        )
    with tabs[2]:
        st.subheader("Quote impact at the chosen assumptions")
        rows = []
        for h in (3, 6, 9):
            row = bom.loc[(bom.target_id == product) & (bom.horizon_m == h)].iloc[0]
            adj = material_share * (row.p50 / 100 - 1) + (
                material_share * buffer_fraction * (row.p90 - row.p50) / 100
            )
            rows.append(
                {
                    "Horizon": h,
                    "Target": row.target_month,
                    "BOM P50": round(row.p50, 1),
                    "BOM P90": round(row.p90, 1),
                    "Adjustment": adj,
                    "Quote": base_quote * (1 + adj),
                }
            )
        table = pd.DataFrame(rows)
        st.dataframe(
            table.style.format(
                {
                    "BOM P50": "{:.1f}",
                    "BOM P90": "{:.1f}",
                    "Adjustment": "{:+.1%}",
                    "Quote": "€{:,.0f}",
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
            f"Target margin: {target_margin:.0%}. The margin is shown for context; "
            "the BOM index is material cost only, not full COGS."
        )
    with tabs[3]:
        st.subheader("BOM coverage")
        st.dataframe(coverage, hide_index=True, width="stretch")
        st.markdown(
            """
            **What this tool does not do**

            - It does not forecast full COGS; the BOM index is partial.
            - It does not prove competitor prices are comparable.
            - It does not establish realised margin erosion.
            - It does not replace approval by sales, procurement or finance.
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
            "See documentation/submission_method.md for the full method."
        )


if __name__ == "__main__":
    main()
