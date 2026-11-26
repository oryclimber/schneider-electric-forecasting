"""Build a self-contained HTML presentation from the challenge pack.

The deck uses the generated CSV/XLSX outputs and embeds the two figures as
base64 so it can be opened or shared as a single file. It is a presentation
aid, not a new analysis.

Usage:
    uv run --locked python scripts/build_strategy_deck.py
"""

from __future__ import annotations

import base64
from pathlib import Path

import pandas as pd

from schneider_electric_forecasting import config


def _image(path: Path) -> str:
    """Return a data URI for a PNG file."""
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _table(frame: pd.DataFrame, columns: list[str]) -> str:
    """Render a compact HTML table."""
    head = "".join(f"<th>{column}</th>" for column in columns)
    body = "".join(
        "<tr>" + "".join(f"<td>{row[column]}</td>" for column in columns) + "</tr>"
        for _, row in frame.iterrows()
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def build():
    """Write the self-contained deck to data/submission."""
    out = config.DATA_DIR / "submission"
    coverage = pd.read_csv(out / "bom_coverage.csv")
    scenarios = pd.read_csv(out / "offer_rule_scenarios.csv")
    bom_image = _image(out / "bom_cost_index.png")
    material_image = _image(out / "material_forecasts.png")

    offer = scenarios.copy()
    offer["recommended_price_adjustment_pct"] = offer[
        "recommended_price_adjustment_pct"
    ].map(lambda value: f"+{value:.1f}%")
    offer["bom_p90_index"] = offer["bom_p90_index"].map(lambda value: f"{value:.0f}")
    offer = offer.rename(
        columns={
            "product_ref": "Product",
            "horizon_m": "Horizon",
            "target_month": "Target",
            "bom_p90_index": "P90 index",
            "recommended_price_adjustment_pct": "Adjustment",
        }
    )
    offer_table = _table(
        offer, ["Product", "Horizon", "Target", "P90 index", "Adjustment"]
    )

    coverage_table = _table(
        coverage.assign(
            coverage_pct=coverage.coverage_pct.map(lambda value: f"{value:.1f}%")
        ).rename(
            columns={
                "product_ref": "Product",
                "coverage_pct": "Mass coverage",
                "priced_mass_kg": "Priced kg",
                "total_mass_kg": "Total kg",
            }
        ),
        ["Product", "Mass coverage", "Priced kg", "Total kg"],
    )

    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Schneider Electric challenge — Team 2</title>
<style>
  :root {{ --green:#3DCD58; --ink:#0C3027; --cream:#F4F1E8; --muted:#6D7F76; }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; background:var(--ink); color:var(--ink);
         font-family:Helvetica,Arial,sans-serif; }}
  .slide {{ display:none; width:100vw; height:100vh; padding:6vh 7vw;
            background:var(--cream); overflow:hidden; }}
  .slide.active {{ display:flex; flex-direction:column; }}
  .slide.dark {{ background:var(--ink); color:var(--cream); }}
  .kicker {{ color:var(--green); font-size:16px; letter-spacing:2px;
             text-transform:uppercase; font-weight:700; margin-bottom:12px; }}
  h1 {{ font-size:52px; line-height:1.05; margin:0 0 18px; }}
  h2 {{ font-size:36px; line-height:1.1; margin:0 0 18px; }}
  h3 {{ font-size:22px; margin:0 0 8px; }}
  p, li {{ font-size:19px; line-height:1.35; }}
  .sub {{ color:var(--muted); font-size:20px; max-width:900px; }}
  .dark .sub {{ color:#B9C9C0; }}
  .grid {{ display:grid; gap:18px; margin-top:22px; }}
  .g3 {{ grid-template-columns:repeat(3,1fr); }}
  .g4 {{ grid-template-columns:repeat(4,1fr); }}
  .card {{ background:#fff; border-top:6px solid var(--green); padding:18px;
           box-shadow:0 2px 0 rgba(12,48,39,.08); }}
  .dark .card {{ background:#123B31; color:var(--cream); }}
  .card p {{ font-size:16px; margin:6px 0 0; }}
  .big {{ font-size:44px; font-weight:700; color:var(--green); }}
  table {{ width:100%; border-collapse:collapse; margin-top:14px;
           font-size:16px; background:#fff; }}
  th {{ text-align:left; background:var(--ink); color:var(--cream);
        padding:8px 10px; }}
  td {{ padding:7px 10px; border-bottom:1px solid #DDE5DF; }}
  .dark table {{ background:#123B31; }}
  .dark td {{ border-color:#2A574A; }}
  img {{ max-width:100%; max-height:58vh; object-fit:contain;
         background:#fff; }}
  .foot {{ margin-top:auto; color:var(--muted); font-size:13px; }}
  .dark .foot {{ color:#8FA79B; }}
  .nav {{ position:fixed; bottom:12px; right:16px; color:#8FA79B;
          font-size:13px; user-select:none; }}
  .bar {{ position:fixed; left:0; bottom:0; height:4px; background:var(--green);
          width:0; transition:width .2s; }}
  @media print {{
    .slide {{ display:block !important; page-break-after:always;
              width:13.333in; height:7.5in; }}
    .nav, .bar {{ display:none; }}
    @page {{ size:13.333in 7.5in; margin:0; }}
  }}
</style>
</head>
<body>

<section class="slide active dark">
  <div class="kicker">X-HEC × Schneider Electric · Team 2</div>
  <h1>From raw-material risk to a winning SF6-free offer</h1>
  <p class="sub">A September 2026 forecasting engine, a BOM cost index and a
  quote rule that protects fixed-price ETO margins.</p>
  <div class="grid g3">
    <div class="card"><div class="big">12</div><p>priceable materials</p></div>
    <div class="card"><div class="big">92.7%</div><p>RM6 BOM mass priced</p></div>
    <div class="card"><div class="big">+7.7%</div><p>9-month RM6 risk adjustment</p></div>
  </div>
  <div class="foot">Decision support, not Schneider Electric policy. September 2026 origin.</div>
</section>

<section class="slide">
  <div class="kicker">The ask</div>
  <h2>One engine, three answers</h2>
  <div class="grid g3">
    <div class="card"><h3>A · Forecast</h3><p>Project every priceable material
    at 3, 6 and 9 months, with P10/P50/P90 and the BOM cost index for RM6 and
    RM AirSeT.</p></div>
    <div class="card"><h3>B · Offer rule</h3><p>Translate the forecast into a
    quote adjustment, a risk buffer, a validity period and an indexation
    trigger.</p></div>
    <div class="card"><h3>C · Strategy</h3><p>Define the markets, segments and
    value proposition that make customers switch to SF6-free where regulation
    does not force them.</p></div>
  </div>
</section>

<section class="slide">
  <div class="kicker">What we deliver</div>
  <h2>Four linked artifacts</h2>
  <div class="grid g4">
    <div class="card"><h3>Forecast</h3><p>42 challenge rows: 36 material,
    6 BOM index. September 2026 origin.</p></div>
    <div class="card"><h3>BOM index</h3><p>Monetary, not mass-weighted.
    Copper, aluminium, steel and epoxy.</p></div>
    <div class="card"><h3>Offer rule</h3><p>Adjustment by horizon, 50% tail
    buffer, 3-month validity, 5% trigger.</p></div>
    <div class="card"><h3>Strategy</h3><p>US, Middle East and APAC ex-China;
    data centres first.</p></div>
  </div>
</section>

<section class="slide">
  <div class="kicker">Forecast approach</div>
  <h2>LOCF median, empirical uncertainty</h2>
  <p class="sub">The last observed price is the point forecast. P10/P90 come
  from centred historical cumulative changes over the same horizon. This is
  deliberate: the Phase 3 backtest found LOCF hard to beat.</p>
  <div class="grid g3">
    <div class="card"><div class="big">36</div><p>material forecast rows</p></div>
    <div class="card"><div class="big">5,000</div><p>empirical draws per horizon</p></div>
    <div class="card"><div class="big">3/6/9</div><p>months to Dec 2026, Mar 2027, Jun 2027</p></div>
  </div>
  <div class="foot">Intervals are scenario ranges, not calibrated guarantees.</div>
</section>

<section class="slide">
  <div class="kicker">Forecast output</div>
  <h2>All priceable materials</h2>
  <img src="{material_image}" alt="Material forecasts">
  <div class="foot">P50 is LOCF; shaded band is P10-P90. Units stay in source units.</div>
</section>

<section class="slide">
  <div class="kicker">BOM cost index</div>
  <h2>RM6 and RM AirSeT, September 2026 = 100</h2>
  <img src="{bom_image}" alt="BOM cost index">
  <div class="foot">Partial monetary index. Plastics, brass and unidentified rows are excluded.</div>
</section>

<section class="slide">
  <div class="kicker">Why monetary, not mass</div>
  <h2>Mass would misstate the exposure</h2>
  <div class="grid g3">
    <div class="card"><div class="big">12.5%</div><p>Copper share of RM6 mass</p></div>
    <div class="card"><div class="big">~50%</div><p>Copper share of priced material cost</p></div>
    <div class="card"><div class="big">46.6%</div><p>Carbon steel share of RM6 mass</p></div>
  </div>
  {coverage_table}
  <div class="foot">Coverage is by mass; cost coverage requires approved weights.</div>
</section>

<section class="slide">
  <div class="kicker">Offer rule</div>
  <h2>Recover half the adverse tail</h2>
  <p class="sub">adjustment = material share × (P50/100 − 1) + material share ×
  50% × (P90 − P50)/100</p>
  {offer_table}
  <div class="foot">Material share: 80% RM6 and AirSeT. Working assumptions.</div>
</section>

<section class="slide">
  <div class="kicker">Worked example</div>
  <h2>€10,000 RM6 quote, June 2027</h2>
  <div class="grid g3">
    <div class="card"><div class="big">€767</div><p>material risk adjustment</p></div>
    <div class="card"><div class="big">3 mo</div><p>quote validity</p></div>
    <div class="card"><div class="big">5%</div><p>indexation trigger</p></div>
  </div>
  <p class="sub">If matching a competitor falls below the conservative margin
  floor (25% RM6, 20% AirSeT), do not match without changing scope or terms.</p>
</section>

<section class="slide">
  <div class="kicker">Strategy</div>
  <h2>Where to play</h2>
  <div class="grid g3">
    <div class="card"><h3>United States</h3><p>Large data-centre pipeline,
    no EU-style SF6 ban, ESG procurement.</p></div>
    <div class="card"><h3>Middle East</h3><p>UAE/Saudi growth, European vendor
    relationships, voluntary switch.</p></div>
    <div class="card"><h3>APAC ex-China</h3><p>India/SE Asia growth and price
    sensitivity; commercial case.</p></div>
  </div>
  <p class="sub">EU/UK is the regulation-led reference; China is a separate
  watch case. Segment: data centres first.</p>
</section>

<section class="slide">
  <div class="kicker">Competitive position</div>
  <h2>Not the cheapest — win on TCO and risk</h2>
  <table>
    <thead><tr><th>Brand</th><th>2026 P50 (synthetic)</th><th>Position</th></tr></thead>
    <tbody>
      <tr><td>Siemens</td><td>13,762</td><td>Highest</td></tr>
      <tr><td>ABB</td><td>12,273</td><td>High</td></tr>
      <tr><td>Schneider Electric</td><td>12,233</td><td>Mid-high</td></tr>
      <tr><td>Eaton</td><td>11,560</td><td>Mid</td></tr>
      <tr><td>Ormazabal</td><td>9,652</td><td>Lower</td></tr>
      <tr><td>SwitchGear Company</td><td>9,154</td><td>Lowest</td></tr>
    </tbody>
  </table>
  <div class="foot">Synthetic benchmark. Use for relative positioning, not as evidence of actual competitor prices.</div>
</section>

<section class="slide">
  <div class="kicker">Roadmap</div>
  <h2>From analysis to a repeatable offer</h2>
  <div class="grid g3">
    <div class="card"><h3>0-3 months</h3><p>Confirm product mapping, steel
    licence and material share; run the rule on live quotes.</p></div>
    <div class="card"><h3>3-12 months</h3><p>Two or three TCO/ESG reference
    cases; train sales; track win rate and margin.</p></div>
    <div class="card"><h3>12+ months</h3><p>Extend the index and add supplier
    risk to the quote review.</p></div>
  </div>
</section>

<section class="slide dark">
  <div class="kicker">What must be confirmed</div>
  <h2>Five decisions, not five footnotes</h2>
  <ol>
    <li>September 2026 cutoff and October/November exclusion.</li>
    <li>Permission to use the steel index workbook.</li>
    <li>Target margin and cost-risk tolerance.</li>
    <li>RM6/RM AirSeT to SM6/SM AirSeT mapping.</li>
    <li>Three priority markets and the official team ID.</li>
  </ol>
  <p class="sub">The forecast is not the answer. The rule, the coverage disclosure
  and the market choice are what make it usable.</p>
  <div class="foot">Team 2 · September 2026 challenge pack</div>
</section>

<div class="nav">← → to navigate · <span id="count"></span></div>
<div class="bar" id="bar"></div>
<script>
  const slides = [...document.querySelectorAll('.slide')];
  let index = 0;
  function show(next) {{
    index = Math.max(0, Math.min(slides.length - 1, next));
    slides.forEach((slide, i) => slide.classList.toggle('active', i === index));
    document.getElementById('count').textContent = `${{index + 1}} / ${{slides.length}}`;
    document.getElementById('bar').style.width = `${{100 * (index + 1) / slides.length}}%`;
  }}
  document.addEventListener('keydown', (event) => {{
    if (event.key === 'ArrowRight' || event.key === ' ') show(index + 1);
    if (event.key === 'ArrowLeft') show(index - 1);
  }});
  show(0);
</script>
</body>
</html>
"""
    path = out / "strategy_deck.html"
    path.write_text(html, encoding="utf-8")
    print(path)
    return path


if __name__ == "__main__":
    build()
