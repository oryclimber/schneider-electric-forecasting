# SF6-free strategy pack

Question: in markets where regulation does not force the switch, how does
Schneider Electric turn its SF6-free lead into market share?

This is a hypothesis-led strategy. The available files do not contain market
size, adoption, customer-preference or realised-margin data. Label every market
claim accordingly.

## 1. Where to play

| Priority | Market | Why it is attractive | What must be checked |
|---|---|---|---|
| Core 1 | United States | Large data-centre pipeline, no EU-style SF6 ban, corporate ESG procurement. | State-level rules, utility acceptance, local competitor prices. |
| Core 2 | Middle East (UAE / Saudi Arabia) | Data-centre growth, European vendor relationships, voluntary switch. | Project pipeline, local content rules, competitor presence. |
| Core 3 | APAC ex-China (India / Southeast Asia) | Growth and price sensitivity; commercial rather than regulatory case. | Country-level regulation, price competitiveness, service network. |
| Reference | EU / UK | Regulation-led baseline; shows what mandated adoption looks like. | Do not use as the voluntary-switch case. |
| Watch | China | Large but different competition and industrial-policy context. | Separate partner/localisation case; not a core recommendation. |

**Segment:** data centres first — hyperscale and colocation — then
utilities/infrastructure. The challenge brief identifies data-centre demand as
the main growth driver.

## 2. Why customers switch

The value proposition is not “SF6-free” alone. It is:

1. **Total cost of ownership** — avoided gas handling, leak management,
   end-of-life and compliance costs.
2. **ESG and decarbonisation commitments** — SF6 has a very high global-warming
   potential; avoiding it supports customer reporting and procurement criteria.
3. **Regulatory optionality** — a technology that is already compliant if local
   rules tighten.
4. **Raw-material outlook** — the forecast helps show whether the AirSeT cost
   premium is narrowing or widening.

The offer rule supplies the pricing side of this story: a transparent material
index, a documented risk buffer, a validity window and an indexation trigger
(see section 5 for the current validity result).

## 3. Competitive position

The synthetic RMU benchmark gives 2026 P50 prices for six brands:

| Brand | 2026 P50 (synthetic) | Position |
|---|---:|---|
| Siemens | 13,762 | Highest |
| ABB | 12,273 | High |
| Schneider Electric | 12,233 | Mid-high |
| Eaton | 11,560 | Mid |
| Ormazabal | 9,652 | Lower |
| SwitchGear Company | 9,154 | Lowest |

Implication: Schneider is not the cheapest. The case must be won on TCO, ESG,
SF6 risk and service, not on list price. The benchmark is synthetic; use it for
relative positioning and scenario framing, not as evidence of actual competitor
prices.

The RM6 teardown adds a cost view: Schneider material cost is about €2,404 and
total should-cost about €3,009 for the benchmarked configuration. Competitor
should-costs vary widely and the workbook is marked provisional, so use this to
identify cost gaps and questions for procurement, not to claim a verified cost
advantage.

**Data limitation: teardown part masses are inferred.** Every material
line in the workbook is marked "mass: inferred, total held to the mass budget",
so part masses are estimates, not sourced figures. A few lines drive most of
each brand's material cost and look implausible:

| Brand | Line | Issue |
|---|---|---|
| Schneider | Fasteners 2,500 g x 200 (EUR 648) and packaging charged on a 715 kg BOM (EUR 1,073) | 500 kg of screws; datasheet mass is 275 kg. About 72% of material cost. At datasheet mass, material would be nearer EUR 1,100 (rough estimate). |
| Siemens | HV fuse-links, 700 g silver x 3 at EUR 800/kg (EUR 1,814) | About 68% of material cost. A real fuse-link weighs about that in total. |
| Ormazabal | HV fuse-links, 400 g silver x 3 (EUR 1,037) | About 71% of material cost. Same issue. |
| ABB | Copper busbars, 43,980 g x 3 (EUR 1,354) | 132 kg of copper, about 48% of material cost. |

We use the teardown figures as published, as an assumption, and treat the
should-cost totals, the "price at 30% margin" and the rankings against
competitors (including who is cheaper to build) as directional, not verified. Overhead is a percentage of the other costs, so it inherits the
errors. Labour minutes and process operation counts come from the teardown and
are unverified. Only the price comparison against the synthetic P50s does not
depend on should-cost. We keep the figures unadjusted and state the limitation
instead of correcting them. The same lines are the procurement questions to put
to suppliers and to Schneider: silver content of HV fuse-links, copper busbar
mass, fastener and packaging mass, and the labour and process counts.

**Our quote against competitors (RM6).** `scripts/competitive_risk_view.py` compares
our quote, including the raw-material adjustment and the supplier buffer, with the
median of the comparable rivals (Siemens, ABB, Eaton; 12,273, held flat). The gap is
about +4% at a 3-month offer, +7% at 6 months and +14% at 9 months. The supplier
buffer contributes only 0.5-1.4pp of that; the raw-material adjustment drives it. If
we matched the reference price in a P90 cost case, gross margin would be about 21%,
14% and 4% against a 30% target. Competitors exist for RM6 only. Details:
`documentation/supplier_competitor_report.md`.

## 4. Supply-chain resilience

Nine supplier financial workbooks are available (Electris, Gindre Duchavany,
Green Transfo, Groupe Aim, Iltom, KUVAG, Maintenance Travaux et Câblage,
MSS International, Rötzer Maschinenbau). They contain overview, income
statement, balance sheet, cash flow and ratio sheets.

Implemented in `scripts/supplier_risk.py` (scorecard), `supplier_material_map.py`
(supplier-to-material map, with BOM shares computed from the authoritative BOM),
`supplier_product_risk.py` and `supplier_offer_terms.py`:

- a five-indicator scorecard (leverage, liquidity, profit level, profit trend,
  interest cover) rates three suppliers high (Gindre Duchavany, Groupe Aim,
  Green Transfo), one medium and five low;
- suppliers are linked to priced materials from public and user-supplied
  descriptions (the BOM has no supplier names, so this link is an assumption);
  Green Transfo and Maintenance Travaux et Cablage supply no priced material;
- supply-base risk is medium in the base case and high in the downside case for
  both products (RM6 1.68 / 2.73, AirSeT 1.77 / 2.80 on a 1-3 scale);
- it feeds offer terms (a 0.5-1.5pp price buffer, a validity cap and a supplier
  clause flag) and stress cases, not the price forecast. A single supplier failure
  costs 3-5pp of margin.

Ratings and stress sizes are working assumptions. See
`documentation/supplier_competitor_report.md`.

## 5. Pricing and commercial approach

Use the offer rule documented in `documentation/submission_method.md`, with the
offer matrix from `scripts/offer_matrix.py`.

- RM6 / SM6: 30% target gross margin, 50% tail buffer.
- RM AirSeT / SM AirSeT: 25% target gross margin, 50% tail buffer.
- Indexation on the material with the largest share of BOM cost variance (copper
  for RM6, carbon steel for AirSeT), triggered at about 7% (RM6) and 10% (AirSeT)
  material move.
- Validity: under the default loss tolerance (10% of target margin) the model
  returns the 1-month floor, because even a 1-month window exceeds it. A 25%
  tolerance gives 2 months. This depends on the tolerance and the 80% material
  share, both unconfirmed; do not quote a validity until Schneider confirms them.
- Show the BOM index and base month on the quote.
- If matching a competitor would fall below the conservative margin floor, do
  not match without changing scope or terms.

The AirSeT premium should be presented as a TCO and risk story, not as a
discount requirement.

## 6. Roadmap

**0-3 months**

- Confirm the RM6/RM AirSeT to SM6/SM AirSeT mapping with Schneider.
- Validate the three core markets and pick two lighthouse accounts.
- Run the offer rule on live quotes and record the outcome.
- Confirm steel-index licensing and the material share of price.

**3-12 months**

- Build two or three reference cases with TCO and ESG evidence.
- Train sales on the indexation clause and the risk buffer.
- Track win rate, discount depth and margin at the competitor price.
- Add country-level regulatory and competitor evidence.

**12+ months**

- Extend the index to more materials and regions.
- Calibrate the supplier-risk overlay (already prototyped) with procurement quotes and volume splits.
- Recalibrate the buffer with realised outcomes.

## 7. Win conditions

- A customer accepts the SF6-free premium on TCO/ESG grounds.
- The quote includes a documented material index and indexation trigger.
- The realised margin stays above the conservative floor.
- The same offer rule works in at least two markets.
- Sales can explain the rule without a data scientist in the room.

## 8. Evidence gaps

- No market-size or adoption data.
- No customer-preference or willingness-to-pay study.
- No validated SM6/SM AirSeT cost mapping.
- No realised quotation, sales or margin data.
- Competitor benchmark is synthetic; the RM6 teardown is provisional.
- Supplier-to-material links are inferred from public descriptions, not from Schneider purchasing data; no supplier volume splits exist.

These gaps are the agenda for the next client conversation, not footnotes to
hide.
