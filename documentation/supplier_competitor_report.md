# Supplier risk, product risk and competitive position: findings

Scope: how supplier counterparty risk and competitor pricing change the offer
terms we propose for RM6 and RM AirSeT. All numbers come from the CSVs in
`data/submission/`, produced by `scripts/supplier_risk.py`,
`supplier_product_risk.py`, `offer_matrix.py`, `supplier_offer_terms.py`,
`competitor_pricing.py` and `competitive_risk_view.py`.

**Evidence status.** These are scenario-based decision-support results, not
validated pricing. Supplier-to-material links, rating cut-offs, the 80% material
share of price, the loss tolerance and the 30% / 25% target margins are working
assumptions. Competitor prices are synthetic.

## 1. Summary

1. **Supply-base risk is concentrated, not broad.** Most suppliers rate low, but
   the weakest supplier on copper (Gindre Duchavany) and on steel (Groupe Aim)
   are rated high. Copper is 54% of RM6 priced cost, so one fragile supplier
   matters a lot.
2. **Both products have a medium base risk but a high downside risk**
   (RM6 1.68 base / 2.73 downside; AirSeT 1.77 / 2.80 on a 1-3 scale). In the
   downside case, 86% (RM6) and 90% (AirSeT) of priced cost sits with a
   high-risk supplier.
3. **Supplier risk adds a small price buffer** (0.5-1.5pp) and shortens the
   safe validity window. It does not drive our competitiveness.
4. **The raw-material volatility adjustment drives our price gap.** With it,
   the RM6 quote is 4% above the reference competitor at 3 months and 14%
   above at 9 months.
5. **A fixed-price validity is effectively not offerable** under the current
   tolerance: even a 1-month window breaches it. This is an assumption to
   confirm with Schneider, not a finding about the market.

## 2. Supplier risk scorecard

Source: PitchBook balance-sheet files, latest fiscal year with data. Each of five
indicators scores 0 (low), 1 (medium) or 2 (high risk). The rating is the mean:
below 0.67 low, below 1.33 medium, otherwise high. Equal weights and all
cut-offs are judgment values, not calibrated to defaults.

| Supplier | Year | Liabilities / equity | Current ratio | EBITDA / assets | Interest cover | Score | Rating |
|---|---|---|---|---|---|---|---|
| Electris | 2024 | 0.8 | 1.77 | 19% | 24.1 | 0.00 | low |
| MSS International | 2025 | 1.3 | 1.73 | 12% | n/a | 0.00 | low |
| Iltom | 2025 | 0.9 | 1.81 | 15% | 26.0 | 0.20 | low |
| KUVAG | 2025 | 2.6 | n/a | 7% | 4.0 | 0.50 | low |
| Maintenance Travaux et Câblage | 2024 | 1.7 | 1.12 | 7% | 21.7 | 0.60 | low |
| Rötzer Maschinenbau | 2024 | 2.0 | 1.11 | n/a | n/a | 1.00 | medium |
| Gindre Duchavany | 2024 | 7.2 | 1.31 | 1% | 0.1 | 1.40 | **high** |
| Groupe Aim | 2025 | 1.6 | 0.84 | -2% (loss) | -1.1 | 1.60 | **high** |
| Green Transfo | 2023 | 34.3 | 0.81 | 3% | 0.8 | 1.75 | **high** |

Data caveats:
- Rötzer has only two indicators available; its medium rating is thin.
- KUVAG has no current-ratio data; Green Transfo's latest data is 2023.
- Two suppliers (Green Transfo, Maintenance Travaux et Câblage) make no
  relevant material per public descriptions, so they are not mapped to any
  material and do not enter product risk.

## 3. Supply-base risk by material

Suppliers of the same material are weighted equally (no volume data exists).
Base risk is the mean rating of a material's suppliers (1 = low, 3 = high).
Downside is the weakest supplier.

| Material | Suppliers (rating) | Base | Downside | Weakest |
|---|---|---|---|---|
| Carbon / stainless / galvanised steel | Groupe Aim (high), Iltom (low), Rötzer (medium) | 2.00 medium | 3 high | Groupe Aim |
| Copper | Gindre Duchavany (high), MSS (low), Electris (low) | 1.67 low | 3 high | Gindre Duchavany |
| Aluminium | Electris (low) | 1.00 low | 1 low | Electris |
| Epoxy | KUVAG (low) | 1.00 low | 1 low | KUVAG |

The three steel grades share one supplier set because none of the fabricators
states which grade it handles.

## 4. Product risk

| | RM6 | RM AirSeT |
|---|---|---|
| Copper share of priced cost | 54.2% | 38.0% |
| Steel share (all grades) | 32.1% | 51.9% |
| Epoxy share | 12.1% | 8.5% |
| Aluminium share | 1.6% | 1.5% |
| Base supply-base risk | 1.68 (medium) | 1.77 (medium) |
| Downside supply-base risk | 2.73 (high) | 2.80 (high) |
| Cost share with a high-risk weakest supplier | 86% | 90% |
| Priced mass coverage of the BOM | 92.7% | 85.9% |

Every priced material has a mapped supplier, so unmapped cost is 0%. RM6 depends
most on one copper supplier. AirSeT depends more on steel, where Groupe Aim is
loss-making.

**Epoxy.** Epoxy resin is in the authoritative BOM (7.2% of RM6 mass, 3.0% of
AirSeT mass), which gives the 12.1% / 8.5% priced-cost shares. What remains
unconfirmed is that KUVAG is Schneider's epoxy supplier: the link is inferred from
its product line, and it is the only epoxy supplier, so epoxy risk rests on it.

## 5. Effect on offer terms

Supplier risk adds a buffer on top of the price-volatility quote and caps
validity. Stress cases hit the flagged supplier's volume (equal split):
price rise of 15%, or failure with re-sourcing 30% above market.

| Product | Horizon | Price-based quote adj. | Supplier buffer | Total adj. | Supplier validity cap | Final validity |
|---|---|---|---|---|---|---|
| RM6 | 3m | 3.9% | 0.46pp | 4.4% | 2m | 1m |
| RM6 | 6m | 6.5% | 0.91pp | 7.4% | 4m | 1m |
| RM6 | 9m | 12.7% | 1.37pp | 14.0% | 7m | 1m |
| AirSeT | 3m | 4.0% | 0.52pp | 4.5% | 2m | 1m |
| AirSeT | 6m | 8.7% | 1.03pp | 9.8% | 4m | 1m |
| AirSeT | 9m | 14.7% | 1.55pp | 16.2% | 7m | 1m |

**Stress results (margin at P90 cost, with buffer, vs target margin).**
The flagged supplier is Gindre Duchavany (copper) for both products.

| Product | Horizon | Supplier raises price | Supplier fails |
|---|---|---|---|
| RM6 (target 30%) | 3m | 28.5% (-1.5pp) | 26.3% (-3.7pp) |
| RM6 | 6m | 27.7% (-2.3pp) | 25.5% (-4.5pp) |
| RM6 | 9m | 28.9% (-1.1pp) | 26.7% (-3.3pp) |
| AirSeT (target 25%) | 3m | 23.9% (-1.1pp) | 22.4% (-2.6pp) |
| AirSeT | 6m | 22.7% (-2.3pp) | 21.1% (-3.9pp) |
| AirSeT | 9m | 21.7% (-3.3pp) | 20.2% (-4.8pp) |

A single supplier failure costs 3-5 points of margin, more than the buffer
recovers. The model flags a supplier clause for AirSeT (copper is not the
price-indexed material there, carbon steel is) but not for RM6, where copper is
already indexed.

## 6. Competitor position (RM6 only)

Prices are synthetic 2026 P50s, held flat over the horizon. The reference is the
median of the comparable rivals (Siemens, ABB, Eaton) = 12,273.

| Brand | Synthetic P50 | vs ours (12,233) | Should-cost total | Note |
|---|---|---|---|---|
| Siemens | 13,762 | +12.5% | 3,307 | observed list price 15,530 (webshop) |
| ABB | 12,273 | +0.3% | 3,462 | |
| Schneider | 12,233 | 0 | 3,009 | |
| Eaton | 11,560 | -5.5% | 767 | should-cost implausibly low |
| Ormazabal | 9,652 | -21.1% | 1,692 | excluded as a low-cost outlier |
| SGC | 9,154 | -25.2% | 19 | should-cost implausible; excluded |

Today we sit level with ABB and below Siemens. Material is 80-86% of should-cost for
every brand except SGC (60%), but the should-cost workbook is provisional, built on
inferred part masses, and should be treated as directional.

## 7. Our quote against competitors

| Offer | Our quote | Gap to reference | of which supplier buffer | GM if we match the reference (P90 cost) | GM if supplier also fails |
|---|---|---|---|---|---|
| 3m | 12,767 | +4.0% | 0.45pp | 21.1% | 16.8% |
| 6m | 13,142 | +7.1% | 0.91pp | 13.9% | 9.5% |
| 9m | 13,950 | +13.7% | 1.36pp | 3.6% | -0.8% |

- Removing the supplier buffer does not close the gap at any horizon. The
  volatility adjustment is the main driver, so the status is
  "uncompetitive before buffer".
- Matching a rival at the 9-month horizon leaves almost no margin, and a supplier
  failure turns it negative.
- Short offers are roughly competitive. A 9-month fixed-price offer is not.

## 8. Why validity is capped at one month, and how sensitive it is

Validity is the longest window whose P90 price impact stays within a tolerance
(by default 10% of target margin = 3.0pp of price for RM6). The P90 impact is
3.8pp at 1 month, 5.4pp at 2 months and 8.8pp at 3 months, so even 1 month breaches
the tolerance; the search floor of 1 month is returned. The supplier cap never
binds under the default.

Sensitivity (RM6, `competitive_risk_sensitivity.csv`):

| Tolerance (share of margin) | Validity: 3m / 6m / 9m offer | Gap to reference: 3m / 6m / 9m |
|---|---|---|
| 10% (base) | 1 / 1 / 1 | +4.0% / +7.1% / +13.7% |
| 25% | 2 / 2 / 2 | +5.5% / +10.0% / +14.6% |
| 50% | 2 / 4 / 5 | +5.5% / +11.2% / +15.9% |
| 100% | 2 / 4 / 7 | +5.5% / +11.2% / +19.2% |

Longer validity costs competitiveness: a looser tolerance raises the indexation
trigger, so less raw-material risk is passed through the clause and more sits in the
quote. Margin at the competitor price does not change with the tolerance (it depends
on cost only).

## 9. Implications and recommendations

1. **Lead with indexation, not fixed-price validity.** Short validity plus an
   indexation clause on the main driver (copper for RM6, carbon steel for AirSeT)
   moves risk into the contract instead of the headline price.
2. **Add a copper supplier clause for AirSeT**, where the weakest-supplier exposure
   (12.7% of cost, Gindre Duchavany) is not covered by the price index.
3. **Consider dual-sourcing or monitoring Gindre Duchavany and Groupe Aim.** Their
   interest cover (0.1x) and loss position are the largest single risks.
4. **Do not price 9-month fixed offers against rivals** without accepting margin
   close to zero; prefer 3-month offers with indexation.
5. **Confirm with Schneider:** the loss tolerance, the 80% material share of
   price, real supplier volume splits, that KUVAG supplies the epoxy, and which steel
   grades each fabricator supplies.

## 10. Limitations

- Competitor P50s are synthetic and the RMU benchmark workbook is
  DRM-protected; P50s are taken from `strategy_pack.md`.
- Competitor data covers RM6 only. AirSeT has no comparable competitor.
- RM6/AirSeT costs are not the same as SM6/SM AirSeT; Schneider must validate the
  mapping.
- Supplier ratings use public financial data, equal indicator weights and judgment
  cut-offs, and are not calibrated to defaults. Supplier links come from public
  web descriptions and user-supplied company descriptions.
- Supplier stress sizes (15% price rise, 30% re-sourcing premium) and the
  buffer and validity parameters are assumptions, not procurement quotes.
- Should-cost totals for SGC and Eaton are implausible and are not used for pricing.
- `documentation/strategy_pack.md` now states the same validity result and its dependence on the tolerance.
