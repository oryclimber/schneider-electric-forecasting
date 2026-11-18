# Business Problem Statement

## Status and evidence convention

This statement reflects the X-HEC Data Challenge and the latest discussion with Schneider Electric.

- **Confirmed** describes challenge requirements, client feedback, or locally observed data, not client approval.
- **Proposed** describes methods and decision support to evaluate.
- **Limitation / pending clarification** identifies evidence not available or data not yet validated.

## 1. Business context and problem description

**Central business problem:** How can Schneider Electric accurately forecast future raw-material and product costs to protect gross margins while maintaining competitive pricing and supporting the transition to SF6-free products?

Margin protection is the primary business objective, supported by accurate and reliable material and BOM cost forecasts. Forecast costs and uncertainty should inform actionable ETO pricing recommendations while balancing target profitability and commercial competitiveness. The project also supports a commercial strategy for SF6-free adoption.

**Product distinction.** Available cost inputs refer to RM6 and RM AirSeT configurations; the strategy challenge refers to SM6 and SM AirSeT. Do not assume these products or costs are interchangeable. Schneider Electric must validate product and configuration mappings.

Forecasts support decisions; they are not automatic prices or proof of customer adoption, savings, or margin improvement.

## 2. Business decisions supported

### ETO pricing and margin protection

Subject to validated costs, forecasts and scenarios may inform quotation prices, validity periods, indexation triggers, cost-risk buffers, and the trade-off between target margins and competitive selling prices for supported configurations.

The synthetic RMU benchmark and provisional RM6 competitor/teardown benchmark may inform competitive comparisons. Verify comparability, price meaning, provenance, and permitted use first. Recommendations are scenario-based decision support, not empirically validated margin optimization; approved prices and contract terms remain business decisions.

Accurate and reliable forecasts are essential to effective margin protection. Given an estimated total cost of goods sold (COGS) and a target gross margin (as a fraction), proposed decision-support calculations are:

- **Target quotation price:** `Quotation_price = Forecasted_COGS / (1 - Target_gross_margin)`. This estimates the price required to achieve the target margin at forecasted cost.
- **Competitive price gap:** `Price_gap_pct = 100 * (Quotation_price - Competitor_price) / Competitor_price`. This assesses whether the margin-protecting quotation is competitive against a comparable offer.
- **Gross margin at competitor price:** `GM_at_competitor_price = (Competitor_price - Forecasted_COGS) / Competitor_price`. This indicates whether matching the competitor price would meet the target gross margin.
- **Risk-adjusted quotation:** `Risk_buffer = P90_COGS - P50_COGS`; `Risk_adjusted_quotation = (P50_COGS + Risk_buffer) / (1 - Target_gross_margin)`. This illustrates how cost uncertainty could inform a risk-adjusted quotation strategy.

These calculations require **estimated total COGS**, not only partial BOM costs or mass-weighted proxy indices. Product-level cost quantiles must account for dependencies between material prices. Competitor prices must be comparable in product configuration, market, currency, and commercial conditions. These are proposed decision-support mechanisms, not validated Schneider Electric pricing policies.

### SF6-free adoption

**Priority markets from client discussion:** European Union (including the regulation-driven transition), United Kingdom and broader Europe, United States, and China. Focus the initial assessment on these markets rather than an exhaustive global analysis.

Assess regulation, competitive positioning, product economics, TCO, and customer value propositions. Verify market-specific regulatory requirements before making recommendations.

**Limitation.** Current inputs do not establish adoption rates, customer preferences, market eligibility, complete product TCO, or realized outcomes. Distinguish evidence-based recommendations from unvalidated adoption hypotheses.

## 3. Business baseline and forecasting methodology

**Limitation.** Available data do not include ETO quote history, current pricing policy, selling prices, realized order costs, or gross-margin and adoption outcomes. Current performance and actual improvement over existing practice therefore cannot yet be quantified.

**Proposed methodology.** Experiment with suitable forecasting methods rather than preselecting a final model. Compare candidates with a transparent naive baseline: last observation carried forward (LOCF) for each validated series, holding its latest eligible observation constant for 3, 6, and 9 months. Use derived series only when their inputs were available at origin. For BOM, LOCF holds the latest eligible cost estimate or proxy index; it does not make an incomplete BOM complete.

Conduct rigorous rolling-origin, time-ordered backtesting for all three horizons. Each fold uses only information available at its cutoff and consistent targets and availability rules. Evaluate point accuracy and uncertainty calibration. Explain and reproduce the forecasting, evaluation, and validation process; assess process quality, not just final predicted values.

**Proposed inflation experiment.** Where an appropriate and permitted inflation series is available, compare nominal-price forecasting, inflation-adjusted forecasting transformed back to nominal terms, and nominal forecasting with inflation as an exogenous feature. Use origin-available inflation data and rolling backtests to determine whether adjustment improves performance. Present final pricing recommendations in relevant nominal currency terms.

## 4. Primary business KPI and model metrics

### Primary KPI: gross margin and gross-margin erosion

For comparable revenue and cost scopes:

- `GM_quote = (Revenue_quote - COGS_quote) / Revenue_quote`
- `GM_realized = (Revenue_realized - COGS_realized) / Revenue_realized`
- `GM_erosion = GM_quote - GM_realized`

With unchanged fixed selling price, isolated per-unit material cost erosion is `material_cost_realized - material_cost_quote`; its percentage-point margin impact is `100 * material_cost_erosion / Revenue_quote_per_unit`, assuming comparable cost scope and no offsetting changes.

**Limitation.** Realized margin erosion and actual margin improvement cannot currently be measured: the reviewed inputs lack quotation, sales, realized-cost, order, and approved margin-definition data. Do not claim realized or avoided margin impact without them.

### Operational forecast metrics

- Material and partial BOM forecast error (MAE; MASE and RMSE as supporting metrics) versus LOCF, by target and horizon.
- Cost-underestimation frequency and magnitude when P50 or a selected scenario is below subsequent observed cost.
- P10/P50/P90 quantile loss and empirical P10–P90 coverage by target and horizon; verify `p10 <= p50 <= p90`.

These are operational proxies, not substitutes for realized margin or adoption outcomes. Business aggregation and weights require validation.

## 5. Data and BOM cost methodology

**Confirmed locally, at a high level:** monthly raw-material and energy series, BOM mappings, auxiliary metal/epoxy/steel inputs, a synthetic RMU benchmark, a provisional RM6 competitor/teardown benchmark, and supplier financial files. Core material data include direct prices, FX, derived EUR prices, PPI proxies, and intentionally empty licensed series. Energy data may be explanatory inputs, not BOM prices. Detailed schemas and series coverage belong in Issue #2.

`Data/` is excluded from Git. Local presence does not authorize publication or redistribution. Use benchmark inputs only after verifying comparability and permissions.

**BOM limitations / pending clarification.** `share_pct` represents mass share, not product-cost share. Material coverage is incomplete, and local BOM representations conflict on labels and quantities. Schneider Electric has acknowledged inconsistencies and expects to clarify them. Do not resolve conflicts or infer missing materials before that clarification; report omitted coverage.

**Preferred proposed method — partial monetary BOM index.** For materials with validated quantities and compatible, approved unit prices:

`partial_BOM_cost_t = sum_i(quantity_i * unit_price_i,t)`

Convert units and currencies consistently, then normalize to an agreed base:

`partial_BOM_index_t = 100 * partial_BOM_cost_t / partial_BOM_cost_base`

Disclose included materials, omissions, configuration, units, currency, and base month. This is partial, not complete product cost, until relevant quantities and baseline prices are confirmed.

**Fallback proposed method — mass-weighted proxy.** If monetary costing is infeasible, normalize eligible price relatives to 100 and combine using validated masses or mass shares. Label it a proxy; never represent mass shares as cost weights.

Direct prices and PPI proxies require different treatment:

- **Direct prices:** convert compatible prices to common currency and physical units, then multiply by validated quantities. A market benchmark is not necessarily a Schneider purchase price.
- **PPI proxies:** forecast relative index movement; do not add index points to currency prices. Apply relative movement to validated baseline spend for monetary estimates; without spend data, use only in a labelled proxy.

## 6. Frequency, coverage, and forecast origin

**Confirmed:** core RM and energy calendars are monthly, November 2016 through October 2026. Individual series have gaps and different end dates; most core RM price series end in September 2026. Auxiliary files have differing coverage and may include provisional, licensed, or unverified data.

**Required challenge origin: September 2026.**

| Horizon | Target month |
|---|---|
| 3 months | December 2026 |
| 6 months | March 2027 |
| 9 months | June 2027 |

Exclude October and November 2026 observations from all inputs and features, including energy, FX, auxiliary, and revised data. A September value in a later download is eligible only if it was published by the September decision cutoff. Account for publication lags and revisions; do not backfill with later releases. Validate the precise cutoff and apply the same availability rule in backtests.

## 7. Uncertainty and commercial interpretation

**Proposed:** provide P10, P50, and P90 for every target and horizon. For costs, P10 is a lower-cost outcome, P50 the median, and P90 a higher-cost outcome. If calibrated, about 10%, 50%, and 90% of outcomes respectively fall below those quantiles; they are scenarios, not guarantees.

`P90 - P50` may be considered as a potential adverse cost-risk buffer relative to the median. Its confidence, use, and relationship to quotation terms require business validation; it is not an automatic surcharge. Evaluate quantile loss and empirical coverage by target and horizon. Express final pricing recommendations in relevant nominal currency terms.

## 8. Deliverables and success criteria

### Technical deliverables

- Validated input inventory covering availability, units, licensing, and limitations.
- September 2026 forecasts for validated targets at 3, 6, and 9 months, with candidate-model comparison against LOCF and leakage-safe rolling-origin backtests.
- P10/P50/P90 forecasts with point-accuracy and uncertainty evaluation.
- Partial monetary BOM index where feasible; otherwise a labelled mass-weighted proxy.
- Challenge CSV and explainable, reproducible workflow using authorized data.

### Business deliverables

- Scenario-based ETO cost exposure and pricing decision support that balances margin and competitiveness without claiming validated margin optimization.
- SF6-free strategy for priority markets, distinguishing verified evidence from adoption hypotheses.
- Business presentation conveying recommendations, uncertainty, and validation needs; a decision tool if scope and approved inputs permit.

Success means models are rigorously compared with LOCF using origin-available data; point accuracy and calibration are evaluated by horizon; BOM indices disclose method and coverage; the workflow is explainable and reproducible; and strategy recommendations are evidence-based or explicitly hypotheses. The challenge weights forecasting and strategy quality equally. Numerical thresholds require stakeholder agreement.

## Open questions for Schneider Electric

1. Which BOM representation and product/configuration mapping is authoritative, and can unresolved materials and conflicting quantities be reconciled?
2. What September 2026 decision cutoff, publication vintages, and inflation series are approved?
3. Which materials, regions, currencies, base month, and baseline prices or spend weights define the BOM index?
4. How, if at all, do RM6/RM AirSeT cost references map to SM6/SM AirSeT products in the adoption strategy?
5. Can protected quotation, cost, sales, order, and margin data be provided, with an approved margin-erosion definition and current-business comparator?
6. Which priority-market regulatory/customer evidence, competitor inputs, TCO assumptions, licences, and acceptance thresholds are approved?

## Governance and validation

Confirm rights before using or redistributing external, licensed, internal BOM, benchmark, or supplier information. Keep confidential inputs out of public repositories, reports, and generated artifacts unless explicitly approved. Document authorized sources, versions, transformations, and AI-tool use for reproducibility. Schneider Electric must validate BOM authority, product mapping, forecast cutoff, commercial definitions, benchmark comparability, and market claims before recommendations become business policy.
