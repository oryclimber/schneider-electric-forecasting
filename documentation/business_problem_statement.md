# Business Problem Statement

## Document status and evidence convention

This statement frames Issue #1 for the Schneider Electric forecasting project.
It is based on the X-HEC Data Challenge presentation, *From raw material prices
to winning the data center era* (November 2026, slides 12–21), and inspection
of the repository.

- **Confirmed** means explicitly stated in the challenge presentation or
  directly observed in this repository. A presentation describing an input
  does not confirm that its data file has been delivered.
- **Proposed** means a project approach to evaluate; it is not a confirmed
  Schneider Electric policy or current process.
- **Validation required** means the challenge team or client must confirm the
  information before it is treated as fact or used for a recommendation.

## 1. Business context and problem description

**Confirmed — challenge context.** The challenge focuses on custom-engineered
medium-voltage switchgear for data-center and other customers, including RM6
and RM AirSeT. Engineer-to-order (ETO) quotations may fix prices for 3, 6, or
9 months, while raw-material costs can move during the quotation or contract
period. The presentation identifies copper, carbon steel, insulating
materials, electronics, PPI, and a broader raw-material basket as relevant
cost drivers. It describes RM AirSeT as SF6-free and identifies its higher
cost structure and thinner margin as a commercial challenge.

Two linked problems motivate the project:

1. **ETO pricing and margin protection:** uncertainty in material costs can
   expose a fixed-price offer to cost increases after quotation. A quotation
   that includes too much risk premium can also be less competitive.
2. **SF6-free adoption:** in markets where regulation does not require the
   switch, customers need a persuasive economic and commercial reason to
   adopt SF6-free switchgear.

**Problem statement.** How can historical raw-material information be used
to estimate material and BOM cost exposure 3, 6, and 9 months ahead, support
competitive and margin-aware ETO quotation terms, and inform an evidence-based
commercial strategy for profitable SF6-free switchgear adoption?

This is a decision-support project. Forecasts are not automatic prices, a
guarantee of future supplier costs, or proof of customer savings.

## 2. Business decisions supported

### ETO offer decisions

Subject to validated cost and product inputs, the analysis is intended to
inform:

- Whether the material-cost outlook supports a price adjustment or risk
  buffer for each quotation horizon.
- Whether quotation validity should differ by horizon or risk scenario.
- Whether an indexation trigger or other contractual protection merits
  consideration.
- How raw-material scenarios affect RM6 and RM AirSeT BOM cost estimates.
- How to balance margin exposure against offer competitiveness.

The project will not determine approved prices, target margins, or contract
terms without client-provided commercial rules and approval.

### SF6-free market and adoption decisions

For markets verified to have no applicable SF6 mandate, the analysis is
intended to inform:

- Which markets, customer segments, or use cases warrant prioritization.
- How the SM6-to-SM AirSeT cost gap and total cost of ownership (TCO) affect
  the customer proposition.
- How verified customer decarbonization priorities and SF6 lifecycle
  considerations may support adoption.
- Which pricing, value-proposition, and go-to-market levers merit testing.
- How raw-material cost convergence or divergence could affect relative
  product economics.

**Validation required.** Market eligibility, regulation, customer evidence,
product performance, lifecycle claims, competitor position, and approved
product economics must be verified before recommendations are made.

## 3. Current business baseline and proposed naive forecasting baseline

### Current business baseline

**Confirmed.** The presentation describes the risk of fixed-term ETO offers
when material prices change and asks for a model to support offer decisions.
It does not provide a measured existing forecasting process, current quoting
rule, historical forecast error, realized margin erosion, current risk buffer,
or SF6-free adoption baseline.

**Validation required.** Establish the actual current-state comparator with
the business: current price-setting and indexation practices; quote and
contract validity; historical quotation-time cost assumptions; realized
material/BOM costs; offer, win/loss, and margin outcomes; and the current
product migration or adoption rate. Until those inputs are supplied, no
quantitative improvement over current practice can be claimed.

### Proposed naive forecasting baseline

**Proposed.** Use a last-observation-carried-forward (LOCF) forecast for each
validated material-price series: at each historical forecast origin, carry
the most recently observed price forward unchanged for 3, 6, and 9 months.
Apply the same fixed-price assumption to a BOM index, using validated,
unchanged product BOM weights and the latest available material costs. This
baseline is transparent and tests whether more complex forecasts add value
beyond assuming no change from the latest observed cost level.

The baseline must be recomputed within each time-ordered backtest fold using
only information available at that fold's forecast origin. Any missing
observations, non-monthly series, product-weight changes, or unavailable
targets require a documented rule before scoring. A probabilistic naive
baseline is not yet specified; if required, its interval method must be
defined using training-fold information only and assessed for calibration.

## 4. Primary business KPI and machine-learning metrics

### Primary business KPI

**Proposed primary KPI:** ETO gross-margin erosion attributable to material
cost movement during the quoted fixed-price period, reported per unit and, if
validated order quantities are available, per order. Report both currency
impact and percentage-point impact on gross margin against the quotation-time
cost basis.

This KPI connects cost forecasting to the margin-protection objective. Its
calculation requires an agreed definition of quotation-time cost, realized
or scenario BOM cost, selling price, product configuration, and applicable
margin treatment.

**Validation required.** The client must confirm that these fields are
available, the margin definition and aggregation level, how to treat price
adjustments/indexation, and which baseline offer policy is the comparator.
Without them, report forecasted BOM-cost movement as a proxy; do not describe
it as realized or avoided margin erosion.

### Corresponding machine-learning metrics

**Proposed primary model metric:** mean absolute error (MAE) of the forecast
RM6/RM AirSeT BOM cost per unit, evaluated separately at 3, 6, and 9 months
in the agreed currency or cost-index units. Compare the model with the naive
baseline using identical targets, forecast origins, and folds. Where a
validated business weighting scheme exists, also report a pre-agreed
weighted aggregate; do not invent weights.

**Proposed supporting metrics:**

- Material-level MAE and mean absolute scaled error (MASE), by target and
  horizon, to show errors across materials and relative to the naive
  baseline.
- RMSE as a supplementary measure that emphasizes larger forecast errors.
- For P10/P50/P90 forecasts, quantile loss and empirical interval coverage,
  by horizon and target. Verify that `p10 <= p50 <= p90`.
- BOM-cost error and forecast-versus-naive comparisons in units that can be
  interpreted alongside the business KPI.

Use time-ordered rolling backtests and exclude information that would not
have been available at each forecast origin. Metric selection and any
aggregation weights are **proposed** until confirmed by the project
stakeholders. No numerical pass threshold is specified in the presentation.

The SF6-free commercial objective is complementary, not reducible to a price
forecast metric. Potential business measures include qualified opportunity
conversion, adoption/migration rate, and incremental gross profit; each
requires an agreed definition, observation period, and suitable customer or
market data before it can be evaluated.

## 5. Available data

### Confirmed in the repository

The repository currently contains project scaffolding and documentation. No
challenge dataset, product BOM file, quotation history, realized costs,
margin outcomes, or market/customer research files were present during
inspection. No data dictionary or schema is available in the repository at
this stage.

### Described by the challenge presentation; file availability unconfirmed

The presentation describes a 10-year synthetic benchmark panel and mentions:

- Raw-material price proxies, including copper, carbon steel, insulating
  materials, and electronics.
- PPI and a full raw-material basket.
- Product configurations from 1F to 4F.
- Supplier balance-sheet information.
- Product benchmarks and Schneider Electric product raw-material
  percentages or weights.
- Competitor pricing landscape.

It also specifies forecast CSV fields including target type and identifier,
region, origin and target month, horizon, unit, P10/P50/P90, model name, and
AI-tool disclosure.

These are **confirmed descriptions in the presentation**, not confirmation
of delivered files, exact columns, measurement units, coverage by target,
license, or suitability for every decision. The data inventory and definitions
must be established before analysis.

### Data required or to validate

- Actual data files, schema, provenance, usage rights, and data owner.
- Series frequency, units, currency, region, aggregation, missing values, and
  any revisions or synthetic-data generation rules.
- BOM composition, material weights, product/configuration mapping, costing
  rules, and effective dates.
- For margin evaluation: quotation prices, quotation-time cost estimates,
  realized costs, order quantities, and the agreed margin definition.
- For commercial strategy: product cost gap, relevant customer/segment
  evidence, competitor information, market-specific regulatory status, and
  verified TCO or lifecycle inputs.

## 6. Frequency, historical coverage, and forecasting horizons

- **Historical coverage — confirmed as described:** 2017–2026 (10 years) for
  the synthetic benchmark panel.
- **Data frequency — validation required:** the presentation expresses
  origins and targets as months and gives a monthly CSV example, but the
  actual cadence of every supplied series is not confirmed. Confirm whether
  data are monthly, consistently sampled, and aligned before modeling.
- **Forecast horizons — confirmed challenge requirement:** 3, 6, and 9
  months ahead.
- **Forecast origin and final observation — validation required:** the
  example filename uses `202609`, but this is an example, not confirmation
  that September 2026 is the final available observation or the required
  submission origin.

For monthly series, the intended target months are origin month plus 3, 6,
and 9 calendar months. Confirm the origin-month convention, target alignment,
and treatment of incomplete periods when the data are received.

## 7. Technical and business deliverables

### Technical deliverables

1. Data inventory and data dictionary documenting verified inputs, units,
   frequency, provenance, limitations, and transformations.
2. Reproducible, time-aware analysis and forecasting workflow.
3. Naive LOCF baseline and comparable model evaluations using rolling
   backtesting for all validated targets and horizons.
4. P10/P50/P90 forecasts, with uncertainty diagnostics and ordered quantiles.
5. Forecast-to-BOM cost calculation for RM6 and RM AirSeT where the supplied
   inputs support it; document assumptions and units.
6. Forecast submission CSV following the challenge schema and naming
   convention confirmed for the actual forecast origin.
7. Documented dependencies, reproducibility instructions, and automated
   checks as appropriate to the final project scope.
8. A Streamlit decision tool only if supported by validated inputs and
   implementation scope.

### Business deliverables

1. Horizon-specific ETO offer guidance covering cost exposure and, where
   evidence permits, price adjustment, quotation validity, indexation
   triggers, and risk buffers across scenarios.
2. SF6-free market strategy identifying validated target segments, customer
   value proposition, economic considerations, commercial levers, and
   conditions for success.
3. Business presentation communicating evidence, uncertainty, limitations,
   recommendations, and next steps.

Recommendations should distinguish observed data, model outputs, business
assumptions, and client decisions.

## 8. Success criteria

The project will be considered successful when:

1. **Coverage:** every validated priceable material target and required BOM
   index has forecasts at 3, 6, and 9 months, or an explicit documented
   reason for omission.
2. **Forecast evidence:** rolling backtests compare the proposed models with
   the naive baseline at each horizon using agreed metrics and leakage-safe
   forecast origins. Performance is reported honestly, including cases where
   models do not outperform the baseline.
3. **Uncertainty:** P10/P50/P90 outputs are ordered and their quantile loss
   and empirical coverage are reported for evaluated targets and horizons.
4. **Business linkage:** BOM cost changes are traceable to validated
   materials, weights, units, and product configurations; margin impacts are
   quantified only if the required commercial data and definitions are
   provided.
5. **ETO usability:** the analysis supports actionable, horizon-specific
   offer discussions without representing forecasts as automatic pricing
   decisions.
6. **SF6-free strategy:** recommendations specify evidence-backed segments,
   customer value, economic levers, and validation gaps for markets confirmed
   to lack an applicable mandate.
7. **Reproducibility:** a documented workflow reproduces the validated
   forecast outputs and CSV from the available inputs.

The presentation says 3/6/9-month forecast accuracy and the quality of the
non-regulated-market strategy matter equally. It does not define numerical
accuracy, margin, adoption, or pass/fail thresholds. Stakeholders must agree
those thresholds after the data, business baseline, and evaluation protocol
are confirmed.

## Client validation checklist

- Confirm dataset delivery, provenance, license, schema, frequency, units,
  and regional coverage.
- Confirm forecast origin, target-month alignment, and expected CSV naming.
- Confirm BOM weights, costing rules, product mappings, and current product
  cost gap.
- Confirm the current ETO quotation and pricing baseline and provide the
  fields needed to calculate margin impact.
- Agree primary and supporting metrics, aggregation weights, and
  acceptance thresholds.
- Confirm target markets, regulatory status, customer evidence, and
  substantiation for TCO and lifecycle claims.
- Confirm how recommendations will be reviewed and which decisions remain
  with authorized commercial stakeholders.
