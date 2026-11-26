# Submission method and validation

This note documents how the September 2026 challenge pack was produced and what
it does and does not prove. The generated files live under
`data/submission/`.

## Reproduce

```bash
uv sync --locked --dev
uv run --locked python scripts/build_challenge_pack.py
```

The script is standalone and does not modify the Phase 3-5 experiment outputs.

## Forecast method

- **Origin:** September 2026. October and November 2026 are excluded.
- **Targets:** December 2026 (3 months), March 2027 (6 months), June 2027
  (9 months).
- **Point forecast:** LOCF, the mandatory benchmark from the research pipeline.
- **Intervals:** for each horizon, historical cumulative log changes are
  centred on their median and resampled 5,000 times. The P10 and P90 are applied
  to the latest observed price. This keeps the median at LOCF and produces an
  asymmetric empirical interval.
- **Why not an advanced model:** the Phase 3 backtest found LOCF hard to beat;
  the advanced and probabilistic pipelines were not frozen in time for this
  pack. A flat median with empirical uncertainty is the transparent fallback
  agreed in the decision log.

Limitations:

- overlapping historical windows mean the effective sample is much smaller than
  the nominal count;
- the September cutoff is the challenge convention, not verified publication
  timing;
- intervals are not calibrated guarantees;
- no inflation adjustment, regime model or joint material/FX model is used for
  the material rows.

## BOM index method

`BOM_cost_t = sum(quantity_i x compatible EUR/t price_i,t)`

`BOM_index_t = 100 x BOM_cost_t / BOM_cost_2026-09`

Priced materials and sources:

| BOM material | Price series |
|---|---|
| Copper | D01 (LME copper in EUR/t) |
| Aluminium | D02 (LME aluminium in EUR/t) |
| Carbon steel | German CR coil, MEPS |
| Stainless steel | 304 stainless, Germany, CRU base + alloy |
| Galvanised steel | German HDG coil, MEPS |
| Epoxy | European epoxy resin, EUR/t |

AirSeT uses the German steel series as an EU proxy because no Hungary-specific
steel series is available. September stainless steel is carried forward from the
last August observation.

Excluded: plastics, brass, "Other", unidentified remainder. They are reported in
`bom_coverage.csv`, not silently weighted.

Coverage: RM6 92.7% of mass; RM AirSeT 85.9% of mass. Coverage by cost is not
known because the BOM has no approved cost weights.

The BOM P10/P90 use a joint historical simulation: for each horizon, complete
rows of simultaneous cumulative log changes for the six price series are
resampled. This preserves cross-material dependence. It does not reconcile the
BOM index to a full product cost or to the material forecasts.

## Offer rule

The rule is implemented in `scripts/build_challenge_pack.py` and extended by `scripts/offer_matrix.py`. It uses the BOM P50/P90,
a material-share assumption, a risk-buffer fraction and a target margin. The
working assumptions are an 80% material share for RM6 (from the should-cost
teardown) and AirSeT (assumed similar), and 50% of the P90-P50 gap.

## Offer matrix, supplier risk and competitor view

Run after the challenge pack, in this order (the last two read CSVs written by
the earlier scripts in `data/submission/`):

```bash
uv run --locked python scripts/offer_matrix.py            # quote adjustment, validity, indexation
uv run --locked python scripts/supplier_risk.py           # supplier scorecard
uv run --locked python scripts/supplier_material_map.py   # supplier-to-material map (BOM shares from the BOM)
uv run --locked python scripts/supplier_product_risk.py   # supply-base risk by material and product
uv run --locked python scripts/supplier_offer_terms.py    # supplier buffer, validity cap, stress cases
uv run --locked python scripts/competitor_pricing.py      # competitor and should-cost comparison
uv run --locked python scripts/competitive_risk_view.py   # our quote vs competitors, with sensitivity
```

Findings are summarised in `documentation/supplier_competitor_report.md`. Validity
is the longest window whose P90 price impact stays within a tolerance (10% of the
target margin by default); when no window qualifies the result is the 1-month floor.
Competitor prices are synthetic and held flat; supplier links, ratings, stress sizes
and the tolerance are working assumptions.

## Validation checklist

- 42 submission rows: 36 material and 6 BOM index.
- `p10 <= p50 <= p90` on every row.
- P50 equals the latest observed price for material rows and 100 for BOM rows.
- Coverage disclosed for both products.
- No October or November 2026 values used.
- Reproducible with one command.
- All assumptions repeated in the Excel `Assumptions` sheet.

## What the pack does not claim

- a complete product cost;
- a validated SM6/SM AirSeT mapping;
- realised margin improvement;
- calibrated P10/P90 coverage;
- actual competitor prices (the RMU benchmark is synthetic);
- permission to redistribute licensed steel or supplier data;
- validated supplier-to-material links or supplier volume splits;
- a confirmed quote validity (the default tolerance returns the 1-month floor).
