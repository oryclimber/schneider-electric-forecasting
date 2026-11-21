# Forecasting pipeline handoff: Phases 1–5

This document contains architecture, schema and qualitative findings only. Client
quantities, prices, forecasts, simulation values and detailed performance results
remain in authorized local outputs. No BOM costs or pricing optimizer have been
implemented. See README.md for installation and AUDIT.md for release checks.

## What is complete

| Phase | Implementation | Interpretation |
|---|---|---|
| 1 | Canonical BOM precedence, direct workbook ingestion, metadata and validation | The canonical v1 workbook supplies BOM content; retained RM and energy workbooks supply non-overlapping histories. |
| 2 | Coverage, missingness, trends, returns, volatility, diagnostics and comparisons | Full-sample descriptions are separate from origin-available inputs. Old/new comparisons are optional when no private archive is supplied. |
| 3 | Origin-specific features and paired rolling baseline evaluation | LOCF is mandatory; alternative baselines do not consistently justify replacing it. |
| 4 | Nested rolling ARIMA, ETS, Ridge, Elastic Net, XGBoost and LightGBM experiments; local MLflow | Price-level and log-return formulations are evaluated in original units; final families are frozen before confirmation. |
| 5 | LOCF-based probabilistic methods, expanded random-walk diagnostics and joint scenarios | This is a separate probabilistic policy, with incomplete tail calibration and no operational risk guarantee. |

## Inputs and access

Obtain authorized workbooks from the data owner; none should be placed in Git.
Default paths relative to the repository are:

- `data/RM_Prices_XHEC_Challenge_v1.xlsx`: `BOM_Materials` only; summary/footer rows
  are exposed separately by the loader.
- `data/data/RM_Prices_XHEC_Challenge.xlsx`: `Monthly_Wide`, `Monthly_Long`,
  `Source_Catalog` and source documentation. Its embedded BOM is superseded.
- `data/data/Energy_Prices_XHEC_Challenge.xlsx`: independent energy/FX observations
  and `Source_Catalog`.

`config.py` lists the required columns. Keep series IDs, provider definitions,
units, currencies and mappings intact. `SCHNEIDER_DATA_DIR` can point to an approved
absolute data directory outside the repository with the same internal layout.
The loader never updates source workbooks. Private CSV exports and historical
scripts are not required to execute the current pipeline.

Historical migration comparisons can additionally use `.archive/before_v1/` under
the data root. Missing archives are explicitly reported; current EDA still runs.
`scripts/consolidate_data.py` is a historical, mutating utility, not an execution
prerequisite. Do not run it as part of a release audit or without authorization.

## Entry points

Run from a full repository checkout after `uv sync --locked --dev`:

| Purpose | Command | Generated output directory under the data root |
|---|---|---|
| Read-only source inventory | `uv run --locked python scripts/audit_data.py` | `audit/` |
| Current EDA | `uv run --locked python scripts/execute_eda.py` | `eda_presentation/` |
| Features and baselines | `uv run --locked python scripts/execute_eda.py --phase3` | `phase3/` |
| Advanced model evaluation | `uv run --locked python scripts/execute_eda.py --phase4` | `phase4/` |
| Probabilistic forecasting | `uv run --locked python scripts/execute_eda.py --phase5` | `phase5/` |

Equivalent non-notebook experiment entry points are `scripts/run_phase3.py`,
`scripts/run_phase4.py` and `scripts/run_phase5.py`. Phase 5 requires the Phase 4
configuration, frozen selection and final forecast artifacts, and verifies Phase
4 file hashes without modifying them. All helper/reporting scripts are versioned
source candidates in `scripts/`; none is imported from an ignored data directory.
The package's console greeting is not the pipeline entry point.

Source notebooks in `notebooks/` contain no cached outputs. Their executed copies
are written under the configured private output root. Running on a different
machine requires access to the authorized inputs, not another user's home paths
or virtual environment. Native tree packages need a compatible OpenMP runtime.

## Reusable modules

| Module | Responsibilities |
|---|---|
| `config.py`, `ingestion.py` | Paths, schemas, canonical BOM separation and reproducible currency/unit-derived series |
| `validation.py`, `availability.py` | Structural checks, strict publication evidence and explicitly estimated availability |
| `transforms.py`, `eda.py` | Calendar alignment, non-imputed returns, summaries, statistical diagnostics and figures |
| `features.py` | Publication-gated origin snapshots, lag/rolling/calendar/driver features and supervised-label gating |
| `baselines.py` | Transparent point baselines, actual observation-to-target distances and paired metrics |
| `advanced.py` | Training-only preprocessing, inner rolling tuning and frozen model selection |
| `probabilistic.py` | Known-error ledgers, simulated distributions, conformal ranks, conditional-volatility gates and joint paths |
| `random_walk.py` | Explicit test nulls, robust variance ratios, serial/ARCH/drift/break diagnostics and multiplicity control |

## Time and leakage contract

- The final decision is September 30, 2026; target months are December 2026,
  March 2027 and June 2027. Forecast steps include the publication-delay gap from
  each series' last eligible observation, not just the decision-to-target horizon.
- Retrieval time is not publication evidence. Strict mode requires verified
  timestamps and fails closed. The executed studies use documented lag assumptions
  plus a separately labeled provisional FX rule; derived EUR prices inherit the
  maximum dependency delay.
- Feature lags are relative to the last observable month. Rolling transformations
  use only the selected snapshot, preserve calendar gaps and expose observation
  age. Fitted preprocessing belongs inside training folds.
- A supervised table may contain later realized labels. Before fitting, apply
  `training_rows(dataset, as_of)` and exclude `target_value` and label-release
  metadata from estimator inputs. Calendar target dates are known features;
  realized targets are not.
- Phase 4 uses earlier outer folds for selection, an intervening publication gap,
  and later confirmation folds. Confirmation rankings are descriptive and must
  not override the frozen policy. Hyperparameters retune only on available labels.
- Phase 5 has its own earlier selection period, dedicated conformal calibration
  period and later confirmation period. Confirmation-origin errors never enter
  residual calibration pools. History-based innovation parameters may update using
  information available at each origin. These retrospective dates were examined in
  earlier research, so they are not a new blind external test set.

## Output schemas

All files below are private generated outputs, not staging candidates. The
relevant manifest and audit ledgers define detailed provenance and assumptions.

| Artifact | Keys | Important fields |
|---|---|---|
| EDA `data_catalog.csv` | `series_id` | Definition, provider/code, unit/currency, classification, coverage, lag evidence and limitations |
| Phase 3 `supervised_features_and_labels.csv` | `series_id, origin, horizon` | `target_month`, features, `last_observed_month`, observation ages, `target_value`, `target_available_at_estimated` |
| Phase 3 `baseline_performance.csv` | `series_id, horizon, model` | `MAE, RMSE, MASE, bias, valid_windows, unit` |
| Phase 4 `outer_predictions.csv` | `series_id, horizon, origin, model` | Prediction, actual, error, training scale, partition, target-label release and serialized settings |
| Phase 4 `model_performance.csv` | `partition, series_id, horizon, model` | Point metrics, relative MAE improvement and common-window counts |
| Phase 4 `frozen_selection.csv` | `series_id, horizon` | `selected_model`, selection evidence, freeze date and rule |
| Phase 4 `final_selected_forecasts.csv` | `series_id, horizon` | Target month, original unit, selected point prediction and availability label |
| Phase 4 `nested_tuning_audit.csv` | Material/horizon, outer/inner origin, candidate/option | Training windows, latest label release, validation release, losses, warnings and failure status |
| Phase 5 `locf_error_ledger.csv` | `series_id, origin, horizon` | Latest available price, actual, log error and estimated target release |
| Phase 5 `primary_performance.csv` | `partition, series_id, horizon, method` | Pinball losses, quantile calibration, central coverage, width, Winkler and scaled metrics with sample counts |
| Phase 5 `challenger_matched_performance.csv` | Comparison set plus primary keys | GARCH/conformal and core scores on the same eligible subset; do not rank unpaired coverage |
| Phase 5 `final_selected_probabilistic_forecasts.csv` | `series_id, horizon` | Target month, selected method, `P10, P50, P90, expected_price`, latest observation/date, effective horizon, unit and assumptions |
| Phase 5 `distribution_audit.csv` | `series_id, origin, horizon, method` | Error/calibration counts, latest releases, GARCH gate/fit status and numerical failures |
| Phase 5 `random_walk_tests.csv` | `series_id, test` | Test family/null, statistic, raw p-value, adjusted p-values, observations and warnings |
| Phase 5 `joint_material_fx_scenarios.csv` | `scenario, series_id, horizon` | Target month, simulated price, common anchor and unit; scenario IDs are shared across materials |

P10/P90 are nominal central-80% limits, not certified coverage guarantees. P50 may
differ from LOCF. Conformal expected price is intentionally unavailable because
interval endpoints do not specify a full distribution. Source units must remain
attached to values; index points cannot be treated as currency per tonne.

MASE uses each training origin's one-month naive MAE, not a same-horizon benchmark.
Bias is prediction minus actual. Pinball and Winkler losses remain in original
units unless explicitly scaled. Aggregate tables use unitless macro averages;
never average raw price-unit MAE across unrelated materials. Overlapping horizons
and shared source series reduce effective sample sizes.

## Phase 6 prerequisites and next steps

1. Resolve authoritative BOM row identifiers, missing mappings, unknown composition
   and mass inconsistencies with the data owner. Retain blanks until confirmed.
2. Establish procurement-grade/geographic matches, purchase units, FX conventions,
   regional premiums, licensed-data permissions and quote/contract terms.
3. Obtain approved spend anchors or monetary weights. Mass share is not cost share;
   proxy indices require approved monetary anchors before costing.
4. Improve upper-tail and median calibration on new evidence. Investigate regime
   changes, dependent calibration errors and mean-whitened ARCH diagnostics before
   relying on conformal or GARCH density assumptions. Retain LOCF as a comparator.
5. Reconcile selected marginal distributions with a coherent joint risk model.
   Current shared-block scenarios omit extra asynchronous FX information and are
   exploratory. Preserve pathwise FX identities and validated dependence.
6. Define cost-coverage disclosures, risk tolerances, stress scenarios and decision
   criteria before enabling a BOM cost or fixed-price-offer calculation. The P90
   of a weighted total is generally not the sum of weighted marginal P90s.

Current limitations prevent reliable fixed-price-offer guarantees: potentially
revised snapshots, provisional availability, small effective calibration samples,
poor material-specific tail calibration, incomplete BOM information and unapproved
cost weights. The implemented pipeline supports research and validation; Phase 6
costing/pricing work needs those inputs and a separately approved scope.

## Release discipline

Run the complete synthetic tests and exact locked lint/format/dependency checks.
Review the exact staging list in AUDIT.md. No private workbook, dataset, report,
executed notebook, fitted model, MLflow store or simulation file belongs in Git.
Do not include private local checkpoint refs or use forced/mirror publication.
Coordinate the open team PRs before approving staging or publishing this branch.
