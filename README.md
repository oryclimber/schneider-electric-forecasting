# Schneider Electric forecasting

The canonical source for material definitions, quantities and mappings is
`data/RM_Prices_XHEC_Challenge_v1.xlsx`. Its only sheet is `BOM_Materials`; it
contains no price history. The loader separates material rows from summary metadata
and never modifies the workbook.

Retained price observations and source definitions come from
`data/data/RM_Prices_XHEC_Challenge.xlsx`; independent energy and regional FX data
come from `data/data/Energy_Prices_XHEC_Challenge.xlsx`. The historical BOM embedded
in the price workbook is superseded. CSV exports are unnecessary for execution.
EUR material prices are recomputed from USD prices and USD-per-EUR rates.
Supplementary steel, epoxy, LME, supplier and market benchmark files remain local
for reconciliation and context; they are not silently merged into the core panel.

```text
src/schneider_electric_forecasting/   ingestion, validation, availability, features, models, diagnostics
scripts/                            audit, consolidation, notebook execution, reporting
notebooks/                          four unexecuted, reproducible source notebooks
tests/                              synthetic regression tests; no client data required
data/                               confidential, ignored local sources and outputs
  audit/                            inventory, sheet inspection, consolidation decisions
  .archive/before_v1/                earlier BOMs, exports and prior EDA
  eda_presentation/                  current catalog, checks, tables, figures and report
  phase3/ phase4/ phase5/             generated private results, experiments and scenarios
```

Install and execute from a full repository checkout (Python 3.14+, uv 0.13.8 as in CI). The script entry points below require the repository scripts/ directory; the package console greeting is not a forecasting entry point. On macOS, native tree models require an OpenMP runtime (for example libomp installed through your package manager).

Obtain the three authorized input workbooks from the data owner. They are intentionally absent from Git. Preserve the paths above or set SCHNEIDER_DATA_DIR to an approved absolute directory outside the repository with the same internal layout. A fresh clone needs no private Python scripts or historical EDA archive.

Install and run:

```bash
uv sync --locked --dev
uv run --locked python scripts/audit_data.py
uv run --locked python scripts/execute_eda.py
uv run --locked pytest -q
uv run --locked ruff check .
uv run --locked ruff format --check .
```

The historical migration utility `scripts/consolidate_data.py` is optional and is NOT part of normal execution. It moves source exports and removes redundant generated copies, so run it only with separate authorization and backups. The final audit does not execute it.

Consolidation preserves the first inventory and prior EDA, compares readable CSV
exports against workbook sheets, and archives superseded/uncertain files locally.
Cloud placeholders are archived by rename without reading or deleting their bytes.
The workbook inspection and decision log explain retained or archived files.
No source workbook is rewritten. Set `SCHNEIDER_DATA_DIR` to use another local data
root with the same layout. Historical comparisons can use preserved prior EDA under `.archive/before_v1/`; when it is absent, current EDA still completes and explicitly reports that the old/new comparison is unavailable.

The notebook regenerates numerical results and nine figure pairs (300 DPI PNG and
SVG). Its executed copy, catalog, validation results and detailed migration report
are written under `data/eda_presentation/`. The tracked source notebook has no
outputs. Keep all derived client data in ignored `data/`; never commit it.

Full-sample descriptive analysis includes retrospective observations through the
source endpoint. September-origin inputs use the explicit September 30, 2026
decision date in `config.py`, with documented maximum lag assumptions. Estimated
availability is separate from verified availability: retrieval times are not
publication evidence. Unsupported providers remain outside the origin subset.
Phase 1–2 ingestion/EDA performs no interpolation, licensed-value fabrication or forecasting-model training. Phases 3–5 fit only their documented models.
Historical P10/P50/P90 changes are descriptive, not predictive intervals. BOM mass
shares are not cost shares, and PPI proxies are not material-grade prices.

## Phase 3: features and baseline forecasting

`features.py` builds each origin's snapshot before calculating price/return lags,
rolling statistics, momentum, seasonal calendar variables, energy/FX drivers and
economically defined cross-material return spreads. Lags are relative to the last
observable month; cutoff and age metadata expose publication-delay staleness.
`baselines.py` compares LOCF, seasonal naive, drift and historical mean return on
identical eligible rolling windows. Forecast distances include the gap from the
last observable month to the actual target, rather than assuming current prices
are already available.

```bash
uv run --locked python scripts/execute_eda.py --phase3
# Alternatively execute the pipeline without notebook display:
uv run --locked python scripts/run_phase3.py
```

The source notebook is `notebooks/feature_engineering_baseline_forecasting.ipynb`.
Generated features, label ledgers, performance tables, exclusions, final September
forecasts, material recommendations, 300 DPI PNG/SVG plots, experiment assumptions
and the executed notebook are under ignored `data/phase3/`. The final decision is
September 30, 2026; targets are December 2026, March 2027 and June 2027.

The experiment uses documented conservative IMF/PPI lags plus an explicitly
provisional FX timing scenario; EUR derived prices inherit the maximum dependency
delay. Missing licensed series and unsupported BLS release rules exclude those
targets. Historical revisions remain a limitation: this is an assumption-based
backtest of a revised snapshot, not verified historical-vintage performance.

`supervised_features_and_labels.csv` includes future labels for dataset assembly.
Never pass its label columns to an estimator. Apply `training_rows(dataset, as_of)`
at each fold's decision before fitting; it excludes labels whose estimated release
time is later than that cutoff. Fit any future scaling, imputation or feature
selection inside those folds. Phase 3 itself fits no preprocessing or advanced
models. MASE uses the training-only one-month naive MAE; bias is prediction minus
actual. Final unobserved target months have forecasts only, with no fabricated
actuals or uncertainty intervals.

## Phase 4: advanced forecasting and nested evaluation

`advanced.py` extends the existing feature, availability and baseline modules with
ARIMA, ETS, Ridge, Elastic Net, XGBoost and LightGBM, each evaluated as price levels
and cumulative log returns. A fixed 24-feature economic subset limits complexity.
Imputation and scaling fit only on training rows inside each inner/outer fold.
Each training fold excludes labels not yet published under the availability rules.

```bash
uv sync --locked --dev
uv run --locked python scripts/execute_eda.py --phase4
# Or run the experiment and reporting without notebook display:
uv run --locked python scripts/run_phase4.py
# Optional local experiment UI:
uv run --locked mlflow ui --backend-store-uri sqlite:///data/phase4/mlflow.db --host 127.0.0.1
```

The source notebook is `notebooks/advanced_forecasting_evaluation.ipynb`. All
predictions, scores, tuning and coverage ledgers, configurations, local MLflow
SQLite/artifacts, figures, final forecasts and the executed notebook stay under
ignored `data/phase4/`. MLflow does not use an external tracking service.

Quarterly 2021–2022 outer folds supply model-selection data. All their labels must
be available before the December 31, 2023 family-selection freeze. Separate
confirmation folds begin January 2024. The inner two-fold rolling search uses two
predeclared configurations per candidate. The selected family can displace LOCF
only with at least 5% selection MAE gain, 60% origin wins, six selection folds and
positive gains in both 2021 and 2022. Final families remain frozen even when later
confirmation results are unfavorable; retrospective holdout rankings cannot choose
the final model. Hyperparameters refit inside later training-only inner folds.

Candidates use identical successful outer windows for each material/horizon.
Numerical failures and reduced coverage are exported explicitly. Unit-dependent
MAE/RMSE/bias remain per series; aggregate summaries use macro MASE and relative
MAE improvements. The quarterly selection/confirmation design differs from Phase
3's monthly baseline evaluation and the report explains that coverage difference.

Package versions, source hashes, parameter grids, feature sets, seed, fold windows,
availability assumptions and scores are recorded in the experiment and MLflow.
Native tree packages require the platform's OpenMP runtime (available in the
current environment). Neither historical revision leakage nor provisional FX
availability is resolved by nested validation. No BOM costing, pricing optimization
or probabilistic forecasts are implemented in this phase.

## Phase 5: probabilistic forecasting and joint scenarios

```bash
uv sync --locked --dev
uv run --locked python scripts/execute_eda.py --phase5
# Alternatively run the same calculations without notebook display:
uv run --locked python scripts/run_phase5.py
```

`probabilistic.py` reuses the existing origin snapshots and source-specific lag
assumptions. It implements LOCF forecast-error quantiles, historical simulation,
Gaussian log random walks, centered empirical innovation bootstrap, three-month
moving-block bootstrap, statistically gated GARCH and signed-error conformal
intervals. `random_walk.py` extends the earlier ADF/KPSS and ACF/PACF diagnostics
with robust variance ratios, Ljung–Box, HAC drift, ARCH-LM and break sensitivity.
Global Holm and exploratory test-family BH corrections are exported with each
test's null; no non-rejection is labeled proof of a random walk.

The new probabilistic policy is distinct from the frozen Phase 4 point policy.
Phase 4 artifact hashes are verified unchanged. Core methods are selected on
2020–2021 origins and frozen December 31, 2022. Dedicated 2022–2023 LOCF errors
calibrate conformal endpoints. Confirmation begins January 2025; outcomes must be
available by September 30, 2026. Residual pools never include confirmation-origin
errors. Other innovation/volatility parameters update using only each origin's
available history. All five core methods share paired evaluation dates; GARCH and
conformal have separately matched comparison tables and explicit failures.

The source notebook is `notebooks/probabilistic_forecasting.ipynb`. Reports,
forecasts, calibration/coverage ledgers, diagnostics, joint scenarios, figures and
the executed notebook are under ignored `data/phase5/`. P10/P50/P90 use the full
6/9/12-month observation-to-target gaps. LOCF is a point benchmark, not necessarily
the predictive median. Conformal expected prices are left missing because its
intervals do not define a full distribution. Dependence and shifts prevent claims
of exact conformal coverage. Monte Carlo draw counts do not increase the number
of economic observations.

Joint scenarios resample aligned innovation vectors, preserving shared material
and FX moves. They deliberately use a common June anchor, omitting extra July FX
information. EUR prices are derived pathwise from USD and FX. These scenarios are
not reconciled to every selected marginal distribution, and no BOM costs or
monetary weights are generated. Summing marginal P90 forecasts does not produce
the P90 of total exposure. Coverage, tail calibration and unresolved BOM inputs
must be reviewed before considering fixed-price-offer or cost decisions.

## Reproducibility, findings and Phase 6 handoff

Run the four notebooks in order: EDA, --phase3, --phase4, --phase5. Phase 5 now fails
closed if the Phase 4 experiment configuration, frozen selection and final point
forecasts are missing; restoring approved local artifacts is an alternative to
rerunning Phase 4. Executed notebooks are written to SCHNEIDER_DATA_DIR, while
source notebooks remain free of outputs. Do not run the migration/consolidation
utility merely to reproduce forecasts.

The test suite uses synthetic fixtures and needs no confidential workbook. CI runs
tests, lint and formatting; it does not possess the private inputs or run the full
client experiments. The final audit also installs the exact lock into an isolated
source-only copy and executes all four notebooks using clearly marked synthetic
workbooks. Synthetic smoke-test results validate execution, not forecast accuracy.

Use these checks before proposing a commit:

```bash
uv sync --locked --dev
uv run --locked pytest -q
uv run --locked ruff check .
uv run --locked ruff format --check .
uv lock --check
git diff --check
git diff --cached --name-only
```

Reusable code, source notebooks, tests, dependency files and documentation belong
in Git. Private inputs, raw exports, fitted-model derivatives, MLflow databases and
artifacts, forecasts, simulation paths, generated figures/reports and executed
notebooks do not. The ignore rules cover the default data tree and common
research-artifact extensions; inspect every proposed path rather than using
`git add .`, `git add -A`, forced adds or mirror pushes.

The non-confidential scientific conclusions are:

- LOCF remains the mandatory point benchmark. More complex tested models do not
  consistently justify their complexity on later evaluation windows.
- The probabilistic policy is separate from the frozen Phase 4 point policy.
  Near-nominal aggregate interval coverage hides weak tail/median calibration and
  material-specific undercoverage. It is not a fixed-price-offer guarantee.
- Diagnostics reject some specific random-walk-related nulls, especially for
  producer-price proxies; non-rejection elsewhere does not prove a random walk.
- Historical revisions, provisional FX availability, overlapping forecast errors,
  small effective calibration samples and regime shifts remain material limits.
- Joint scenarios preserve sampled material/FX dependence and exact currency
  identities, but deliberately omit extra asynchronous FX information and are not
  reconciled to every selected marginal forecast distribution.

Detailed numerical findings remain only in the ignored generated reports. See
[HANDOFF.md](HANDOFF.md) for entry points, output schemas and Phase 6 prerequisites,
and [AUDIT.md](AUDIT.md) for the exact staging proposal and branch/PR review.
Phase 6 should first resolve BOM mappings, quantities, units, approved monetary
weights, procurement terms and calibration requirements. Complete BOM costs and
pricing optimization remain outside the implemented scope.
