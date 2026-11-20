# Pre-commit and pre-push audit

Audit date: November 20, 2026 (Europe/Madrid). This is a non-confidential release
review. The original audit snapshots below precede staging. The user subsequently
approved staging these exact 41 paths and creating one local commit with the
message specified below. Publishing, merging, rewriting history, deleting files
and changing branches remain unapproved. See the refreshed preparation review.

## Git and remote snapshot

- Current branch: `feat/data-engineering` at `d6f7a88f7a5203dc4ee4a709a6db82c342ba9470`.
- Upstream: none configured. The current branch does not exist on origin and has
  no existing pull request.
- Origin: `https://github.com/oryclimber/schneider-electric-forecasting.git`.
- Live remote main and local origin/main match at
  `d028d319b9c97d37ba9c1c9ee246bca9e3f28349`.
- Current HEAD is two commits ahead of main and zero behind: the data-understanding
  commit followed by the ingestion/validation commit. Phases 2–5 extensions remain
  uncommitted in the working tree.
- Staged files: zero. The proposed change set below has ten modified tracked files
  and thirty-one new files. Five other existing source/configuration files remain
  tracked and unchanged; they do not need restaging.
- GitHub reports WRITE permission for the authenticated account. No push, including
  a dry-run push, was attempted, so branch protection/push acceptance is untested.
- Remote inspection used `git ls-remote` and GitHub read APIs; no fetch or local
  ref updates were needed. Internal local checkpoint refs are not branch ancestry
  and must not be published with all-ref or mirror commands.

## Existing PRs and integration risk

| PR | Current evidence | Relationship and recommendation |
|---|---|---|
| [#4 Feat/setup](https://github.com/oryclimber/schneider-electric-forecasting/pull/4) | Open; approved; latest CI successful; mergeable/clean against current main. GitHub compare: seven commits ahead, zero behind main. | Overlapping paths are README.md and .gitignore. Reconcile team onboarding text and retain all confidentiality rules when integration is approved. Its added glossary, business-case/data-dictionary placeholders and placeholder test do not overlap this proposal. |
| [#6 Data understanding](https://github.com/oryclimber/schneider-electric-forecasting/pull/6) | Open; review required; mergeable but blocked; latest CI failed because no tests were collected (pytest exit 5). | Its head commit `6f9b72d7e0f2f821753a3df91afb0a0a9198260b` is already the parent of this branch's ingestion commit. This proposal further updates the same documentation file. Coordinate ownership and merge sequence; preserving that commit's ancestry avoids duplicate review history. A squash/rewrite or independent document edits could require manual reconciliation. |
| [#5 Business problem statement](https://github.com/oryclimber/schneider-electric-forecasting/pull/5) | Already merged into main. | Included in this branch's base; no separate outstanding PR dependency. |

No merge/rebase/conflict-resolution operation was performed. These are verified
ancestry/path overlaps, not a claim that Git has simulated merging the uncommitted
working tree. If the team merges or updates either open PR, refresh the remote
snapshot and rerun checks after an approved reconciliation. Do not change the
branch or rewrite its history as part of staging this audit.

## Findings fixed during this audit

1. Removed literal client BOM quantities from the reusable EDA report builder.
   Sensitive values are now obtained from authorized inputs and rendered only into
   ignored local reports. Coverage/observation counts are calculated rather than
   copied as source-code data snapshots.
2. Made historical migration comparisons optional. Current EDA and its source
   notebook now complete with only the three authorized workbooks and explicitly
   report unavailable old/new comparisons when private archives are absent.
3. Corrected the notebook runner to write executed outputs under the configured
   SCHNEIDER_DATA_DIR, and made phase flags mutually exclusive.
4. Added a fail-closed Phase 5 prerequisite for the frozen Phase 4 experiment
   configuration, selection and final forecasts, rather than treating an empty
   directory as preserved evidence.
5. Excluded generated phase directories from raw-source inventory scans. The
   consolidation utility remains unexecuted because it can move/remove files.
6. Added ignore safeguards for common raw exports, executed notebooks, serialized
   models and local MLflow artifacts outside the default ignored data tree.
7. Aligned pre-commit Ruff with the exact locked 0.17.5 version and uv-lock with CI's
   uv 0.13.8. Both upstream tags were verified read-only. Updated stale availability
   docstrings, README instructions and the non-confidential HANDOFF.md.
8. Added regression tests for archive-free EDA, configured notebook-output roots
   and missing Phase 4 evidence. Forecast estimation/selection rules were not
   retuned on confirmation data during the audit.

## Validation evidence

- Complete workspace suite: **96 passed**, with one existing statsmodels API
  FutureWarning in a reference-comparison test. No failures or skipped private
  data tests.
- Isolated source-only copy: **96 passed**, installed with
  `uv sync --locked --dev` from the exact repository lock. No client files or
  ignored helper scripts were copied. Ruff 0.17.5 was used, matching the lock.
- Ruff check and formatting: passed on all 35 versionable Python/notebook files,
  including unchanged tracked source, using the exact locked Ruff version.
- `uv lock --check --offline`: passed, 150 locked packages.
- Git diff whitespace check: passed. Python AST, YAML and TOML parsing passed.
- Manual equivalents of newline/trailing-whitespace and added-file-size hooks
  passed. The existing modified uv.lock is above the default new-file size limit;
  it is intentionally retained, and the standard check-added-large-files hook
  checks newly added files unless --enforce-all is specified. All newly proposed
  files are below the hook's default threshold.
- All four source notebooks executed in order inside the isolated exact-locked
  environment using explicitly marked synthetic workbooks only: EDA 18 cells,
  Phase 3 12 cells, Phase 4 10 cells, Phase 5 10 cells, with no error outputs. The
  synthetic clean room had no private historical archive. These are execution
  smoke tests, not additional evidence of client forecast accuracy.
- The software needs the three approved private workbooks for real research runs.
  It does not need ignored Python scripts. Every reporting/orchestration helper
  imported by the source notebooks is included in the explicit staging proposal.
- Existing authorized local phase reports/executed notebooks remain private.
  The audit did not regenerate or overwrite client forecasts, alter frozen Phase
  4 results, or run the mutating consolidation utility.
- New remote CI is not available until a commit/PR is approved and published.
  The local exact-locked clean-room checks do not claim a new Linux CI run.

## Confidentiality and staging boundary

The current tracked tip, index and reachable HEAD history contain no raw data,
Excel files, forecast exports, executed notebook outputs, MLflow stores, trained
model files or simulation datasets. The index is empty. All four proposed source
notebooks have null execution counts and no outputs, attachments or widget state.
Reviewed reusable source has no hard-coded client BOM quantities or raw price
observations. Existing documentation describes schemas/context; numerical research
results remain in ignored local outputs.

Private artifact paths and generated extensions were checked against tracked,
staged and proposed paths. No such files are in the proposal. The scan excludes
private local host/checkpoint refs from the branch publication claim: only the
explicit current branch is proposed for a later push. Ignore rules are safeguards,
not permission to force-add a restricted file or serialize model JSON beside
source. Keep all future generated artifacts under the configured private root.

HANDOFF.md records output schemas and the Phase 6 prerequisites. The qualitative
handoff is suitable for repository review; client prices, quantities, performance
samples and simulation values are not embedded in it or this audit.

## Exact proposed staging list

Only these 41 paths are proposed. This list includes this audit itself. Existing
unchanged CI, Python-version, package initializer, business-problem statement and
validation-test files are already tracked in branch history.

```text
.gitignore
.pre-commit-config.yaml
AUDIT.md
HANDOFF.md
README.md
documentation/data_understanding.md
notebooks/advanced_forecasting_evaluation.ipynb
notebooks/exploratory_data_analysis.ipynb
notebooks/feature_engineering_baseline_forecasting.ipynb
notebooks/probabilistic_forecasting.ipynb
pyproject.toml
scripts/audit_data.py
scripts/build_eda_report.py
scripts/consolidate_data.py
scripts/execute_eda.py
scripts/phase4_reporting.py
scripts/phase5_reporting.py
scripts/run_phase3.py
scripts/run_phase4.py
scripts/run_phase5.py
src/schneider_electric_forecasting/advanced.py
src/schneider_electric_forecasting/availability.py
src/schneider_electric_forecasting/baselines.py
src/schneider_electric_forecasting/config.py
src/schneider_electric_forecasting/eda.py
src/schneider_electric_forecasting/features.py
src/schneider_electric_forecasting/ingestion.py
src/schneider_electric_forecasting/probabilistic.py
src/schneider_electric_forecasting/random_walk.py
src/schneider_electric_forecasting/transforms.py
src/schneider_electric_forecasting/validation.py
tests/test_advanced.py
tests/test_availability.py
tests/test_canonical_migration.py
tests/test_eda.py
tests/test_ingestion.py
tests/test_phase3.py
tests/test_probabilistic.py
tests/test_reproducibility.py
tests/test_transforms.py
uv.lock
```

## Refreshed local-commit preparation review

Read-only GitHub and remote-head queries reconfirmed the same main, PR #4 and
PR #6 commit IDs above. There is still no remote feat/data-engineering branch or
open PR for it. No fetch, branch change or remote write was performed.

PR #6 run [44596058927](https://github.com/oryclimber/schneider-electric-forecasting/actions/runs/44596058927)
has one failed step: `uv run pytest -q` printed `no tests ran in 0.00s` and exited
5. Installation, Ruff lint and formatting passed. Its commit tree contains no
tests. The checkout Node.js deprecation warning was not the cause of failure.
The current branch adds real synthetic tests and resolves the collection cause
locally; this does not change PR #6's status, because its head remains unchanged.
New Linux CI still requires publication and a PR.

PR #4's README adds the team identity, uv onboarding, pre-commit installation and
an unfinished checklist to the original one-line README. Our branch replaces that
one-line README with pipeline instructions, so these are overlapping edits that
need manual reconciliation. Retain the team's identity and hook onboarding along
with the locked-install and forecasting instructions. PR #4's .gitignore retains
data/ and Data/, changes their comment and adds *.csv, *.zip, repomix-output.md,
.DS_Store/ and _MACOSX/. Our branch already adds *.csv and broader private research
artifact rules, but not the other additions. Preserve the union during an approved
integration; do not replace either file wholesale. No integration was attempted.

Recommended publication strategy, requiring separate approval:

1. Keep this branch and all existing commit identities. Publish only this branch
   and open a draft pipeline PR against main, explicitly noting that it contains
   PR #6's original documentation commit. Agree with the PR #6 owner to supersede
   that PR rather than bypassing its failed check or adding a dummy test there.
2. Coordinate PR #4's merge first. After separate merge authorization, fetch main
   and merge it normally into this feature branch. Reconcile README and .gitignore
   as described above, keep this phase's private-artifact safeguards, rerun checks
   and publish the resulting additional merge commit without force.
3. Require successful new Linux CI and review on the pipeline PR before merging.
   Do not rebase, squash-rewrite this shared branch, force-push or publish internal
   checkpoint refs. If the team prefers retaining PR #6 independently, agree on
   its CI repair/merge order before retargeting any draft PR.

The exact path list was checked against all modified/untracked versionable files
and the preceding audit's SHA-256 manifest before preparation. Only this audit
document was refreshed to record the requested CI investigation, strategy and
commit message; the approved path list remains unchanged. No model or confidential
output was regenerated or retuned during local commit preparation.

## Approved local commands and publication commands pending approval

The user approved the enumerated staging operation and local commit. Review the
staged diff and confidentiality checks before committing. The push commands remain
recommendations only; do not run them without explicit publication approval.

```bash
git status --short --branch
git diff --check
git add -- \
  .gitignore \
  .pre-commit-config.yaml \
  AUDIT.md \
  HANDOFF.md \
  README.md \
  documentation/data_understanding.md \
  notebooks/advanced_forecasting_evaluation.ipynb \
  notebooks/exploratory_data_analysis.ipynb \
  notebooks/feature_engineering_baseline_forecasting.ipynb \
  notebooks/probabilistic_forecasting.ipynb \
  pyproject.toml \
  scripts/audit_data.py \
  scripts/build_eda_report.py \
  scripts/consolidate_data.py \
  scripts/execute_eda.py \
  scripts/phase4_reporting.py \
  scripts/phase5_reporting.py \
  scripts/run_phase3.py \
  scripts/run_phase4.py \
  scripts/run_phase5.py \
  src/schneider_electric_forecasting/advanced.py \
  src/schneider_electric_forecasting/availability.py \
  src/schneider_electric_forecasting/baselines.py \
  src/schneider_electric_forecasting/config.py \
  src/schneider_electric_forecasting/eda.py \
  src/schneider_electric_forecasting/features.py \
  src/schneider_electric_forecasting/ingestion.py \
  src/schneider_electric_forecasting/probabilistic.py \
  src/schneider_electric_forecasting/random_walk.py \
  src/schneider_electric_forecasting/transforms.py \
  src/schneider_electric_forecasting/validation.py \
  tests/test_advanced.py \
  tests/test_availability.py \
  tests/test_canonical_migration.py \
  tests/test_eda.py \
  tests/test_ingestion.py \
  tests/test_phase3.py \
  tests/test_probabilistic.py \
  tests/test_reproducibility.py \
  tests/test_transforms.py \
  uv.lock
git diff --cached --check
git diff --cached --name-only
git diff --cached --stat
git diff --cached
git commit -m "feat: complete data engineering and probabilistic forecasting through phase 5"
# Optional network preflight after publication approval:
git push --dry-run origin HEAD:refs/heads/feat/data-engineering
# Publish only this branch, without force or mirror flags:
git push --set-upstream origin feat/data-engineering
```

The original audit executed no Git writes. Subsequent staging and the single local
commit are authorized by the user's preparation request. Publication, merging,
deletion, history rewriting and branch changes remain outside that authorization.
