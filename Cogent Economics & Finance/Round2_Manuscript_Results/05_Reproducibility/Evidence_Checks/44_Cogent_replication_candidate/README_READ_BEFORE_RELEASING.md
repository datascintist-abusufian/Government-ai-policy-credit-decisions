# Cogent Economics & Finance: candidate replication materials

IMPORTANT: This is a **local candidate archive**, not a verified public release.
It has not been replayed end-to-end in a clean environment. Before publishing,
inspect notebooks and logs for personal information, absolute local paths, and licences.

## Distinguish the experiments
This is **not** the NeurIPS 2026 2015-vintage / 38-feature study.
The Cogent classification experiment uses an outcome-selected 41,145-loan test
cohort and the original 150-feature specification.

## Data
Mendeley Data v3, DOI 10.17632/wb3ndt69gf.3. The raw `P2P_Macro_Data.dta`
is NOT included in this candidate archive.

## Files
- `original_notebooks/`: notebook candidates needing manual provenance checking.
- `revision_outputs/`: original saved prediction arrays and tabular outputs.
- `Round2_Manuscript_Results/`: Round 2 scripts and outputs, including corrected
  `32_clustered_test_uncertainty_v2.py`.
- `SHA256_MANIFEST.csv`: hashes of included files.
- `44_holm_BH_adjusted_six_test_auc_comparisons.csv`: defined six-test AUC family.

## Important limitations
The saved test predictions are conditional on an outcome-selected cohort.
Historical decision-time availability of predictors and fixed-horizon outcomes
have not been independently verified. The Round 2 pip versions reflect the
current environment, not necessarily the original model-training environment.
Observed human review and governance interventions are unavailable.

## Publication prerequisite
Run an independent clean-environment replay, including source data parsing,
preprocessing, model fitting and recomputation of reported metrics. Verify that
all included materials can lawfully be published. Resolve any mismatches and
document them. Do not claim that this ZIP has already passed these tests.
