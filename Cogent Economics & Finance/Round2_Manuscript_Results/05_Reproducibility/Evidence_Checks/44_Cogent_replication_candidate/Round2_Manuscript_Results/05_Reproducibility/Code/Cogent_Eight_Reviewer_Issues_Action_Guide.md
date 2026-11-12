# Cogent reviewer issues: evidence-gathering and correction plan

**Purpose:** Improve the eight remaining reviewer concerns without rewriting original model results or claiming that unobserved facts have been verified. The scripts below read original data and predictions but write only new files in `Round2_Manuscript_Results`. They are not a substitute for an independent replication.

## Starting point

Project root: `E:\Governance-Policy\Cogent Economics & Finance`

Already verified locally: 13 Round 2 Python scripts, 40 CSV, 14 TXT, 1 Markdown; original `revision_outputs` includes six paired AUC comparisons, 30 NumPy arrays, a 153-variable locked feature specification (150 used), recalibration predictions and threshold tables. The Mendeley source is version 3; the 2019 selected test data represent 41,145 of 401,169 originations (all selected via recorded terminal status).

## New scripts and tasks

| Script | Question | Output / interpretation |
|---|---|---|
| 44 | Is the existing replication material present, and do six endpoint AUC tests survive Holm correction? | One candidate local ZIP with SHA-256 manifest, missing-file report, Holm/BH CSV. This **does not** verify that original notebooks execute in a clean environment. |
| 45 | Does a **post-hoc** independent chronological split of validation observations alter selection, calibration and threshold results? | Selection on Jul–Sep 2018, calibration Oct–Dec 2018, threshold fitting Jan–Mar 2019, descriptive evaluation on the unchanged selected 2019 test cohort. Saved model predictions only; **no model retraining**. Cannot undo the original use of all validation data. |
| 46 | What review cost and intervention effectiveness would be needed for a hypothetical review-action model to beat simple policies? | Grid and break-even conditions using actual saved confusion counts, but assumed effectiveness and costs. **Not** measured lender economics. |

### PowerShell commands

After extracting this ZIP into the existing `Round2_Manuscript_Results\05_Reproducibility\Code` directory, use these commands from the project root:

```powershell
$code = ".\Round2_Manuscript_Results\05_Reproducibility\Code"
python "$code\44_build_cogent_evidence_package_and_holm.py"
python "$code\45_independent_validation_windows_sensitivity.py"
python "$code\46_review_cost_effectiveness_scenario_bounds.py"
```

Use Step 44 first. Step 45 reads approximately 2.7 million Stata rows to align validation months and may require more time/memory than the other scripts. All stages stop if critical outcome counts or metadata alignment fail. Step 46 is lightweight and uses saved `.npy` arrays.

## Current status of the eight concerns

1. **Outcome-selection bias:** Existing mature test samples are outcome-dependent. A 2019 all-origination fixed-horizon classification evaluation cannot be identified from final status alone, even with competing-risk curves. A defensible resolution needs independently verified common-horizon event/status histories or another outcome-independent validation cohort with appropriately trained models. A 2015 fixed-vintage 36-month analysis in a different NeurIPS study is not an exact reproduction of the Cogent 2019 classifier.
2. **Validation reuse:** Step 45 adds a stronger alternative chronological, disjoint-stage sensitivity without reusing a single validation observation between selection/calibration/threshold fitting, but original reported results remain unchanged. Post-hoc design and selected-cohort bias persist.
3. **Predictor timing/leakage:** The source Stata construction suggests date-derived and macroeconomic fields can be post-origination. The original 150 predictors and reduced 77-predictor sensitivity must be included in the archive. To demonstrate a point-in-time evaluation, acquire original field timestamps and release-vintage provenance, rebuild predictors from the point-in-time data and retrain. Neither the name audit nor exclusion sensitivity certifies leakage freedom.
4. **Reproducibility:** Step 44 generates a candidate ZIP and detailed manifest. Verify the exact notebook version and run a clean-environment replay with the original public data. Review logs and code for personally identifying details before any public release. Do not equate a ZIP download with independently verified reproduction.
5. **Cost ratios, review actions and capacity:** Step 46 shows cost/effectiveness assumptions transparently and quantifies break-even conditions. Real lender loss data, approval/denial and intervention effects, workforce capacity and review costs are necessary to establish institutional economic value. Flagged loans are not necessarily human reviews.
6. **Multiple comparisons, clusters and training uncertainty:** Step 44 applies Holm and BH to exactly six original paired test AUC comparisons; these p-values are bootstrap-resolution-limited at 0.001. Prior 1,500 state and month bootstraps are **separate one-way** exercises. Neither adjustment provides end-to-end retraining uncertainty, two-way clustering, adjustment for all other exploratory tests or independent post-selection inference.
7. **Sparse subgroup/confounding:** Only four jurisdictions supply all 641 high-unemployment test loans and 74 adverse outcomes, with effective geographic sample near 3.3. No within-jurisdiction unemployment comparison at the fixed threshold is identifiable. Repeated resampling cannot remove that confounding; additional time/states or an appropriately designed matched external validation set is required. No protected-group fairness claim is supported.
8. **Governance validation:** The existing retrospective alert matrix is a demonstrator of logical rules. Efficacy needs independent pre-specified evaluation with observed escalation, reviewer actions, decisions, resources and subsequent outcomes, ideally a prospective controlled or stepped implementation. No statistical procedure on existing static probabilities can manufacture these missing interventions.

## Manuscript reporting guidance

Do not replace existing seed-42 tables unless a new analysis is explicitly identified as an alternative sensitivity. Report separate validation-window metrics as post-hoc only. Report multiplicity correction only for its defined six AUC hypothesis tests. Label Step 46 numbers as hypothetical economics. Preserve the distinction between 'reviewed/model-score flag' and an observed human-review action. Deposit original data source instructions but not the 2.8GB raw copy without checking license/data conditions.

## Minimum completion criteria

A credible submission can have **all reviewer requests responsibly addressed**, even if some causal/empirical questions cannot be resolved with the available data. A claim of **all underlying limitations fully resolved** would require: a new unbiased outcome cohort, verified point-in-time features, a clean code replay, actual lender costs/actions/capacity, a more representative subgroup validation sample, and evidence on implemented governance outcomes.
