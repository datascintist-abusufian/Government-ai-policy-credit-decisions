# AI Governance for Credit Decision Systems in P2P Lending

**Reproducibility and research-evidence guide for the manuscript submitted to *Cogent Economics & Finance*.**

> **Study identity.** This README documents the retrospective, outcome-maturity-selected LendingClub analysis in the *Cogent Economics & Finance* manuscript. It is **not** the README for the separate NeurIPS 2026 MusIML study in this repository. The two studies use different cohorts, targets, predictors, analyses, and reported performance figures. Do not interchange their notebooks, tables, or replication claims.
>
> **Release status (12 November 2026).** This document describes the manuscript and the locally reported Round 2 audit evidence. It does **not** certify that all referenced analysis files have been uploaded to GitHub or that the full original training pipeline has been independently reproduced in a clean environment. See [Reproducibility status](#reproducibility-status) and [Release checklist](#release-checklist).

## Authors and correspondence

Kumar Utkarsh Singh, Jayasree Varadarajan, **Md Abu Sufian**, Md Aminul Islam Rana, Nadeem Qazi, and Nabeela Berardinelli.

**Corresponding author:** Md Abu Sufian, Department of Computer Science, University of East London, London, United Kingdom. Email: [m.sufian@uel.ac.uk](mailto:m.sufian@uel.ac.uk). Website: [abusufian.dev](https://abusufian.dev).

## Overview

The manuscript examines how the governance of AI-assisted credit decisions depends on the **population whose outcomes can be evaluated**, the **time at which predictors become available**, model discrimination and calibration, decision-threshold costs, and support for subgroup comparisons. The empirical analysis uses **2,703,430 originated US peer-to-peer loans** dated August 2012 through December 2019 from version 3 of the public LendingClub/macro-economic data archive.

The study is an **observational, retrospective governance analysis**, not a trial of deployed credit decisions. The reported model-performance results describe loans satisfying a reconstructed, outcome-dependent maturity rule. They are not fixed-horizon default-performance estimates for all loan originations, and they are not estimates for rejected applicants.

The Round 2 analysis includes original and alternative predictor specifications, training-size and seed sensitivity, paired and clustered uncertainty, calibration diagnostics, trivial decision-policy benchmarks, outcome-selection audits, a distinct competing-risk analysis, and retrospective illustrative governance alerts.

## Data source and provenance

- **Dataset:** *Dataset from the US Peer-to-peer Lending Platform with Macroeconomic Variables*, Nigmonov, Mendeley Data.
- **Version used:** **3**, DOI: [10.17632/wb3ndt69gf.3](https://doi.org/10.17632/wb3ndt69gf.3); deposited 30 September 2021. The manuscript reports access on 5 June 2025.
- **Source size:** **2,703,430 records**.
- **Source file identified in the manuscript:** `P2P_Macro_Data.dta`.
- **Source construction material audited:** `P2P_Macro_Codes.do`.
- **Dataset licence:** CC BY 4.0, as reported by the source deposit.
- **Version 4 was not used.** Reproductions should not silently substitute a later version.

**Data access:** Obtain the original **version 3** data and corresponding source materials from the DOI above. The loan-level data are **not redistributed** in this README. The manuscript's evidence transfer archive likewise does not contain the complete original modelling dataset. Data availability at the time of a historical credit decision has not been verified for every retained predictor.

### Chronological cohorts and selected evaluation samples

| Partition | Origination dates | All originations | Selected by reconstructed maturity rule | Selected share |
|---|---|---:|---:|---:|
| Training | August 2012 to June 2018 | 1,931,267 | 1,304,058 | 67.52% |
| Validation | July 2018 to March 2019 | 370,994 | 27,374 | 7.38% |
| Test | April 2019 to December 2019 | 401,169 | 41,145 | 10.26% |
| **Total** | August 2012 to December 2019 | **2,703,430** | **1,372,577** | **50.77%** |

The reconstruction uses **1 April 2020 as an analytical follow-up reference date**, not as a verified historical snapshot date. An observation is classified as mature when its *recorded* status is terminal (`Fully Paid`, `Charged Off`, or `Default`) **or** the elapsed calendar time from origination to the reference date is at least its contractual 36- or 60-month term. This does not prove that every older loan has complete follow-up.

All selected validation and test loans meet the rule through **recorded terminal status**, not completion of a contractual 36- or 60-month term by April 2020. Accordingly, test discrimination, probability-quality scores, and threshold statistics are calculated **only for the selected 41,145 loans**.

### Outcome definitions: do not conflate endpoints

**Classification target:** the source `badloan` status indicator, defined as 1 for recorded categories:

- `Charged Off`
- `Default`
- `Does not meet the credit policy. Status:Charged Off`
- `Late (16-30 days)`
- `Late (31-120 days)`

All other recorded source statuses were coded as 0 by the source construction. The implementation also included a zero-fill rule for missing labels; audits found no missing `badloan` values in reconstructed mature training samples. A zero is **not** a verified no-default outcome through a common future horizon, and the classification target is broader than charge-off alone.

**Separate competing-risk endpoint:** observed charge-off, with fully paid loans as competing repayments and unresolved loans right-censored. Event times use available payment-date information as a proxy; exact independently verified charge-off timestamps were not available. **Do not compare or combine the classification event counts directly with competing-risk event counts.**

## Original predictive modelling design

### Predictor sets and timing audit

| Specification | Raw predictor count | Encoded model inputs | Interpretation |
|---|---:|---:|---|
| Original | 150 | 384 | Historical original experiment; not verified point-in-time safe |
| Payment-date quarantine | 145 | 379 | Five documented payment-date-dependent fields excluded; other timing questions remain |
| Conservative candidate | 77 | 159 | Excludes 73 uncertain variables; not independently certified as leakage-free |

The original 150 predictors were derived from the locally saved `cell26_locked_feature_specification.csv`: its 153 locked fields exclude `emp_title`, `desc`, and `title` before model fitting. The exact specification should be published alongside the code, rather than recreated from this description alone.

The source-code and full-data timing audit identifies five predictors with documented last-payment-date dependencies:

`year`, `month`, `quarter`, `loan_vol`, `median_dummy`.

The stored `year`, `month`, and `quarter` match components of the last-payment date across all 2,703,430 records. The source Stata program traces `loan_vol` and `median_dummy` through last-payment-date grouping. The `issue_year` field was separately checked and matches origination year; it is **not** one of those five payment-derived fields. The 145-variable refits quarantine these five fields but do not certify the remaining variables' historical availability.

The separate NeurIPS analysis reconstructs different predictors and a different cohort. **Its results and claims about corrected point-in-time features must not be substituted for the Cogent experiments.**

### Preprocessing

The documented original scikit-learn preprocessing uses:

- Numeric fields: median imputation followed by `StandardScaler`.
- Categorical fields: most-frequent imputation followed by `OneHotEncoder(handle_unknown="ignore", max_categories=50)` with dense output.
- The fitted transformer is learned from the **full selected mature training cohort** and then applied unchanged to validation and test.
- For the nested training-size comparison, that same full-training-fitted transformer is applied to smaller training samples. Consequently, those comparisons are not independent end-to-end preprocessing runs for each subsample.

The original 150-variable representation produces 384 encoded inputs. Categorical handling includes state and sub-grade; rare categories can be grouped, and previously unseen validation/test categories are ignored.

### Models and training-size design

| Model | Documented original settings |
|---|---|
| SGD logistic regression | Log loss, L2 penalty, `alpha=1e-5`, `max_iter=100`, `tol=1e-4` |
| Histogram gradient boosting (HistGB) | Max depth 6, learning rate 0.05, 300 boosting iterations, seed 42 |
| Multilayer perceptron (MLP) | Hidden layers (128, 64), Adam, `max_iter=50`, early stopping with 10% internal validation and five-iteration patience |

Original nested training sizes are **180,000**, **360,000**, **540,000**, and **1,304,058**. The nested membership was reconstructed using one `numpy.random.default_rng(42)` permutation of the selected mature training frame and taking the first *n* rows for each condition. Original full-training model selection uses the **highest validation ROC-AUC**, which selects HistGB (validation AUC 0.6966) for downstream calibration and threshold work.

The original fitting seed is **42**. A separate, limited endpoint sensitivity reports **7, 42, and 101**; it is not a comprehensive estimate of training-pipeline variability or a nested tuning experiment.

### Primary test results (original 150-predictor specification)

All figures below are manuscript-reported values on the **41,145-loan selected test cohort**.

| Training n | Model | Test ROC-AUC | Test Brier | Test log loss |
|---:|---|---:|---:|---:|
| 180,000 | Logistic SGD | 0.6461 | 0.1575 | 0.5158 |
| 180,000 | HistGB | 0.6977 | 0.1344 | 0.5006 |
| 180,000 | MLP | 0.7189 | 0.1161 | 0.3887 |
| 1,304,058 | Logistic SGD | 0.6993 | 0.1294 | 0.4205 |
| 1,304,058 | HistGB | 0.6750 | 0.1367 | 0.5292 |
| 1,304,058 | MLP | 0.6463 | 0.1315 | 0.5032 |

The intermediate 360,000 and 540,000 results appear in manuscript Table `tab:training_size`. These figures illustrate a ranking reversal across training regimes **within the original predictor specification**; the alternative 77-variable specification does not preserve that reversal.

**Payment-date-quarantined 145-variable full-training test ROC-AUC:** Logistic SGD **0.7136**, HistGB **0.6969**, MLP **0.6520**. These remain estimates for the same selected test cohort, not prospective underwriting performance.

## Selection representativeness and censoring

Among test originations, **360,024 of 401,169** are excluded from the selected classification evaluation. Selection rates decline from **17.66% (April 2019)** to **4.17% (December 2019)**. Selected and excluded test loans differ on several observable characteristics:

| Characteristic | Selected | Excluded | Comparison |
|---|---:|---:|---|
| Mean loan principal (USD) | 14,465.69 | 16,625.59 | SMD −0.212 |
| Mean annual income (USD) | 86,555.82 | 85,722.99 | SMD +0.008 |
| Mean debt-to-income ratio | 19.42 | 21.04 | SMD −0.085 |
| Mean interest rate | 13.65% | 12.84% | SMD +0.157 |
| 60-month term | 25.72% | 33.42% | −7.70 percentage points |

Recorded adverse-status prevalence is **14.64%** in selected test loans compared with **2.96%** across all recorded test statuses. **The latter is not an unbiased fixed-horizon default comparator** because many unresolved loans are recorded as non-adverse at the available snapshot. The selection audit documents composition changes; it does not estimate a statistically corrected full-origination ROC-AUC, Brier score, or calibration curve.

A complementary, separate competing-risk analysis uses **2,700,690** loans with usable timing. It identifies **175,144** observed charge-offs, **765,339** full repayments, and **1,760,207** right-censored observations; an additional **2,740** charge-off records are excluded from that event-time construction for missing usable timing.

| Months since origination | Charge-off cumulative incidence | Full-repayment cumulative incidence |
|---:|---:|---:|
| 12 | 0.0413 | 0.1438 |
| 24 | 0.1061 | 0.3329 |
| 36 | 0.1419 | 0.6748 |

These cumulative-incidence estimates **do not validate the three fitted classification scores under censoring**. Fixed-horizon, censoring-aware classifier performance would require independently verified event/censoring histories and a harmonised endpoint.

## Validation, calibration, uncertainty, and decision policies

### Statistical comparisons

- Primary pairwise inference: **2,000 paired, outcome-stratified test-cohort bootstrap replicates** at the 180,000 and full-training endpoints. The same resampled test observations are used for compared models.
- Holm multiple-testing adjustment is reported for the **six specified endpoint ROC-AUC pairwise comparisons**, not for every exploratory analysis.
- Supplementary resampling: **1,500 state-cluster** and **1,500 origination-month-block** draws evaluated separately, holding fitted models and decisions fixed. There is no joint two-way clustering or end-to-end refitting bootstrap.
- The test cohort covers nine origination months; month-block uncertainty should be treated as exploratory.

### Probability recalibration

The original validation-selected **full-training HistGB** model undergoes logistic recalibration fitted on the **27,374 selected validation observations** and applied unchanged to the selected test observations:

`logit(p_cal) = a + b * logit(p_raw)`

where the manuscript reports `a = 1.4450` and `b = 1.0570`. Expected Calibration Error (ECE) uses **ten equal-width probability bins on [0, 1]**, excluding empty bins.

| Test diagnostic | Raw HistGB | Recalibrated HistGB |
|---|---:|---:|
| ROC-AUC | 0.6750 | 0.6750 |
| Brier | 0.1367 | 0.1232 |
| Log loss | 0.5292 | 0.4123 |
| ECE | 0.1188 | 0.0583 |
| Mean predicted adverse risk | 0.0277 | 0.0882 |

Selected-test recorded adverse-status prevalence is **0.1464**. Thus recalibration improves proper scores but the mean prediction remains below the selected-cohort recorded rate. The reported ten-bin reliability diagnostics should not be interpreted as evidence of calibration in sparse tails or every subgroup.

**Simple predictive benchmark:** Constant training-prevalence prediction has full-training selected-test Brier **0.125047**. Raw HistGB Brier **0.136718** is worse; recalibrated HistGB Brier **0.123161** is slightly better (approximately **+1.51% Brier skill**). These are retrospective selected-cohort comparisons; an incumbent industrial lender scorecard was unavailable.

### Misclassification costs and operating thresholds

The analytical loss is `C(t) = c_FN × FN(t) + c_FP × FP(t)` with **illustrative** false-negative to false-positive ratios **5:1, 10:1, and 20:1**, normalising `c_FP = 1`. These ratios are **not** measured lender losses, review staffing costs, or experimentally established intervention effects.

| FN:FP ratio | Validation-selected threshold | Selected-test FNR | Selected-test flag share | Model cost units | Flag-none units | Flag-all units |
|---|---:|---:|---:|---:|---:|---:|
| 5:1 | 0.1730 | 84.35% | 7.75% | 27,654 | 30,125 | 35,120 |
| 10:1 | 0.0962 | 41.36% | 37.53% | 36,829 | 60,250 | 35,120 |
| 20:1 | 0.0327 | 4.90% | 85.06% | 35,166 | 120,500 | 35,120 |

These are **classification-loss proxies**, not observed economic profit or real human-review decisions. At 10:1 and 20:1, the model policy does **not** outperform the flag-all classification rule under the stated simplified loss.

A hypothetical intervention sensitivity calculates the required prevention effectiveness *under zero review cost* for the original rule to beat flag-none: **47.59%** (5:1), **33.71%** (10:1), and **25.54%** (20:1). These are algebraic break-even conditions, **not measured intervention effectiveness**.

### Alternative disjoint-window check

An explicitly **post-hoc** validation-stage separation uses:

| Stage | Window | Selected loans | Recorded adverse outcomes |
|---|---|---:|---:|
| Family selection | July–September 2018 | 15,090 | 2,064 |
| Probability recalibration | October–December 2018 | 8,599 | 609 |
| Threshold selection | January–March 2019 | 3,685 | 98 |

It selects a different model family and an unstable decision policy, including a test flag rate of about **0.012%**. This illustrates sensitivity to stage separation and sparse outcome support. It is not a prospectively pre-registered or independently refitted validation pipeline.

### Subgroup and governance checks

The high-state-unemployment subgroup is defined using **state unemployment rate ≥ 4.9%** in the existing study, with **641 selected test loans**, **74 recorded adverse outcomes**, and representation from only **four jurisdictions**. Operational grade-by-term cells are reported only when both groups have **at least ten adverse outcomes and ten non-events**; the four supported cells are C–36 months, C–60 months, D–36 months, and D–60 months. State and grade are operational descriptors, **not protected demographic attributes**.

Retrospective governance illustrations apply post-hoc alert examples, including a 50% maturity floor, two-percentage-point calibration-gap ceiling, 25% FNR ceiling, and 40% flag-capacity ceiling. The alerts are **if–then demonstrations**, not evidence that institutions deployed, enforced, or benefited from these controls.

## Manuscript-to-evidence map

The identifiers below are LaTeX `\label{...}` anchors from the manuscript, **not a claim that corresponding CSV files are already uploaded**.

| Manuscript analysis | Table/figure labels | Evidence or computation to publish |
|---|---|---|
| Cohort reconstruction and maturity | `tab:flow_mature`, `tab:maturity` | Eligibility script; split/selection counts; label/status manifests |
| Selection representativeness | `tab:selection_add`, `tab:selection_balance` | Monthly selection and selected/excluded covariate audit |
| Competing risks | `tab:competing_risk` | Event-time construction, competing-risk script, risk-set and interval outputs |
| Training-size comparison | `tab:training_size`, `fig:training_size_auc` | Original fit scripts, feature specification, nested membership, predictions and metrics |
| Seed and predictor sensitivity | `tab:r2_seed_sensitivity`, `tab:r2_conservative_features` | Seed 7/42/101 endpoints, 77-feature manifest and fits |
| Payment-date quarantine | `tab:145_quarantine` | Stata dependency audit, date match, 145-feature refits |
| Paired inference and clustered checks | `tab:paired_bootstrap`, `tab:supp_clustered` | Bootstrap scripts, sampled units, settings and interval tables |
| Recalibration and risk bands | `tab:recalibration`, `tab:r2_reliability_bins`, `fig:recalibration_reliability` | Validation calibrator, test probabilities, bin assignments and metric code |
| Baselines | `tab:r2_brier_baselines`, `tab:r2_policy_benchmarks` | Constant forecast and flag-all/flag-none computations |
| Thresholds and cost sensitivity | `tab:cost_threshold`, `fig:threshold_review_tradeoff` | Validation-only threshold selection and hypothetical review-effect calculations |
| Subgroups | `tab:supported_slices`, `tab:supp_geographic` | Outcome counts, Wilson intervals and jurisdiction support |
| Governance controls and alerts | `tab:r2_governance_alerts_main`, `tab:controls`, `tab:supp_governance` | Illustrative alert definitions and retrospective evaluation |
| Overall provenance and validation uncertainty | `tab:reanalysis_manifest`, `tab:supp_selection` | Experiment manifest and model-selection bootstrap |

### Exact local audit filenames identified in the manuscript

The manuscript identifies the following as **local** files or evidence outputs. **Check each file's presence and contents before advertising it as a public download.**

- `cell26_locked_feature_specification.csv`: locked feature specification (153 original fields before three exclusions).
- `cell26_feature_missingness.csv`: missingness audit required for replication.
- `cell29_test_reliability_after.csv`: original recalibrated reliability-bin output.
- `31_conservative_feature_manifest.csv`: conservative 77-variable audit.
- `36_requirements_round2.txt`: versions recorded for the additional Round 2 audits.
- `44_holm_BH_adjusted_six_test_auc_comparisons.csv`: six-test AUC multiplicity adjustment.
- `45_three_window_validation_sensitivity.csv`: disjoint-stage post-hoc sensitivity.
- `46_break_even_review_cost_bounds.csv`: hypothetical review-effect/cost calculations.
- `53_temporal_feature_date_comparison.csv`: date comparisons from source audit.
- `56_corrected_payment_history_dependency_register.csv`: corrected predictor lineage.
- `57_145_feature_quarantine_sensitivity.csv`: 145-predictor model refits.
- `58_monthly_outcome_selection.csv`: origination-month selection audit.
- `59_selected_vs_excluded_numeric_balance.csv`: numeric selected/excluded comparison.
- `60_replication_readiness_audit.json`: archive preflight/readiness audit.
- `61_independent_archived_auc_reproduction.csv`: archived AUC recalculations.

Other analysis steps are referenced by **step number** in the paper (including Steps 22, 23, 30, 32, 33, and 34), but their precise script filenames and public paths are not established from the supplied manuscript. Do not invent paths when preparing the release.

## Reproduction workflow

**Important:** The sequence below describes the intended study-specific work. It is **not** a tested one-command reproduction workflow. A runnable command should be published only after the original Cogent notebook/scripts, relative input paths, dependencies, and outputs have been verified. **Do not run the NeurIPS 2026 notebook as if it reproduced these Cogent results.**

1. **Retrieve original data:** Download Mendeley Data version **3** via DOI `10.17632/wb3ndt69gf.3`; retain and verify the source Stata file and source construction materials. Record a file hash, extraction date, and any data conversion.
2. **Recreate original cohorts:** Implement and log the August 2012–December 2019 chronological windows, 1 April 2020 analytical maturity rule, source `badloan` mapping, row ordering, exclusion flags, and the exact selected counts in the table above.
3. **Recreate the feature matrix:** Use the locked 150-feature specification and documented transformation settings; record encoded column count, fit/sample membership, missingness, and preprocessing objects. Quarantine payment-history-dependent variables only for the labelled 145-variable sensitivity, without changing original reported experiment definitions.
4. **Re-fit model candidates:** Run the three models at the four nested training sizes using the documented settings and seed 42. Keep model selection based on validation AUC; do not use test labels to choose a family or tune a threshold.
5. **Reproduce model metrics:** Calculate ROC-AUC, Brier and log loss for the fixed 27,374 validation and 41,145 test selected cases; retain the prediction order and label arrays. Compare against the manuscript's twelve AUCs and reported proper scores.
6. **Reproduce additional audits:** Execute the three-seed endpoint check, feature-quarantine refits, paired/bootstrap comparisons, state/month-block sensitivity, monthly selection and covariate balance, and baseline comparisons from their corresponding verified scripts.
7. **Reproduce calibration and decision results:** Fit logistic calibration on the original selected validation window, apply it unchanged to test, use ten equal-width ECE bins, select thresholds by validation cost minimisation, and compare against flag-none/flag-all. Keep the post-hoc disjoint-window sensitivity separate from the original policy results.
8. **Reproduce the distinct competing-risk analysis:** Apply the charge-off/fully-paid/right-censoring endpoint definition using the documented proxy timing and compare counts and cumulative incidence. Do not treat this as classifier validation.
9. **Publish a machine-readable output manifest:** For each manuscript table, record source version/hash, script/notebook path and revision, random seed, row-selection logic, feature list, environment, output path/hash, and the outcome of an independent numeric check.

### Environment information

The **additional Round 2 audit** was documented as having run on Windows with:

| Package | Reported Round 2 audit version |
|---|---|
| Python | 3.14.7 |
| NumPy | 2.5.2 |
| pandas | 2.3.3 |
| scikit-learn | 1.9.0 |
| SciPy | 1.18.1 |
| Matplotlib | 3.11.1 |
| joblib | 1.5.3 |
| lifelines | 0.30.3 |
| statsmodels | 0.15.0 |
| threadpoolctl | 3.6.0 |

These are **Round 2 audit versions**, **not independently established package versions for the original 150-feature training runs**. The manuscript identifies `36_requirements_round2.txt`; it does **not** establish a fully verified original-training dependency lock. Do not re-label the NeurIPS `requirements.txt` as the Cogent environment without verification.

## Reproducibility status

| Component | Evidence described in manuscript | Independent/public status |
|---|---|---|
| Source dataset | Version 3 DOI and source filename specified | Public source available via Mendeley Data; local extraction/hashes must be checked |
| Sample-size consistency | Chronological and selected counts reconciled; output-array lengths match | Verified in local audit; not independently re-run here |
| Original settings and preprocessing | Feature counts, pipeline settings and deterministic sample rule documented | Manuscript-described, but complete original training environment not confirmed |
| Numeric output verification | All **12 original test ROC-AUCs recalculated from archived predictions**, agreeing to six decimals | **Archived metric recalculation verified**; not a fresh model fit |
| Ancillary scripts and audit tables | Evidence transfer archive described as 56 entries; earlier candidate archive as 197 entries including 16 Python and two Stata scripts | Archive contents/integrity reviewed locally; completeness in current public GitHub location not verified |
| Additional audit environment | Package versions and code-lineage references recorded | Version manifest described; public packaging and lock verification pending |
| Fresh end-to-end original model fitting | Not demonstrated in manuscript | **Not independently verified** |
| Point-in-time feature availability | Five payment-date-dependent fields identified and quarantined in sensitivity | Other retained variables not certified available at underwriting time |
| Full-origination performance | Outcome-selection/censoring diagnosed and competing-risk incidence analysed | No corrected, fixed-horizon classifier validation for all originations |
| Public Cogent replication package | Manuscript repository URL supplied | **Not yet verified complete**; adding this README alone does not release the missing assets |

The manuscript explicitly separates **reproducing a metric from saved scores** from **independently reconstructing the complete training pipeline**. That distinction must be retained in journal correspondence and public documentation.

## Release checklist

Before representing this repository as a complete Cogent replication package, confirm each item against the **actual files in GitHub**, not only against this README.

- [ ] Publish the exact original Cogent model-training notebook or scripts (distinct from the NeurIPS notebook).
- [ ] Publish `cell26_locked_feature_specification.csv` and source-to-model feature mapping, including excluded fields.
- [ ] Publish or link to the applicable source Stata construction materials and document their provenance.
- [ ] Publish data acquisition/extraction and chronological/eligibility scripts with stable row ordering.
- [ ] Publish the exact code or exported objects for preprocessing, three model families, and four sample sizes.
- [ ] Publish calibration, ten-bin ECE, cost-threshold, trivial-baseline, and uncertainty scripts.
- [ ] Publish the Round 2 audit tables and their source scripts with a manuscript table-to-file map.
- [ ] Publish a verified environment manifest for **original training**; keep it distinct from Round 2 audit versions.
- [ ] Confirm archived AUC metric checks and clearly label archived-score verification versus fresh training.
- [ ] Perform and document a clean-environment end-to-end replay where possible; otherwise specify which steps cannot be independently rerun.
- [ ] Record release commit/tag, file hashes, provenance, data licence, and reproducibility status.
- [ ] Update the Cogent manuscript's Code Availability statement to a **Cogent-specific file or folder link**, once the assets are confirmed public.

## Recommended placement alongside the NeurIPS README

Upload this document at the **root of the existing GitHub repository** as:

`README_Cogent_Economics_Finance.md`

This leaves the existing root `README.md` for NeurIPS **unchanged**. The filename is intentionally different, so GitHub will not replace the default README. A future optional Cogent-specific directory can hold the original scripts and evidence once their exact locations are confirmed. There is no need to rename or relocate the existing NeurIPS notebook merely to add this file.

## Scope and responsible interpretation

This is a retrospective case study of **originated loans** from one US platform and historical period. It does not observe rejected applicants, protected demographic attributes, actual credit-review interventions, lending profit/loss or staffing capacity. Reported cost ratios are analytical scenarios. A state-unemployment slice is a geographic operational grouping, not proof of causal macroeconomic effects or protected-class discrimination. The manuscript proposes accountable policy controls; it does not show a prospective improvement in customer outcomes caused by deploying them.

## Citation

Please cite the **Cogent manuscript** if using this study's code or evidence. As of the version documented here, no journal volume, page range, acceptance, or publication DOI has been supplied for this manuscript; do not invent one.

```bibtex
@unpublished{singh_cogent_ai_governance,
  title  = {AI Governance for Credit Decision Systems in P2P Lending},
  author = {Singh, Kumar Utkarsh and Varadarajan, Jayasree and Sufian, Md Abu and Rana, Md Aminul Islam and Qazi, Nadeem and Berardinelli, Nabeela},
  note   = {Manuscript submitted to Cogent Economics & Finance; revision under review}
}
```

Please also cite the original dataset:

```bibtex
@misc{nigmonov2021p2p,
  author    = {Nigmonov, A.},
  title     = {Dataset from the US Peer-to-peer Lending Platform with Macroeconomic Variables},
  publisher = {Mendeley Data},
  version   = {3},
  year      = {2021},
  doi       = {10.17632/wb3ndt69gf.3}
}
```

## Data and code licensing

The Mendeley source dataset is distributed by its original depositor under **CC BY 4.0**. Code licensing is governed by the licence that actually accompanies each repository file; this README does not independently change copyright or licence permissions. The raw source dataset is not included here.

## Contact

For manuscript-specific questions or evidence-file provenance, contact **Md Abu Sufian** at [m.sufian@uel.ac.uk](mailto:m.sufian@uel.ac.uk).


## Download and place the source data

1. Download version 3 from the [original Mendeley Data record](https://doi.org/10.17632/wb3ndt69gf.3). Obtain `wb3ndt69gf-3.zip` from that record. The Cogent folder uses the original data source for distribution.
2. Extract the archive locally. Keep the source filename `P2P_Macro_Data.dta` and retain the accompanying Stata scripts for reference.
3. Before running either Cogent notebook, check its data-loading cell and point the input path to your local copy of `P2P_Macro_Data.dta`. Replace any machine-specific paths with paths on your computer. The source archive and the extracted Stata dataset are both stored outside Git.
