# Cogent Economics & Finance: Round 2 replication notes (DRAFT)

**Study:** *AI Governance for Credit Decision Systems in P2P Lending* (Cogent Economics & Finance revision)

**Status, 5 November 2026:** Documentation for the existing **Cogent** analysis and Round 2 sensitivity checks. This is **not** evidence of independent end-to-end reproducibility, and it is **not** the replication archive for the NeurIPS 2026 MusIML paper. Check all filenames and publish the required files before describing the revised Cogent analysis as publicly reproducible.

## 1. Keep the two studies distinct

The existing public GitHub ZIP `Government-ai-policy-credit-decisions.zip` documents a separate NeurIPS 2026 MusIML camera-ready reanalysis using **264,344** 2015-vintage, 36-month loans, a **38-raw-feature/106-encoded-feature** origination-aligned design, and seeds **17, 42, 2026**. It does **not** document the present Cogent paper's **1,304,058** selected training loans, **41,145** selected test loans, original **150-input-variable/384-encoded-feature** models, or Round 2 robustness exercises. Shared source data and overlapping authors do not make their estimands, models, or results interchangeable.

A repository root also contains `Governance-Policy/Fintech_25_12_2025.ipynb`. Its presence is not, by itself, proof that every reported Cogent result can be regenerated. Audit its cells and match its outputs to the submitted Cogent tables before making a completeness claim.

## 2. Data provenance and the conditional target

- Public source: A. Nigmonov, *Dataset from the US Peer-to-peer Lending Platform with Macroeconomic Variables*, **Mendeley Data version 3**, DOI `10.17632/wb3ndt69gf.3`, with **2,703,430** records from August 2012 through December 2019. Version 4 was not used.
- Original source construction materials: `P2P_Macro_Codes.do` and `Labels.do`, alongside `P2P_Macro_Data.dta`. Obtain these from the source data archive; do not redistribute the raw dataset unless its licence and any repository constraints permit it.
- Original classification label: source `badloan`, converted to numeric with missing values filled with zero in the original modelling implementation. Check and disclose its source status mapping. In the source code, `badloan` includes late/delinquent statuses as well as charged-off/default loans; it is **not** a consistent fixed-horizon default indicator.
- Original cut-off for the reconstructed maturity rule: **1 April 2020**. A record is selected if its observed status is terminal (`Fully Paid`, `Charged Off`, `Default`) **or** calendar time since origination equals/exceeds its contractual 36- or 60-month term. This is **outcome-dependent selection** for recently originated loans and must not be described as an outcome-independent, representative sample.
- Original chronological partitions: train Aug 2012–Jun 2018, validation Jul 2018–Mar 2019, test Apr 2019–Dec 2019. Full and selected counts, respectively: train **1,931,267 / 1,304,058**; validation **370,994 / 27,374**; test **401,169 / 41,145**.
- The selected test cohort has **6,025** adverse labels (**14.6433%**); **10.26%** of the original 401,169 chronological test records were selected. Do not generalise the selected-test prevalence, ROC-AUC, calibration, or cost proxy to all test-period originations.

## 3. Original modelling specification

- The locked original list contains **153** input fields before excluding three high-cardinality text fields (`emp_title`, `desc`, `title`), leaving **150** inputs: **136 numerical**, **14 categorical**. The original preprocessing produces **384** encoded columns. Deposit the exact `cell26_locked_feature_specification.csv` and `cell26_feature_missingness.csv` and make the field-order and exclusion rules explicit.
- Numeric pipeline: median imputation then `StandardScaler`. Categorical pipeline: most-frequent imputation then `OneHotEncoder(handle_unknown="ignore", max_categories=50)`, dense output.
- **Important training-size detail:** preprocessing was fitted on the full selected training cohort, then applied to the 180,000 / 360,000 / 540,000 / 1,304,058 estimator training subsets. This does not constitute independently refitted subset pipelines.
- Original models: SGD logistic regression, histogram gradient boosting, and MLP; original sample membership generated with seed **42**. Preserve estimator settings and source notebook when depositing the original experiment.
- The source fields and Stata construction code raise significant **decision-time provenance and temporal-alignment concerns**. In particular, `month` and `quarter` and some macroeconomic variables were constructed relative to last-payment timing. A sensitivity experiment that drops uncertain fields is not proof that the original models were leakage-free. The 77-variable conservative set has **not** been independently validated as fully origination-time safe.

## 4. Existing Round 2 files and what they establish

Local structure (confirm and upload files; these paths are **not** a claim that they are already available publicly):

```
Round2_Manuscript_Results/
  01_Tables/                         # labelled output CSVs
  03_Statistical_Results/            # supporting statistical outputs
  05_Reproducibility/
    Code/                             # executed step scripts (include dependencies)
    Execution_Logs/                   # audit and completion reports
    36_requirements_round2.txt       # current software environment snapshot
revision_outputs/                    # original saved predictions, labels, tabulations
```

Key files/steps:

| Evidence | Local script or output | Scope and caveat |
|---|---|---|
| Original sample and modelling audit | Early Round 2 audit scripts and original notebooks | Reconstructed selected samples; outcome-dependent maturity condition |
| Cost benchmarks | Step 17 outputs | 5:1, 10:1 and 20:1 cost scenarios; classification-error proxies, not realised manual-review costs |
| Proper-score comparison | Step 18 outputs | Raw-model Brier scores vs constant-risk baseline; recalibrated HistGB Brier 0.123161 |
| Original feature timing | `30_Reviewer2_Comment5_Stata_Source_Audit.txt` and Stata source | Temporal-alignment concerns; not a verified decision-time lineage |
| Training vintage mix | `20_training_vintage_composition.py` and Step 20 outputs | Aggregate annual vintage mix does not explain all ranking instability |
| Training seed sensitivity | `22_training_seed_stability.py` and Step 22 outputs | Three seeds only; selected model and training-size results vary |
| Validation model selection | `23_validation_auc_selection_uncertainty.py` and Step 23 outputs | HistGB AUC lead over alternatives includes zero in paired validation bootstrap |
| Predictor set sensitivity | `31_conservative_predictor_sensitivity.py` and `31_conservative_feature_manifest.csv` | Removing 73 uncertain fields changes model ranking; set not independently certified leakage-free |
| Selected-test cluster bootstrap | `32_clustered_test_uncertainty_v2.py`, `32_clustered_test_uncertainty_v2.csv` | 1,500 state-cluster and month-block resamples of **fixed predictions**; does not include model-fitting uncertainty or adjust for multiplicity |
| Unemployment support | `33_subgroup_cluster_support_audit.py`, `33_high_unemployment_support_by_state.csv` | 641 high-unemployment loans and 74 adverse cases from **four** jurisdictions; state-cluster subgroup inference fragile |
| Illustrative governance alerts | `34_illustrative_governance_trigger_audit.py`, `34_illustrative_governance_trigger_audit.csv` | Retrospective example thresholds only; no observed governance intervention or operational effect |
| Calibration by risk bin | `revision_outputs/cell29_test_reliability_after.csv` and validation counterpart | 0.1-width bins; residual risk underprediction, sparsely supported high-risk bins |

All checks use original arrays or source data as documented in each script. The Round 2 bootstraps are conditional uncertainty assessments, **not** clean-room reproduction of the original training pipeline.

## 5. Environment and reproducibility instructions

The current Windows environment used to run Round 2 audits reports Python **3.14.7**, NumPy **2.5.2**, pandas **2.3.3**, scikit-learn **1.9.0**, SciPy **1.18.1**, Matplotlib **3.11.1**, joblib **1.5.3**, lifelines **0.30.3**, statsmodels **0.15.0**, and threadpoolctl **3.6.0**. A package snapshot was saved as `Round2_Manuscript_Results/05_Reproducibility/36_requirements_round2.txt`. These are **not verified** as the versions used for the original models. Do not replace the archived NeurIPS environment statement (Python 3.14, scikit-learn 1.8) with these versions or claim cross-study equivalence.

For a complete Cogent release, provide a documented directory layout that places original source-data files, original notebooks, original `revision_outputs` arrays, all Step 14–34 scripts and resulting CSV/logs at the paths expected by the executed scripts. Document exact execution order and any environmental/configuration paths. Scripts that rely on the saved original predictions reproduce **audits of the existing models**; full end-to-end model reproduction additionally requires the original complete notebook, exact estimator code and feature/target definitions, environment, and raw data.

## 6. Required release verification before submission

1. Place the **Cogent-specific** notebook, 150-feature manifest, extraction and status mapping documentation, original predictions/results, Round 2 scripts, tables and logs in a Cogent-specific repository folder or separate archive; do not replace the NeurIPS code or tables.
2. Record the original modelling software versions if recoverable. If unavailable, explicitly say so. Keep the Round 2 environment snapshot separately labelled.
3. Check each published Cogent table and figure against its named source output, and make a machine-readable mapping from manuscript result to script and output file.
4. Check that the public repository URL and relevant commit actually contain the files described as available. A GitHub ZIP containing a different study is insufficient.
5. State clearly that raw source data come from Mendeley Data v3 and that no private loan-level applicant rejection data, decision-time information timestamps, or observed governance interventions were available.
6. Do not make a blanket full-reproducibility or leakage-free claim until those claims have been verified.

## 7. Manuscript data/code availability language (pending deposit)

> The source data are available from Mendeley Data, version 3 (DOI: 10.17632/wb3ndt69gf.3). Original modelling notebooks and saved outputs, together with supplementary scripts for the Round 2 audit, constitute separate replication materials. The publicly linked repository currently includes materials for a related NeurIPS 2026 reanalysis but does not yet establish a complete, independently verified replication archive for the present Cogent study. Original training software versions and historical decision-time predictor availability have not been fully verified. Complete Cogent-specific files and a checked repository commit should be identified before a full public reproducibility statement is made.

*DRAFT ONLY: replace the above with a precise public directory/commit and a verified statement after deposit.*
