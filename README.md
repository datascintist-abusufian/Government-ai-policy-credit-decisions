# Government AI Policy: Outcome Maturity, Calibration, and Threshold Governance in Credit Decisions

Reproducibility code for the camera-ready paper accepted at the **MusIML Workshop (Track 5: ML Research by Muslim Scholars), NeurIPS 2026**.

**Authors:** Md Abu Sufian, Kumar Utkarsh Singh, Jayasree Varadarajan, Muhammad Babar, Md Aminul Islam Rana, Nadeem Qazi, Nabeela Berardinelli

The repository covers the full analysis behind the paper: outcome-independent cohort construction, the temporal-leakage audit and correction, chronological model comparison, paired DeLong inference, proper-score skill and validation-fitted recalibration, cost-sensitive threshold governance with bootstrap uncertainty, and a predeclared grade-slice audit.

---

## Overview

AI credit models are often governed through aggregate accuracy or discrimination metrics. Their operational consequences, though, also depend on how the outcome is defined, how well the probabilities are calibrated, which decision threshold is used, how the data are aligned in time, and how the portfolio is composed. This study revisits those issues on **version 3 of a public archive of 2,703,430 US peer-to-peer (LendingClub) loans** with macroeconomic variables.

Two corrections to the original evaluation shape the analysis:

1. **Outcome-independent cohort.** The original rule let a loan's terminal status confer "outcome maturity", which preferentially selected early terminations in recent vintages. It is replaced by **264,344 fully observed 36-month loans from 2015 contractual vintages**. Eligibility is decided at the vintage level, before any individual outcome is looked at.
2. **Temporal-leakage correction.** An audit of the source Stata code (`P2P_Macro_Codes.do`) showed that `month`, `quarter` and all macroeconomic fields were aligned to the **last-payment date** (`last_date`), which is post-origination information. They are rebuilt from `issue_date` and borrower state before any model is fitted.

## Key results (leakage-corrected, mean ± SD over seeds 17, 42, 2026)

| Model | Val AUC | Test AUC | Test AP | Test Brier | Test log loss |
|---|---|---|---|---|---|
| SGD logistic | 0.6775 ± 0.0079 | 0.6628 ± 0.0109 | 0.2401 ± 0.0068 | 0.1426 ± 0.0269 | 0.4489 ± 0.0615 |
| **HistGB** | **0.6977 ± 0.0005** | **0.6971 ± 0.0014** | **0.2790 ± 0.0013** | **0.1183 ± 0.0001** | **0.3882 ± 0.0004** |
| MLP | 0.6920 ± 0.0015 | 0.6473 ± 0.0220 | 0.2415 ± 0.0103 | 0.1310 ± 0.0038 | 0.4771 ± 0.0413 |

- **Model family (P1):** HistGB is chosen by the pre-specified validation-AUC rule. Paired DeLong tests on the shared test cohort favour it over SGD logistic regression (ΔAUC 0.024–0.045, all p < 10⁻⁶⁶) and over the MLP (ΔAUC 0.036–0.075, all p < 10⁻⁹⁵) for every seed.
- **Probability quality:** HistGB beats a constant test-prevalence predictor, with Brier skill 0.0636 ± 0.0008 and log-loss skill 0.0754 ± 0.0011. Its calibration slope is about 1 (1.051 ± 0.009), and validation-fitted logistic recalibration does not materially improve either proper score.
- **Threshold governance (P3):** For cost ratios r = c_FN / c_FP = 5, 10 and 20, the Bayes thresholds t* = 1/(1+r) (0.167, 0.091, 0.048) are close to the validation-selected HistGB thresholds (0.167, 0.097, 0.052). Normalised decision cost stays below the best constant (no-model) policy for every r from 2 to 20 (300 bootstrap resamples).
- **Predeclared slices (P4):** Calibration and decision value differ across all seven credit grades A–G. The high false-negative rate in grade A follows from its 5.29% base rate sitting below the r = 10 threshold, so it is not by itself evidence of model failure.
- **Stress regime:** The stress threshold is the training-period 75th percentile of state unemployment (aligned to origination), which is 6.20%. The test cohort has 3,069 stress loans with a 16.13% default rate, against 14.79% outside stress. These figures are descriptive only.
- **Leakage diagnostic:** With the original last-payment-aligned fields, mean test AUCs rise to 0.7413 (SGD), 0.7517 (HistGB) and 0.7697 (MLP). These values are reported **only** as a leakage diagnostic.

All manuscript tables are provided as CSV files in [`results/tables/`](results/tables).

## Repository structure

```
.
├── README.md
├── LICENSE                       # MIT (code only; the dataset is CC BY 4.0)
├── CITATION.cff
├── requirements.txt
├── .gitignore                    # raw data and extracted archives are never committed
├── data/
│   └── README.md                 # how to download and place the Mendeley archive
├── notebooks/
│   └── NeurIPS_2026_CameraReady_Reanalysis.ipynb   # full analysis, with saved outputs
├── docs/
│   └── notebook_map.md           # notebook step → manuscript section/table
└── results/
    └── tables/                   # manuscript tables as CSV
        ├── table1_cohort.csv
        ├── table2_model_comparison.csv
        ├── table3_proper_score_skill.csv
        ├── table4_threshold_governance.csv
        ├── table5_grade_audit.csv
        └── table6_cost_continuum_histgb.csv
```

## Notebook layout

The notebook is meant to be run **top to bottom**, because later cells reuse in-memory objects from earlier ones. It has five parts:

| Part | Steps | Role in the paper |
|---|---|---|
| **0. Setup** | configuration, archive extraction | — |
| **A. Outcome and cohort audit** | W1 Steps 1–19 | Final cohort (Table 1, Appendix A) |
| **B. Provisional models on source-aligned fields** | W1 Steps 20–25; Steps 26–34 | **Superseded.** Used only for the post-origination-field diagnostic |
| **C. Temporal-alignment (leakage) audit** | W6 Steps 35B–38 | Motivates the rebuild; defines the stress threshold |
| **D. Leakage-free rebuild and final results** | Steps 39–44; Final W2 / W4 / W5 | **All reported results** (Tables 2–6) |

[`docs/notebook_map.md`](docs/notebook_map.md) maps each step to the section or table it supports.

## Getting started

### 1. Environment

The analysis was run with Python 3.14 and scikit-learn 1.8.

```bash
git clone https://github.com/datascintist-abusufian/Government-ai-policy-credit-decisions.git
cd Government-ai-policy-credit-decisions
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Data

Download **version 3** of the dataset (DOI [10.17632/wb3ndt69gf.3](https://doi.org/10.17632/wb3ndt69gf.3)) and save the archive as `data/raw/wb3ndt69gf-3.zip`. Full instructions are in [`data/README.md`](data/README.md).

If you keep the archive somewhere else, set an environment variable:

```bash
export GOV_AI_DATA_DIR=/path/to/folder/containing/zip      # Windows: set GOV_AI_DATA_DIR=E:\path
```

### 3. Run

```bash
jupyter lab notebooks/NeurIPS_2026_CameraReady_Reanalysis.ipynb
# or headless:
jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=-1 \
    notebooks/NeurIPS_2026_CameraReady_Reanalysis.ipynb --output executed.ipynb
```

**Resources.** The extracted `.dta` file is about 2.8 GB and is read in chunks of 250,000 rows. At least 16 GB of RAM is recommended. The bootstrap cells (Final W4 and W5) are the slowest part of the run.

## Experimental design (summary)

| Item | Setting |
|---|---|
| Endpoint | Final recorded status. `Charged Off` and `Default` → 1; all other statuses → 0. This is not a time-to-default endpoint |
| Cohort | 36-month loans from 2015 vintages with ≥ 36 months of observable follow-up |
| Splits | Train Jan–Jun 2015 (n = 120,766; 15.12%) · Validation Jul–Aug (n = 54,794; 14.46%) · Test Oct–Dec (n = 88,784; 14.84%). September 2015 is excluded as a whole vintage |
| Features | 38 origination-time raw predictors → 106 encoded features. Constant `term` and `issue_year` and zero-variance `riskprem` are removed |
| Preprocessing | Fitted on training data only: median / most-frequent imputation, standardisation, one-hot encoding |
| Models | SGD logistic (L2, α = 1e-5) · HistGradientBoosting (lr 0.05, depth 6, ≤ 300 iterations, early stopping) · MLP (128-64, ReLU, Adam, early stopping). No class weighting |
| Seeds | 17, 42, 2026 |
| Model selection | Highest mean validation ROC-AUC (pre-specified) |
| Inference | Paired DeLong on the shared test cohort |
| Calibration | logit(p_cal) = a + b·logit(p), fitted on validation and applied unchanged to test. Reports Brier/log-loss skill vs. a constant-prevalence predictor, intercept, slope and 10-bin ECE |
| Decisions | Bayes threshold t* = 1/(1+r) vs. a validation-selected threshold. Cost is normalised by the better of approve-all and reject-all. 300 bootstrap resamples, with the threshold re-selected inside each one |
| Slices | All seven grades A–G, declared in advance. 500 within-grade bootstrap resamples at r = 10 |
| Stress | `unem_rate` realigned to origination month × state. Stress means ≥ the training 75th percentile (6.20%) |

## Scope and limitations

- Rejected applicants are not observed, so there is no reject inference.
- The archive has no reliable default-event timestamp, so this is not a survival or competing-risks analysis.
- Macroeconomic values are aligned to the origination month. Their within-month publication timing is unknown.
- Evidence comes from a single US P2P platform. The dataset contains no religious-affiliation or Shariah-contract information, and **no inference is made about Muslim borrowers, Shariah-compliant products or Australian credit markets**.
- Credit grade is an operational slice, **not** a legally protected attribute.
- The governance mapping in the paper is an illustrative template. Its trigger limits have not been prospectively validated.

## Citation

If you use this code, please cite the paper (see also [`CITATION.cff`](CITATION.cff)):

```bibtex
@inproceedings{sufian2026governmentai,
  title     = {Government {AI} Policy: Outcome Maturity, Calibration, and Threshold Governance in Credit Decisions},
  author    = {Sufian, Md Abu and Singh, Kumar Utkarsh and Varadarajan, Jayasree and Babar, Muhammad and Rana, Md Aminul Islam and Qazi, Nadeem and Berardinelli, Nabeela},
  booktitle = {MusIML Workshop at the Conference on Neural Information Processing Systems (NeurIPS)},
  year      = {2026}
}
```

Please also cite the dataset:

```bibtex
@misc{nigmonov2021dataset,
  author    = {Nigmonov, A.},
  title     = {Dataset from the US Peer-to-peer Lending Platform with Macroeconomic Variables},
  version   = {3},
  publisher = {Mendeley Data},
  year      = {2021},
  doi       = {10.17632/wb3ndt69gf.3}
}
```

## License

The code is released under the [MIT License](LICENSE). The dataset is © its authors and is distributed by Mendeley Data under **CC BY 4.0**. It is not redistributed in this repository.

## Contact

Md Abu Sufian, Department of Computer Science, University of East London, Lab ([[databizanalyst.com](https://databizanalyst.com/CardioAi_Lab.html)]([https://databizanalyst.com](https://databizanalyst.com/CardioAi_Lab.html)))
