# Reviewer comments: evidence and remaining requirements

Manuscript: **AI Governance for Credit Decision Systems in P2P Lending**

This audit distinguishes a *complete manuscript response* from an *empirical resolution*. All changes retain the previously reported data, tests, numerical tables, figure artwork and computational procedures. No new model was trained. Newly derived approximately flagged record counts are the pre-existing test size multiplied by the pre-existing rounded flag rates.

| Issue | Addition to manuscript | Evidence boundary still present |
|---|---|---|
| Reviewer 1 & 2: outcome maturity and selection | Explicit selection-conditioned estimand and a release restriction; excluded validation/test denominators stated. | An unbiased all-origination test requires verified status history and independent fixed-horizon or censoring-aware model validation. |
| Reviewer 2: classification target | Exact positive-status list from source Stata, meaning of zero-status labels and mismatch with competing-risk charge-off endpoint. | No common verified default horizon, so cannot claim fixed-horizon probability of default. |
| Reviewer 2: competing-risk classifier validation | Explicit statement that observed cumulative incidence is not validation of the three fitted classifiers. | Time-dependent classifier AUC/calibration and well-observed event times are not available in completed analyses. |
| Reviewer 2: clustered uncertainty and multiplicity | Distinguishes paired-loan resampling, state clustering and 9-month block resampling, and describes why no multiplicity-adjusted claim follows. | No joint two-way bootstrap, pipeline-level uncertainty, or familywise multiple-comparison correction. |
| Reviewer 2: sparse subgroup support/confounding | Describes the 4 jurisdictions, effective 3.26/3.32 support and nearly state-constant unemployment contrast; explicates identification failure. | No identified unemployment effect, demographic fairness conclusion or unconfounded state-adjusted comparison. |
| Reviewer 2: governance efficacy | Retrospective, falsifiable if-then trace and specification of prospective breach logging and comparator. | No observed intervention, causal effectiveness or regulator-approved tolerance. |
| Reviewer 2 & 3: cost and review capacity | Distinguishes classification loss from money; supplies ~3,187 / 15,442 / 34,996 retrospective flag counts; introduces external review capacity K. | No lender-specific cost matrix, staffing throughput, review effectiveness or observed consumer harm. |
| Three-seed table | Adds missing explicit table cross-reference. | No numeric results altered. |
| Editorial and figures | Moves existing three supplementary quantitative figures beside their corresponding results, removes an exact duplicated oversight sentence, fixes an author-name typo, corrects narrow-table note wrapping. | Automated numbering of figures reflects their new LaTeX order. |

**Submission caveat.** The Cogent-specific public code-and-environment archive remains to be deposited and independently replayed. The currently linked embedded GitHub ZIP is for a distinct NeurIPS analysis; the manuscript does not claim otherwise. No manuscript wording can replace unavailable loan status histories, measured lender costs, protected-class data, or prospective intervention records.
