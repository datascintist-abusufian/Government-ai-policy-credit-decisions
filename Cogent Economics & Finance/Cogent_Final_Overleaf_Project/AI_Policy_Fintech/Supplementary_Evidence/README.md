# Cogent Economics & Finance: Supplementary evidence and verification status

These files accompany the manuscript as retrospective analyses and reproducibility checks, not a complete reproduction of model training.

* Original experiment: 150 inputs, 1,304,058 mature-training loans, selected validation n=27,374 and test n=41,145.
* Confirmed last-payment-derived calendar values: year/month/quarter. Stata-source dependent loan_vol/median_dummy. The issue_year false positive was resolved (Stata year() function).
* 145-field quarantine sensitivity: six endpoint runs, seed 42, original selected population. No historical point-in-time certification.
* Population selection: only 7.38%/10.26% of validation/test origination cohorts satisfy terminal-status selection; unselected loans do not have equivalent fixed-horizon outcomes.
* Disjoint selection/calibration/threshold windows: post hoc, no refitting of original base classifiers, sparse threshold events, unstable thresholds.
* Hypothetical review intervention: zero observed reviewer interventions and no institution-specific cost estimates.
* Six Holm-adjusted pairwise test AUC p-values: family of six endpoints only. Other inferential families not adjusted.
* Metric replay: 12/12 saved test ROC-AUC calculations reproduced from original arrays; no independent model training replay.
* Existing candidate model-replication archive ZIP integrity checked, but lacked a Stata data file and has not been clean-replayed. This smaller evidence folder does not include model arrays, source Stata data or an original-training dependency lock.

Original publicly deposited source data were cited in the manuscript; publication or replay of the exact Cogent-specific code repository contents is not verified by these materials.

Files under 01_Tables and 05_Reproducibility originate from the uploaded Step62_Final_Reviewer_Evidence.zip dated 6 November 2026. Some scripts refer to the original researcher's Windows directory; paths need adaptation for another machine.
