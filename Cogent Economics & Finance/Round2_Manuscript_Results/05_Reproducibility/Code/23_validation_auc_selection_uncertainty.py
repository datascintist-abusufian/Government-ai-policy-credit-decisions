"""Round 2 | Reviewer 2, Comment 4: validation AUC selection uncertainty.

Uses ONLY pre-existing validation predictions and labels. No model retraining,
no access to the source data, and no modification of earlier analysis outputs.

Reports validation AUC differences using paired, outcome-stratified bootstrap.
Does not remove the optimism caused by selecting, recalibrating, and choosing
thresholds on the same validation period.
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(r"E:\Governance-Policy\Cogent Economics & Finance")
SRC = ROOT / "revision_outputs"
BASE = ROOT / "Round2_Manuscript_Results"
TABLES = BASE / "01_Tables"
LOGS = BASE / "05_Reproducibility" / "Execution_Logs"
TABLES.mkdir(parents=True, exist_ok=True)
LOGS.mkdir(parents=True, exist_ok=True)

SEED = 20261105
BOOTSTRAPS = 2000
EXPECTED_N = 27374
EXPECTED_AUCS = {"HistGB": 0.6966, "MLP": 0.6932, "Logistic_SGD": 0.6926}

print("ROUND 2 | REVIEWER 2 COMMENT 4 | VALIDATION MODEL-SELECTION UNCERTAINTY")
print("Loading original full-training, seed-42 validation predictions only...")

y = np.load(SRC / "cell27_y_val.npy", allow_pickle=False).reshape(-1)
preds = {
    name: np.load(SRC / f"cell27_1304058_{name}_val_pred.npy", allow_pickle=False).reshape(-1)
    for name in ("HistGB", "MLP", "Logistic_SGD")
}

if y.size != EXPECTED_N or not np.all(np.isin(y, [0, 1])):
    raise RuntimeError(f"Unexpected validation target: length={y.size}; expected {EXPECTED_N} binary labels")
for name, probabilities in preds.items():
    if probabilities.size != EXPECTED_N or not np.isfinite(probabilities).all():
        raise RuntimeError(f"Prediction array mismatch: {name}")
    if not np.all((probabilities >= 0) & (probabilities <= 1)):
        raise RuntimeError(f"Predictions are outside [0, 1]: {name}")

auc = {name: roc_auc_score(y, p) for name, p in preds.items()}
for name, value in auc.items():
    print(f"{name:13} validation ROC-AUC={value:.6f}")
    if abs(value - EXPECTED_AUCS[name]) > 0.00015:
        raise RuntimeError(f"Validation AUC differs from the manuscript for {name}; check prediction alignment")

positive_idx = np.flatnonzero(y == 1)
negative_idx = np.flatnonzero(y == 0)
rng = np.random.default_rng(SEED)
comparisons = [("HistGB", "MLP"), ("HistGB", "Logistic_SGD")]
bootstrap_differences = {pair: np.empty(BOOTSTRAPS, dtype=np.float64) for pair in comparisons}

print(f"Validation n={y.size:,}; adverse={positive_idx.size:,}; nonadverse={negative_idx.size:,}")
print(f"Paired outcome-stratified bootstraps: {BOOTSTRAPS:,}; RNG seed {SEED}")

for rep in range(BOOTSTRAPS):
    indices = np.concatenate([
        rng.choice(positive_idx, positive_idx.size, replace=True),
        rng.choice(negative_idx, negative_idx.size, replace=True),
    ])
    yy = y[indices]
    rep_aucs = {name: roc_auc_score(yy, p[indices]) for name, p in preds.items()}
    for pair in comparisons:
        bootstrap_differences[pair][rep] = rep_aucs[pair[0]] - rep_aucs[pair[1]]
    if (rep + 1) % 500 == 0:
        print(f"  Completed {rep + 1:,}/{BOOTSTRAPS:,} bootstrap replicates", flush=True)

rows = []
for pair in comparisons:
    a, b = pair
    delta = auc[a] - auc[b]
    lower, upper = np.percentile(bootstrap_differences[pair], [2.5, 97.5])
    includes_zero = bool(lower <= 0 <= upper)
    rows.append({
        "cohort": "validation",
        "full_training_n": 1304058,
        "model_seed": 42,
        "model_a": a,
        "model_b": b,
        "n_validation": int(y.size),
        "n_adverse": int(positive_idx.size),
        "auc_model_a": round(auc[a], 8),
        "auc_model_b": round(auc[b], 8),
        "auc_difference_a_minus_b": round(delta, 8),
        "bootstrap_ci_2p5": round(float(lower), 8),
        "bootstrap_ci_97p5": round(float(upper), 8),
        "ci_includes_zero": includes_zero,
        "bootstrap_replicates": BOOTSTRAPS,
        "bootstrap_seed": SEED,
    })

out = TABLES / "23_validation_auc_selection_uncertainty.csv"
pd.DataFrame(rows).to_csv(out, index=False)

report = [
    "ROUND 2 | REVIEWER 2 COMMENT 4 | VALIDATION AUC SELECTION UNCERTAINTY",
    f"Python: {sys.version.split()[0]}; NumPy: {np.__version__}; pandas: {pd.__version__}",
    "Only existing original model predictions were read. No classifier was trained.",
    f"Validation N: {y.size}; adverse: {positive_idx.size}; nonadverse: {negative_idx.size}",
    f"Paired stratified bootstrap B={BOOTSTRAPS}, sampling seed={SEED}",
    "",
    "VALIDATION AUCS:",
    *[f"{name}: {value:.8f}" for name, value in auc.items()],
    "",
    "PAIRED VALIDATION CONTRASTS:",
    pd.DataFrame(rows).to_string(index=False),
    "",
    "LIMITATIONS:",
    "1) These intervals use the SAME validation loans that were used to choose HistGB.",
    "2) They assess pairwise AUC uncertainty, not post-selection performance optimism.",
    "3) They do not account for loan-level dependence within months, states or borrowers.",
    "4) They do not resolve using validation data for selection, recalibration and thresholds.",
    "5) A CI containing zero does not prove two models are equivalent.",
    "6) These are unadjusted percentile intervals; no multiplicity-adjusted claim is made.",
    "7) No test labels or test AUCs were used in this comparison.",
    "",
    f"Saved: {out}",
]
logpath = LOGS / "23_validation_auc_selection_report.txt"
logpath.write_text("\n".join(report) + "\n", encoding="utf-8")
print("\n" + pd.DataFrame(rows)[[
    "model_a", "model_b", "auc_difference_a_minus_b",
    "bootstrap_ci_2p5", "bootstrap_ci_97p5", "ci_includes_zero"
]].to_string(index=False))
print("\nSaved:", out)
print("Saved:", logpath)
print("STEP 23 COMPLETE. Existing files unchanged.")
