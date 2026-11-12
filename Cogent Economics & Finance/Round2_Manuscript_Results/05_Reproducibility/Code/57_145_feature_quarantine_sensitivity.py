"""Round 2, Reviewer 2 Comment 5: conservative predictor-set sensitivity.

This is a targeted diagnostic, NOT proof that retained predictors are historically
available or that the selected-outcome test cohort is unbiased.

Use the exact seed-42 180k/full cohorts, original model hyperparameters, and
original mature-population definition. Keep original prediction arrays read-only.

Run from any directory:
  python "E:\\Governance-Policy\\Cogent Economics & Finance\\Round2_Manuscript_Results\\05_Reproducibility\\Code\\31_145_predictor_sensitivity.py"
"""
from pathlib import Path
import gc
import sys
import platform
import time
import warnings

import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss

warnings.filterwarnings("ignore", category=UserWarning)
ROOT = Path(r"E:\Governance-Policy\Cogent Economics & Finance")
DATA = ROOT / "P2P_Macro_Data.dta"
OLD = ROOT / "revision_outputs"
OUT = ROOT / "Round2_Manuscript_Results"
TABLES = OUT / "01_Tables"
LOGDIR = OUT / "05_Reproducibility" / "Execution_Logs"
TABLES.mkdir(parents=True, exist_ok=True)
LOGDIR.mkdir(parents=True, exist_ok=True)
CSV = TABLES / "57_145_feature_quarantine_sensitivity.csv"
FEATURE_CSV = TABLES / "57_145_feature_quarantine_manifest.csv"
LOG = LOGDIR / "57_145_feature_quarantine_report.txt"

class Recorder:
    def __init__(self): self.lines = []
    def say(self, *items):
        s = " ".join(map(str, items))
        print(s, flush=True)
        self.lines.append(s)
    def save(self): LOG.write_text("\n".join(self.lines) + "\n", encoding="utf-8")
r = Recorder()
r.say("ROUND 2 | REVIEWER 2 COMMENT 5 | CONSERVATIVE PREDICTOR SENSITIVITY")
r.say("Python", sys.version.split()[0], "NumPy", np.__version__, "pandas", pd.__version__, "sklearn", sklearn.__version__)
r.say("Original Stata dataset and revision_outputs are read only.")

# STEP 57: Remove the five documented payment-date-dependent variables.
# Other features have NOT been certified as application-time available.

CANDIDATE_FEATURES = pd.read_csv(
    ROOT
    / "Round2_Manuscript_Results"
    / "05_Reproducibility"
    / "Evidence_Checks"
    / "54_145_feature_direct_leakage_quarantine.csv"
)["feature"].astype(str).tolist()

if len(CANDIDATE_FEATURES) != 145:
    raise RuntimeError("Expected exactly 145 candidate features.")

if len(set(CANDIDATE_FEATURES)) != 145:
    raise RuntimeError("Duplicate candidate features detected.")

SPEC = OLD / "cell26_locked_feature_specification.csv"
for path in [DATA, SPEC, OLD / "cell27_y_val.npy", OLD / "cell27_y_test.npy"]:
    if not path.exists(): raise FileNotFoundError(path)
spec = pd.read_csv(SPEC)
assert {"feature", "type"}.issubset(spec.columns)
original = [str(x) for x in spec.feature if str(x) not in {"emp_title", "desc", "title"}]
if len(original) != 150 or len(original) != len(set(original)):
    raise RuntimeError("Cannot reconstruct the original 150-feature specification.")
features = [x for x in original if x in CANDIDATE_FEATURES]
missing = set(CANDIDATE_FEATURES) - set(features)
if missing:
    raise RuntimeError(f"Candidate features not in original specification: {sorted(missing)}")
excluded = [x for x in original if x not in features]
if len(features) + len(excluded) != 150 or not features:
    raise RuntimeError("Feature inclusion accounting failed.")
type_map = dict(zip(spec.feature.astype(str), spec.type.astype(str)))
cat = [x for x in features if type_map[x].lower() == "categorical"]
num = [x for x in features if x not in cat]
r.say("Original features:", len(original), "| Conservative candidate features:", len(features), "| Excluded:", len(excluded))
r.say("Numerical:", len(num), "| Categorical:", len(cat))
r.say("Important: candidate is not certified leakage-free; decision-stage availability must still be documented.")
manifest = pd.DataFrame([{
    "feature": name,
    "originally_used": True,
    "included_145_candidate": name in features,
    "original_type": type_map[name],
    "note": ("Borrower/credit-history candidate, timestamp not independently verified"
             if name in features else "Excluded from conservative application/credit-history candidate")
} for name in original])
manifest.to_csv(FEATURE_CSV, index=False)
r.say("Excluded names:", ", ".join(excluded))
r.save()

# EXACT SAME original sample reconstruction as Step 22, but read only selected
# raw features to avoid unnecessary use of RAM.
columns = list(dict.fromkeys(["issue_d", "term", "loan_status", "badloan"] + features))
parts = {"train": [], "val": [], "test": []}
scanned = 0
cutoff = pd.Timestamp("2020-04-01")
r.say("Reconstructing source cohorts using original file order and maturity rule...")
with pd.read_stata(DATA, columns=columns, convert_categoricals=False, chunksize=200_000) as reader:
    for block in reader:
        scanned += len(block)
        issue = pd.to_datetime(block["issue_d"].astype(str).str.strip(), format="%b-%y", errors="coerce")
        months = pd.to_numeric(block["term"].astype(str).str.extract(r"(\d+)")[0], errors="coerce")
        statuses = block["loan_status"].astype(str).str.strip()
        terminal = statuses.isin(["Fully Paid", "Charged Off", "Default"])
        elapsed = (cutoff.year - issue.dt.year) * 12 + (cutoff.month - issue.dt.month)
        mature = terminal | (elapsed >= months)
        selectors = {
            "train": issue.between(pd.Timestamp("2012-08-01"), pd.Timestamp("2018-06-01")) & mature,
            "val": issue.between(pd.Timestamp("2018-07-01"), pd.Timestamp("2019-03-01")) & mature,
            "test": issue.between(pd.Timestamp("2019-04-01"), pd.Timestamp("2019-12-01")) & mature
        }
        for split, keep in selectors.items():
            if keep.any(): parts[split].append(block.loc[keep, features + ["badloan"]].copy())
        r.say(f" Scanned {scanned:,} source rows")
if scanned != 2_703_430: raise RuntimeError(f"Source count changed: {scanned}")
frames = {split: pd.concat(groups, ignore_index=True) for split, groups in parts.items()}
del parts; gc.collect()
for split, n_expected in [("train", 1_304_058), ("val", 27_374), ("test", 41_145)]:
    if len(frames[split]) != n_expected:
        raise RuntimeError(f"Unexpected {split} sample n={len(frames[split])}")
    frames[split]["y"] = pd.to_numeric(frames[split].pop("badloan"), errors="coerce").fillna(0).astype(int)
    r.say(split, "n=", len(frames[split]), "adverse share=", f"{frames[split]['y'].mean():.6f}")
for split, labelfile in [("val", "cell27_y_val.npy"), ("test", "cell27_y_test.npy")]:
    if not np.array_equal(frames[split].y.to_numpy(), np.load(OLD / labelfile)):
        raise RuntimeError(f"{split} outcome order not the same as saved original. STOP.")
r.say("PASS: Labels and sample order match original 150-feature prediction arrays.")
r.save()

# Fit preprocessor once on full mature training cohort as in original notebook,
# specifically so the only deliberate change is the restricted feature set.
preprocessor = ColumnTransformer(transformers=[
    ("num", Pipeline([("imputer", SimpleImputer(strategy="median")),
                      ("scale", StandardScaler())]), num),
    ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                      ("onehot", OneHotEncoder(handle_unknown="ignore", max_categories=50,
                                                sparse_output=False))]), cat)
], remainder="drop")
r.say("Fitting preprocessor on full mature training cohort; preserves original preprocessor sampling convention...")
preprocessor.fit(frames["train"][features])
X_val = preprocessor.transform(frames["val"][features]).astype(np.float32)
X_test = preprocessor.transform(frames["test"][features]).astype(np.float32)
y_val = frames["val"].y.to_numpy()
y_test = frames["test"].y.to_numpy()
X_full = preprocessor.transform(frames["train"][features]).astype(np.float32)
y_full = frames["train"].y.to_numpy()
r.say("Conservative transformed input dimensions:", X_full.shape[1]); r.save()
del frames, preprocessor; gc.collect()

sizes = [180_000, len(y_full)]
perm = np.random.default_rng(42).permutation(len(y_full))
def models(seed):
    return {
        "Logistic_SGD": SGDClassifier(loss="log_loss", penalty="l2", alpha=1e-5,
                                      max_iter=100, tol=1e-4, random_state=seed, n_jobs=-1),
        "HistGB": HistGradientBoostingClassifier(max_depth=6, learning_rate=0.05,
                                                 max_iter=300, random_state=seed),
        "MLP": MLPClassifier(hidden_layer_sizes=(128,64), activation="relu", solver="adam",
                             max_iter=50, early_stopping=True, validation_fraction=0.10,
                             n_iter_no_change=5, random_state=seed)
    }
def metrics(y, pred):
    return {"auc": float(roc_auc_score(y,pred)),
            "brier": float(brier_score_loss(y,pred)),
            "log_loss": float(log_loss(y,pred, labels=[0,1])),
            "mean_prediction": float(np.mean(pred))}
rows = []
for n in sizes:
    idx = perm[:n]
    X_train, y_train = X_full[idx], y_full[idx]
    r.say("\n145-FEATURE LEAKAGE QUARANTINE | Train n=", f"{n:,}", "| seed=42")
    for model_name, model in models(42).items():
        start = time.perf_counter()
        model.fit(X_train, y_train)
        val_pred = model.predict_proba(X_val)[:,1]
        test_pred = model.predict_proba(X_test)[:,1]
        val_metric = metrics(y_val, val_pred)
        test_metric = metrics(y_test, test_pred)
        benchmark_file = OLD / f"cell27_{n}_{model_name}_test_pred.npy"
        if not benchmark_file.exists(): raise FileNotFoundError(benchmark_file)
        orig_metric = metrics(y_test, np.load(benchmark_file))
        rec = {"training_n": n, "seed":42,"model":model_name,
               "candidate_features":len(features), "excluded_features":len(excluded),
               "val_auc_145":val_metric["auc"],
               "test_auc_145":test_metric["auc"],
               "test_brier_145":test_metric["brier"],
               "test_logloss_145":test_metric["log_loss"],
               "test_mean_prediction_145":test_metric["mean_prediction"],
               "original_test_auc":orig_metric["auc"],
               "original_test_brier":orig_metric["brier"],
               "original_test_logloss":orig_metric["log_loss"],
               "original_test_mean_prediction":orig_metric["mean_prediction"],
               "test_auc_change_145_minus_original":test_metric["auc"]-orig_metric["auc"],
               "test_brier_change_145_minus_original":test_metric["brier"]-orig_metric["brier"],
               "fit_eval_seconds":round(time.perf_counter()-start,1)}
        rows.append(rec)
        pd.DataFrame(rows).to_csv(CSV, index=False)
        r.say(model_name, "test AUC", f"{test_metric['auc']:.6f}",
              "vs original", f"{orig_metric['auc']:.6f}",
              "| test Brier", f"{test_metric['brier']:.6f}",
              "vs original", f"{orig_metric['brier']:.6f}")
        r.save()
        del model, val_pred, test_pred
        gc.collect()
    del X_train, y_train
    gc.collect()

r.say("\nIMPORTANT LIMITATIONS:")
r.say("- Retained features are candidates, NOT independently timestamp-verified application features.")
r.say("- Baseline 150-feature predictions are reused; only conservative models are refitted.")
r.say("- Seed 42 only; fixed hyperparameters; original full-cohort fitted preprocessing convention.")
r.say("- The outcomes and selected mature evaluation cohort are unchanged and may be selection biased.")
r.say("- Removing features does not identify which variable caused a performance difference.")
r.say("Saved:", CSV, FEATURE_CSV, LOG)
r.save()
print("STEP 57 COMPLETE",flush=True)

