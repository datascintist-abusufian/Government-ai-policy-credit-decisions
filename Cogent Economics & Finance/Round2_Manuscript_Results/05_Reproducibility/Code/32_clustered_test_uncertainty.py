"""Round 2, Reviewer 2 Comments 7 and 8: state/month block-bootstrap uncertainty.

Reuses the ORIGINAL saved test labels and full-training seed-42 predictions.
Reconstructs only TEST identifiers (origination month, state, unemployment)
from the public .dta, in original dataset row order. NO model fitting.

Produces SEPARATE state-cluster and origination-month-block bootstrap intervals.
With only nine test origination months, month-bootstrap intervals are highly
exploratory. Separate one-way bootstraps are NOT a two-way-cluster correction.
They also do not account for fitting, sample selection, hyperparameter search,
or the mature-test outcome-selection bias.
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, brier_score_loss

ROOT = Path(r"E:\Governance-Policy\Cogent Economics & Finance")
DATA = ROOT / "P2P_Macro_Data.dta"
OLD = ROOT / "revision_outputs"
BASE = ROOT / "Round2_Manuscript_Results"
TABLES = BASE / "01_Tables"
LOGDIR = BASE / "05_Reproducibility" / "Execution_Logs"
TABLES.mkdir(parents=True, exist_ok=True)
LOGDIR.mkdir(parents=True, exist_ok=True)
OUT = TABLES / "32_clustered_test_uncertainty.csv"
LOG = LOGDIR / "32_clustered_test_uncertainty_report.txt"

SEED = 20261105
B = 1500
EXPECTED_TEST_N = 41145
EXPECTED_ALL_N = 2703430
CUTOFF = pd.Timestamp("2020-04-01")

class Logger:
    def __init__(self): self.lines = []
    def say(self, *items):
        msg = " ".join(map(str, items))
        print(msg, flush=True)
        self.lines.append(msg)
    def save(self): LOG.write_text("\n".join(self.lines) + "\n", encoding="utf-8")
r = Logger()
r.say("ROUND 2 | REVIEWER 2 COMMENTS 7/8 | TEST CLUSTERED UNCERTAINTY")
r.say("Python:", sys.version.split()[0], "| NumPy:", np.__version__, "| pandas:", pd.__version__)
r.say("Only state/month metadata read from source; original labels/predictions are read-only.")
for path in [DATA, OLD / "cell27_y_test.npy", OLD / "cell29_histgb_full_test_recalibrated.npy", OLD / "cell30_recalibrated_threshold_analysis.csv"]:
    if not path.is_file(): raise FileNotFoundError(path)

r.say("Reconstructing test cohort metadata in original source row order...")
parts = []
count = 0
with pd.read_stata(DATA, columns=["issue_d", "term", "loan_status", "badloan", "addr_state", "unem_rate"],
                   convert_categoricals=False, chunksize=200000) as reader:
    for block in reader:
        count += len(block)
        issue = pd.to_datetime(block["issue_d"].astype(str).str.strip(), format="%b-%y", errors="coerce")
        term = pd.to_numeric(block["term"].astype(str).str.extract(r"(\d+)")[0], errors="coerce")
        status = block["loan_status"].astype(str).str.strip()
        age = (CUTOFF.year - issue.dt.year) * 12 + (CUTOFF.month - issue.dt.month)
        mature = status.isin(["Fully Paid", "Charged Off", "Default"]) | (age >= term)
        keep = issue.between(pd.Timestamp("2019-04-01"), pd.Timestamp("2019-12-01")) & mature
        if bool(keep.any()):
            parts.append(pd.DataFrame({
                "issue_month": issue.loc[keep].dt.strftime("%Y-%m").to_numpy(),
                "state": block.loc[keep, "addr_state"].astype(str).str.strip().to_numpy(),
                "unem_rate": pd.to_numeric(block.loc[keep, "unem_rate"], errors="coerce").to_numpy(),
                "y": pd.to_numeric(block.loc[keep, "badloan"], errors="coerce").fillna(0).astype(np.int8).to_numpy()
            }))
        r.say(f"  Scanned {count:,} source records")
if count != EXPECTED_ALL_N: raise RuntimeError(f"Unexpected source rows: {count}")
if not parts: raise RuntimeError("No test records were selected")
meta = pd.concat(parts, ignore_index=True)
if len(meta) != EXPECTED_TEST_N: raise RuntimeError(f"Expected {EXPECTED_TEST_N} test loans, found {len(meta)}")
y = meta["y"].to_numpy(dtype=np.int8)
y_old = np.load(OLD / "cell27_y_test.npy", allow_pickle=False).reshape(-1)
if not np.array_equal(y, y_old): raise RuntimeError("STOP: test labels/order differ from original saved arrays")
r.say("PASS: Test rows/labels match original predictions, n=", len(y), ", adverse=", int(y.sum()))

pred = {}
for model in ["Logistic_SGD", "HistGB", "MLP"]:
    v = np.load(OLD / f"cell27_1304058_{model}_test_pred.npy", allow_pickle=False).reshape(-1)
    if len(v) != len(y) or not np.isfinite(v).all() or (v < 0).any() or (v > 1).any():
        raise RuntimeError(f"Invalid prediction array: {model}")
    pred[model] = v.astype(np.float64)
cal = np.load(OLD / "cell29_histgb_full_test_recalibrated.npy", allow_pickle=False).reshape(-1).astype(np.float64)
if len(cal) != len(y) or not np.isfinite(cal).all() or (cal < 0).any() or (cal > 1).any():
    raise RuntimeError("Invalid recalibrated predictions")
if abs(roc_auc_score(y, pred["HistGB"]) - .675016) > .00015: raise RuntimeError("HistGB test AUC mismatch")
if abs(brier_score_loss(y, cal) - .123161) > .00015: raise RuntimeError("Recalibrated Brier mismatch")

thresholds_df = pd.read_csv(OLD / "cell30_recalibrated_threshold_analysis.csv")
thresholds_df = thresholds_df.loc[thresholds_df.cohort.astype(str).str.lower().eq("test")]
thresholds = {}
expected_counts = {5: (943, 2244, 5082), 10: (3533, 11909, 2492), 20: (5730, 29266, 295)}
for c in [5, 10, 20]:
    rows = thresholds_df.loc[thresholds_df.fn_cost.eq(c)]
    if len(rows) != 1: raise RuntimeError(f"Expected one test threshold row at FN:FP {c}:1")
    t = float(rows.iloc[0].threshold)
    flag = cal >= t
    tp = int(np.count_nonzero(flag & (y == 1)))
    fp = int(np.count_nonzero(flag & (y == 0)))
    fn = int(np.count_nonzero(~flag & (y == 1)))
    if (tp, fp, fn) != expected_counts[c]:
        raise RuntimeError(f"Threshold counts differ from saved analysis at {c}:1: {(tp,fp,fn)}")
    thresholds[c] = t
r.say("PASS: Original test threshold counts replicated for 5:1, 10:1, 20:1")

meta["state"] = meta.state.replace({"": "__MISSING__", "nan": "__MISSING__", "None": "__MISSING__"})
meta["high_unemployment"] = meta.unem_rate.ge(4.9)
if int(meta.high_unemployment.sum()) != 641 or int(y[meta.high_unemployment.to_numpy()].sum()) != 74:
    raise RuntimeError("High-unemployment group does not match published 641/74; stop and check metadata")
r.say("State clusters:", meta.state.nunique(), "| Origination-month blocks:", meta.issue_month.nunique())
r.say("High-state-unemployment test subgroup: 641 observations; 74 adverse outcomes")
if meta.issue_month.nunique() != 9: raise RuntimeError("Unexpected test origination month count; must be 9")

high = meta.high_unemployment.to_numpy(dtype=bool)
flags = {c: cal >= t for c, t in thresholds.items()}
# Exact original 10 equal-width probability bins, as in original Cell 29.
bins = np.digitize(cal, np.linspace(0, 1, 11)[1:-1], right=False)

def metrics(idx):
    yy = y[idx]
    if yy.min() == yy.max(): return None
    pp = cal[idx]
    result = {}
    for name in ["Logistic_SGD", "HistGB", "MLP"]:
        result["auc_" + name] = float(roc_auc_score(yy, pred[name][idx]))
    result["auc_logistic_minus_histgb"] = result["auc_Logistic_SGD"] - result["auc_HistGB"]
    result["auc_histgb_minus_mlp"] = result["auc_HistGB"] - result["auc_MLP"]
    result["auc_logistic_minus_mlp"] = result["auc_Logistic_SGD"] - result["auc_MLP"]
    result["calibrated_brier"] = float(np.mean((yy - pp) ** 2))
    result["calibration_gap_observed_minus_predicted"] = float(np.mean(yy - pp))
    result["calibrated_mean_prediction"] = float(pp.mean())
    result["observed_rate"] = float(yy.mean())
    # ECE: 10 equal-width bins, weighted absolute calibration gaps.
    bb = bins[idx]
    ece = 0.0
    for b in range(10):
        sel = (bb == b)
        if sel.any(): ece += float(sel.mean() * abs(yy[sel].mean() - pp[sel].mean()))
    result["calibrated_ECE_10_equal_width"] = ece
    for c in (5, 10, 20):
        ff = flags[c][idx]
        tp = np.count_nonzero(ff & (yy == 1))
        fp = np.count_nonzero(ff & (yy == 0))
        fn = np.count_nonzero(~ff & (yy == 1))
        result[f"fnr_{c}to1"] = fn / (fn + tp)
        result[f"review_rate_{c}to1"] = float(ff.mean())
        result[f"cost_per_loan_{c}to1"] = float((c * fn + fp) / len(idx))
        result[f"model_cost_minus_flag_all_per_loan_{c}to1"] = float((c * fn + fp - (len(idx) - int(yy.sum()))) / len(idx))
    hh = high[idx]
    h_events = np.count_nonzero(hh & (yy == 1))
    other_events = np.count_nonzero(~hh & (yy == 1))
    ff10 = flags[10][idx]
    if h_events > 0 and other_events > 0:
        f_h = np.count_nonzero(hh & (yy == 1) & ~ff10) / h_events
        f_o = np.count_nonzero(~hh & (yy == 1) & ~ff10) / other_events
        result["fnr_gap_high_minus_other_10to1"] = float(f_h - f_o)
    else:
        result["fnr_gap_high_minus_other_10to1"] = np.nan
    return result

observed = metrics(np.arange(len(y), dtype=np.int32))
if abs(observed["calibrated_ECE_10_equal_width"] - .058325) > .0002:
    raise RuntimeError("ECE does not match original Cell 29 definition. Stop rather than report a mismatched result")
r.say("Observed recalibrated Brier:", f"{observed['calibrated_brier']:.6f}",
      "| ECE:", f"{observed['calibrated_ECE_10_equal_width']:.6f}",
      "| observed-minus-predicted gap:", f"{observed['calibration_gap_observed_minus_predicted']:.6f}")

rng = np.random.default_rng(SEED)
rows = []
for kind, field in [("state_cluster", "state"), ("origination_month_block", "issue_month")]:
    grouping = meta.groupby(field, sort=True).indices
    keys = list(grouping)
    groups = [np.asarray(grouping[key], dtype=np.int32) for key in keys]
    k = len(groups)
    if k < 2: raise RuntimeError(f"Cannot cluster-bootstrap {kind}: only {k} groups")
    r.say(f"\n{kind}: {k} clusters, {B} bootstrap resamples; no retraining")
    results = []
    for rep in range(B):
        chosen = rng.integers(0, k, size=k)
        ix = np.concatenate([groups[int(i)] for i in chosen])
        result = metrics(ix)
        if result is not None: results.append(result)
        if (rep + 1) % 500 == 0: r.say(f"  Completed {rep+1}/{B}")
    draws = pd.DataFrame(results)
    if len(draws) < .95 * B: raise RuntimeError(f"Too many invalid bootstrap draws for {kind}")
    for metric, value in observed.items():
        valid = draws[metric].dropna()
        if valid.empty:
            low = high_ci = np.nan
        else:
            low, high_ci = np.quantile(valid, [.025, .975])
        rows.append({
            "bootstrap_unit": kind,
            "clusters": k,
            "replicates_requested": B,
            "replicates_valid": len(valid),
            "metric": metric,
            "observed_value": float(value),
            "ci_2p5": float(low),
            "ci_97p5": float(high_ci),
            "seed": SEED,
            "interpretation": ("Exploratory: only 9 origination months, too few for reliable temporal-cluster inference"
                               if kind == "origination_month_block" else
                               "One-way state-cluster bootstrap, conditional on fitted models and maturity-selected test cohort")
        })
    r.say(kind, "complete")
    r.save()

result_table = pd.DataFrame(rows)
result_table.to_csv(OUT, index=False)
r.say("\nSELECTED UNCERTAINTY RESULTS:")
selected = ["auc_logistic_minus_histgb", "auc_histgb_minus_mlp", "calibrated_brier",
            "calibration_gap_observed_minus_predicted", "calibrated_ECE_10_equal_width",
            "fnr_10to1", "review_rate_10to1", "fnr_gap_high_minus_other_10to1"]
print(result_table[result_table.metric.isin(selected)][
    ["bootstrap_unit", "metric", "observed_value", "ci_2p5", "ci_97p5", "replicates_valid"]
].to_string(index=False), flush=True)
r.say("\nINTERPRETATION LIMITS:")
r.say("- Confidence intervals condition on the previously trained models and held-out selected test population.")
r.say("- Separate state and month resampling are NOT joint two-way clustering or pipeline-bootstrap uncertainty.")
r.say("- Only 9 origination months: month-block bootstrap is exploratory, not reliable time-series inference.")
r.say("- No adjustment for feature selection, hyperparameter tuning, validation reuse, or outcome-selected maturity.")
r.say("- Subgroup gap is unadjusted for confounding and multiple testing; do not infer demographic fairness.")
r.say("- No multiplicity-adjusted hypothesis tests are claimed; interpret intervals as exploratory uncertainty ranges.")
r.say("- Cost figures are error-classification proxies, NOT observed human-review costs.")
r.say("Saved:", OUT)
r.say("Saved:", LOG)
r.say("STEP 32 COMPLETE: No new models trained; source and original outputs unchanged.")
r.save()
