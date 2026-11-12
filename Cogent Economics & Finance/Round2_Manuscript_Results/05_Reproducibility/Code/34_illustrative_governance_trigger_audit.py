"""Round 2 / Reviewer 2 Comment 9: ILLUSTRATIVE retrospective alert audit.

Purpose: Show WHEN a fully specified example monitoring policy would flag
problems in validation/test summaries. This is NOT a validation of governance
benefit, a test of a deployed intervention, a justified risk tolerance, or a
recommendation to operationalise these example tolerances.

No model fitting, no raw data scan, no changes to existing analysis files.
Validation was already used for model selection/calibration/threshold choice;
validation metrics are descriptive, not independent model-validation evidence.
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(r"E:\Governance-Policy\Cogent Economics & Finance")
OLD = ROOT / "revision_outputs"
BASE = ROOT / "Round2_Manuscript_Results"
TABLES = BASE / "01_Tables"
LOGS = BASE / "05_Reproducibility" / "Execution_Logs"
OUT = TABLES / "34_illustrative_governance_trigger_audit.csv"
LOG = LOGS / "34_illustrative_governance_trigger_audit_report.txt"

# Explicitly ILLUSTRATIVE limits, chosen for a mechanical alert demo only.
# They have not been derived from lender operational costs or regulator rules.
MIN_MATURE_SHARE = 0.50
MAX_ABS_CALIBRATION_IN_THE_LARGE_GAP = 0.02  # 2 percentage points
MAX_FNR = 0.25
MAX_FLAG_RATE = 0.40

EXPECTED = {
    "validation": {"all_n": 370_994, "selected_n": 27_374, "adverse_n": 2_771},
    "test": {"all_n": 401_169, "selected_n": 41_145, "adverse_n": 6_025},
}
EXPECTED_TEST_CONFUSION = {5: (943, 2244, 5082), 10: (3533, 11909, 2492), 20: (5730, 29266, 295)}
lines = []
def say(message=""):
    msg = str(message)
    print(msg, flush=True)
    lines.append(msg)

say("ROUND 2 | REVIEWER 2 COMMENT 9 | ILLUSTRATIVE RETROSPECTIVE GOVERNANCE ALERT AUDIT")
say(f"Python {sys.version.split()[0]}, NumPy {np.__version__}, pandas {pd.__version__}")
say("This is an EXAMPLE of rule firing, NOT proof that the rule or actions work.")
say("Original source and predictions are read-only; no models are trained.")
say("EXAMPLE TOLERANCES (NOT REGULATORY/INSTITUTIONAL LIMITS):")
say(f"  Maturity share >= {MIN_MATURE_SHARE:.0%}")
say(f"  Absolute mean calibration gap <= {MAX_ABS_CALIBRATION_IN_THE_LARGE_GAP:.0%}")
say(f"  False-negative rate <= {MAX_FNR:.0%}")
say(f"  Flag rate <= {MAX_FLAG_RATE:.0%}")

threshold_path = OLD / "cell30_recalibrated_threshold_analysis.csv"
if not threshold_path.is_file():
    raise FileNotFoundError(threshold_path)
threshold_df = pd.read_csv(threshold_path)
required_cols = {"cohort", "fn_cost", "threshold"}
if not required_cols.issubset(threshold_df.columns):
    raise RuntimeError(f"Threshold CSV missing columns: {required_cols - set(threshold_df.columns)}")

TABLES.mkdir(parents=True, exist_ok=True)
LOGS.mkdir(parents=True, exist_ok=True)
rows = []
for cohort, exp in EXPECTED.items():
    y_path = OLD / f"cell27_y_{'val' if cohort == 'validation' else 'test'}.npy"
    p_path = OLD / f"cell29_histgb_full_{'val' if cohort == 'validation' else 'test'}_recalibrated.npy"
    if not y_path.is_file() or not p_path.is_file():
        raise FileNotFoundError(f"Missing original array: {y_path} or {p_path}")
    y = np.load(y_path, allow_pickle=False).reshape(-1)
    p = np.load(p_path, allow_pickle=False).reshape(-1).astype(np.float64)
    if len(y) != exp["selected_n"] or int(np.sum(y)) != exp["adverse_n"]:
        raise RuntimeError(f"Label count check failed for {cohort}")
    if len(y) != len(p) or not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
        raise RuntimeError(f"Invalid probability array for {cohort}")
    if not np.isin(y, [0, 1]).all():
        raise RuntimeError(f"Expected binary outcomes for {cohort}")

    maturity_share = len(y) / exp["all_n"]
    event_rate = float(np.mean(y))
    pred_mean = float(np.mean(p))
    gap = event_rate - pred_mean
    status = threshold_df.loc[threshold_df.cohort.astype(str).str.lower().eq(cohort)]
    # The old analysis may abbreviate validation as 'val'.
    if status.empty and cohort == "validation":
        status = threshold_df.loc[threshold_df.cohort.astype(str).str.lower().eq("val")]
    if status.empty:
        raise RuntimeError(f"No saved threshold records found for {cohort}")

    say(f"\n{cohort.upper()} | selected={len(y):,} / all={exp['all_n']:,} "
        f"| observed={event_rate:.4%} | mean predicted={pred_mean:.4%} "
        f"| observed-minus-predicted={gap:.4%}")
    for ratio in (5, 10, 20):
        sel = status.loc[pd.to_numeric(status.fn_cost, errors="coerce").eq(ratio)]
        if len(sel) != 1:
            raise RuntimeError(f"Expected one {cohort} threshold row for {ratio}:1, got {len(sel)}")
        threshold = float(sel.iloc[0].threshold)
        flag = p >= threshold
        tp = int(np.count_nonzero(flag & (y == 1)))
        fp = int(np.count_nonzero(flag & (y == 0)))
        fn = int(np.count_nonzero(~flag & (y == 1)))
        if cohort == "test" and (tp, fp, fn) != EXPECTED_TEST_CONFUSION[ratio]:
            raise RuntimeError(f"STOP: test threshold benchmark mismatch for {ratio}:1: {(tp, fp, fn)}")
        fnr = fn / exp["adverse_n"]
        flag_rate = float(np.mean(flag))
        model_cost = ratio * fn + fp
        flag_none_cost = ratio * exp["adverse_n"]
        flag_all_cost = len(y) - exp["adverse_n"]
        best_trivial_cost = min(flag_none_cost, flag_all_cost)
        breaches = {
            "maturity_breach": maturity_share < MIN_MATURE_SHARE,
            "calibration_breach": abs(gap) > MAX_ABS_CALIBRATION_IN_THE_LARGE_GAP,
            "fnr_breach": fnr > MAX_FNR,
            "flag_capacity_breach": flag_rate > MAX_FLAG_RATE,
            "classification_cost_exceeds_best_trivial_proxy": model_cost > best_trivial_cost,
        }
        rows.append({
            "cohort": cohort, "fn_to_fp_ratio": f"{ratio}:1", "threshold": threshold,
            "full_cohort_n": exp["all_n"], "evaluated_n": len(y), "adverse_n": exp["adverse_n"],
            "maturity_share": maturity_share, "observed_rate": event_rate,
            "mean_prediction": pred_mean, "observed_minus_predicted": gap,
            "false_negative_rate": fnr, "flag_rate": flag_rate,
            "model_classification_cost": model_cost,
            "flag_none_proxy_cost": flag_none_cost, "flag_all_proxy_cost": flag_all_cost,
            "best_trivial_proxy_cost": best_trivial_cost,
            **breaches,
            "any_breach": any(breaches.values()),
            "example_limits_only": True,
        })
        active = ", ".join(k for k, v in breaches.items() if v) or "none"
        say(f"  {ratio}:1 | FNR={fnr:.2%} | flagged={flag_rate:.2%} "
            f"| cost={model_cost:,} | best trivial proxy={best_trivial_cost:,} "
            f"| example alerts: {active}")

result = pd.DataFrame(rows)
result.to_csv(OUT, index=False)
say("\nLIMITATIONS AND INTERPRETATION:")
for s in [
    "Example limits are deliberately illustrative; no lender approved them.",
    "This is a two-window, retrospective calculation, not a deployed governance experiment.",
    "The validation cohort was reused for selection/recalibration/threshold fitting.",
    "Maturity selection is outcome-dependent; alert rates do not generalise to all originations.",
    "The classification-cost proxy does not include effective human review or staffing capacity.",
    "The audit does not measure whether escalation, recalibration, or suspension improves outcomes.",
    "Fitted-predictor timing and protected-group fairness remain unverified.",
    "No causal, regulatory-compliance, or operational-performance claim is supported.",
]: say("- " + s)
say(f"Saved: {OUT}")
say(f"Saved: {LOG}")
say("STEP 34 COMPLETE: example alerts only, no original files changed.")
LOG.write_text("\n".join(lines) + "\n", encoding="utf-8")
