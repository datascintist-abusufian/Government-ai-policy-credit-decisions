"""Round 2 / Reviewer 2 Comment 8: subgroup support and confounding audit.

READ-ONLY inspection of the existing Stata source. No training, resampling,
model evaluation, or changes to prior outputs. Summarizes how many states and
origination months actually contribute to the 4.9%-plus unemployment subgroup,
and whether the groups differ in grade/term composition.

The 4.9% cutoff is a descriptive published threshold, NOT a justified causal
exposure or a pre-registered stress definition.
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(r"E:\Governance-Policy\Cogent Economics & Finance")
DATA = ROOT / "P2P_Macro_Data.dta"
OLD = ROOT / "revision_outputs"
BASE = ROOT / "Round2_Manuscript_Results"
TABLES = BASE / "01_Tables"
LOGS = BASE / "05_Reproducibility" / "Execution_Logs"
STATE_OUT = TABLES / "33_high_unemployment_support_by_state.csv"
MONTH_OUT = TABLES / "33_high_unemployment_support_by_month.csv"
GRD_OUT = TABLES / "33_high_unemployment_support_by_grade_term.csv"
REPORT = LOGS / "33_high_unemployment_support_report.txt"

EXPECTED_TOTAL = 2_703_430
EXPECTED_TEST = 41_145
EXPECTED_ADVERSE = 6_025
EXPECTED_HIGH_N = 641
EXPECTED_HIGH_ADVERSE = 74
CUTOFF = pd.Timestamp("2020-04-01")

print("ROUND 2 | REVIEWER 2 COMMENT 8 | SUBGROUP CLUSTER SUPPORT", flush=True)
print("Read-only source audit. No models trained and no original outputs modified.", flush=True)
if not DATA.is_file():
    raise FileNotFoundError(DATA)
if not (OLD / "cell27_y_test.npy").is_file():
    raise FileNotFoundError(OLD / "cell27_y_test.npy")

all_rows = 0
parts = []
with pd.read_stata(DATA, columns=["issue_d", "term", "loan_status", "badloan",
                                  "addr_state", "unem_rate", "grade"],
                   convert_categoricals=False, chunksize=200_000) as blocks:
    for df in blocks:
        all_rows += len(df)
        issue = pd.to_datetime(df.issue_d.astype(str).str.strip(), format="%b-%y", errors="coerce")
        term = pd.to_numeric(df.term.astype(str).str.extract(r"(\d+)")[0], errors="coerce")
        status = df.loan_status.astype(str).str.strip()
        elapsed = (CUTOFF.year - issue.dt.year) * 12 + (CUTOFF.month - issue.dt.month)
        is_mature = status.isin(["Fully Paid", "Charged Off", "Default"]) | elapsed.ge(term)
        keep = issue.between(pd.Timestamp("2019-04-01"), pd.Timestamp("2019-12-01")) & is_mature
        if bool(keep.any()):
            sub = df.loc[keep]
            rate = pd.to_numeric(sub.unem_rate, errors="coerce")
            parts.append(pd.DataFrame({
                "state": sub.addr_state.astype(str).str.strip().replace({"":"__MISSING__", "nan":"__MISSING__"}).to_numpy(),
                "month": issue.loc[keep].dt.strftime("%Y-%m").to_numpy(),
                "term": term.loc[keep].astype("Int64").astype(str).to_numpy(),
                "grade": sub.grade.astype(str).str.strip().to_numpy(),
                "rate": rate.to_numpy(),
                "adverse": pd.to_numeric(sub.badloan, errors="coerce").fillna(0).astype(np.int8).to_numpy(),
            }))
        print(f"  Scanned {all_rows:,} source records", flush=True)

if all_rows != EXPECTED_TOTAL or not parts:
    raise RuntimeError("Source row count or selected test observations unexpected")
meta = pd.concat(parts, ignore_index=True)
if len(meta) != EXPECTED_TEST or int(meta.adverse.sum()) != EXPECTED_ADVERSE:
    raise RuntimeError("Test cohort size or adverse count did not reconcile")
y_saved = np.load(OLD / "cell27_y_test.npy", allow_pickle=False).reshape(-1)
if not np.array_equal(meta.adverse.to_numpy(), y_saved):
    raise RuntimeError("Test labels or row order differ from original predictions")
meta["high_unemployment"] = meta.rate.ge(0.049 - 1e-8)
if (int(meta.high_unemployment.sum()), int(meta.loc[meta.high_unemployment, "adverse"].sum())) != (EXPECTED_HIGH_N, EXPECTED_HIGH_ADVERSE):
    raise RuntimeError("Subgroup count mismatch; no report saved")
print("PASS: 41,145 test loans, 6,025 adverse outcomes, 641 high-unemployment, 74 high adverse outcomes", flush=True)

for dcol, out in [("state", STATE_OUT), ("month", MONTH_OUT)]:
    summary = (meta.groupby([dcol, "high_unemployment"], dropna=False)
               .agg(n=("adverse", "size"), adverse_n=("adverse", "sum"))
               .reset_index())
    summary["adverse_rate"] = summary.adverse_n / summary.n
    summary.to_csv(out, index=False)

summary = (meta.groupby(["grade", "term", "high_unemployment"], dropna=False)
           .agg(n=("adverse", "size"), adverse_n=("adverse", "sum"))
           .reset_index())
summary["adverse_rate"] = summary.adverse_n / summary.n
summary.to_csv(GRD_OUT, index=False)

hi = meta.loc[meta.high_unemployment]
by_state = hi.groupby("state").agg(n=("adverse", "size"), adverse_n=("adverse", "sum")).sort_values("n", ascending=False)
hi_states = int(by_state.shape[0])
event_states = int((by_state.adverse_n > 0).sum())
hi_months = int(hi.month.nunique())
loan_shares = by_state.n.to_numpy(dtype=float) / len(hi)
event_shares = by_state.adverse_n.to_numpy(dtype=float) / hi.adverse.sum()
eff_state_loans = float(1. / np.sum(loan_shares ** 2))
eff_state_events = float(1. / np.sum(event_shares ** 2))

lines = [
    "ROUND 2 / REVIEWER 2 COMMENT 8 / SUBGROUP SUPPORT AUDIT",
    f"Python: {sys.version.split()[0]}, NumPy: {np.__version__}, pandas: {pd.__version__}",
    "This is descriptive and conditional on outcome-mature test selection.",
    "The unemployment threshold is 0.049 in source fraction units (4.9%).",
    f"All test: n={len(meta):,}, adverse={int(meta.adverse.sum()):,}",
    f"High unemployment: n={len(hi):,}, adverse={int(hi.adverse.sum()):,}",
    f"All observed test states: {meta.state.nunique()}",
    f"States contributing >=1 high-unemployment loan: {hi_states}",
    f"States contributing >=1 high-unemployment adverse outcome: {event_states}",
    f"Origination months contributing high-unemployment loans: {hi_months}",
    f"Effective number of high-subgroup states (loan-concentration index): {eff_state_loans:.3f}",
    f"Effective number of high-subgroup states (event-concentration index): {eff_state_events:.3f}",
    "",
    "HIGH UNEMPLOYMENT BY STATE:",
    by_state.to_string(),
    "",
    "INTERPRETATION:",
    "A state-cluster bootstrap can be fragile when the focal group comes from only a few states.",
    "State/month groups may overlap: rate status can change across months within the same state.",
    "Grade/term composition can confound unadjusted subgroup FNR comparisons.",
    "No causal or protected-class fairness conclusion is warranted.",
    "No multivariable adjustment, hypothesis test, or model retraining was performed.",
    "",
    f"Saved {STATE_OUT}",
    f"Saved {MONTH_OUT}",
    f"Saved {GRD_OUT}",
]
LOGS.mkdir(parents=True, exist_ok=True)
REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"High subgroup states: {hi_states}; with >=1 adverse outcome: {event_states}; months: {hi_months}")
print(f"Effective high-group states by loans: {eff_state_loans:.2f}; by adverse outcomes: {eff_state_events:.2f}")
print("\nHIGH UNEMPLOYMENT COUNTS BY STATE:\n", by_state.to_string())
print("\nSaved:", STATE_OUT, MONTH_OUT, GRD_OUT, REPORT, sep="\n")
print("STEP 33 COMPLETE: no training or original-file changes")
