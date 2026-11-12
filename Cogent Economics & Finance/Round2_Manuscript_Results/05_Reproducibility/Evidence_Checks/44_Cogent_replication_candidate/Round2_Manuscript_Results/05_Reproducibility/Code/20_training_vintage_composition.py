"""Round 2, Reviewer 2 Comment 3: origination-year composition of exact nested training samples.

Read-only audit of the original Stata data. Reconstructs the same chronological
eligibility, maturity rule, original row order, and NumPy seed-42 permutation as
Step 13. Does NOT fit a model or change previous outputs.
"""

from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

ROOT = Path(r"E:\Governance-Policy\Cogent Economics & Finance")
DATA = ROOT / "P2P_Macro_Data.dta"
RESULTS = ROOT / "Round2_Manuscript_Results"
TABLE_DIR = RESULTS / "01_Tables"
LOG_DIR = RESULTS / "05_Reproducibility" / "Execution_Logs"

CUTOFF = pd.Timestamp("2020-04-01")
CHUNK_SIZE = 200_000
SEED = 42
EXPECTED_N = 1_304_058
SIZES = (180_000, 360_000, 540_000, EXPECTED_N)

if not DATA.is_file():
    raise FileNotFoundError(f"Source dataset not found: {DATA}")

TABLE_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

print("Round 2 | Reviewer 2 Comment 3 | Training-vintage composition")
print("Dataset:", DATA)
print("Reconstructing the exact seed-42 nested samples from Step 13 ...", flush=True)

# One compact value per selected loan. Keep original dataset row order.
year_chunks = []
scanned = 0
for chunk in pd.read_stata(
    DATA,
    columns=["issue_d", "loan_status", "term"],
    convert_categoricals=False,
    chunksize=CHUNK_SIZE,
):
    scanned += len(chunk)
    # Match Step 13 exactly, including date and term parsing.
    issue = pd.to_datetime(
        chunk["issue_d"].astype(str).str.strip(),
        format="%b-%y", errors="coerce",
    )
    term = pd.to_numeric(
        chunk["term"].astype(str).str.extract(r"(\d+)")[0],
        errors="coerce",
    )
    status = chunk["loan_status"].astype(str).str.strip()
    is_train = (issue >= pd.Timestamp("2012-08-01")) & (
        issue <= pd.Timestamp("2018-06-01")
    )
    terminal = status.isin(["Fully Paid", "Charged Off", "Default"])
    elapsed_months = (CUTOFF.year - issue.dt.year) * 12 + (
        CUTOFF.month - issue.dt.month
    )
    mature = terminal | (elapsed_months >= term)
    selected = (is_train & mature).to_numpy()
    if selected.any():
        year_chunks.append(issue[selected].dt.year.to_numpy(dtype=np.int16))
    print(f"  Scanned {scanned:,} records", flush=True)

if not year_chunks:
    raise RuntimeError("No records matched the training eligibility rule.")

years = np.concatenate(year_chunks)
del year_chunks
n_total = len(years)
if n_total != EXPECTED_N:
    raise RuntimeError(
        f"Eligibility mismatch: {n_total:,} records, expected {EXPECTED_N:,}. "
        "Stopped without writing results."
    )
if not np.isin(years, np.arange(2012, 2019)).all():
    raise RuntimeError("Unexpected origination year in reconstructed training sample.")

# Exactly the same permutation used by Step 13, not a fresh random sample.
permutation = np.random.default_rng(SEED).permutation(n_total)
vintages = np.arange(2012, 2019)
full_counts = np.array([(years == year).sum() for year in vintages])
full_pct = full_counts / n_total * 100

rows = []
summary = []
for size in SIZES:
    selected_years = years[permutation[:size]]
    counts = np.array([(selected_years == year).sum() for year in vintages])
    shares = counts / size * 100
    diff = shares - full_pct
    for year, count, pct, reference, difference in zip(
        vintages, counts, shares, full_pct, diff
    ):
        rows.append({
            "training_n": int(size),
            "sampling_seed": SEED,
            "origination_year": int(year),
            "loans_n": int(count),
            "share_pct": round(float(pct), 4),
            "full_training_share_pct": round(float(reference), 4),
            "difference_from_full_pp": round(float(difference), 4),
        })
    summary.append({
        "training_n": int(size),
        "max_abs_year_share_diff_pp": round(float(np.max(np.abs(diff))), 4),
        "total_variation_distance_pct": round(float(0.5 * np.sum(np.abs(diff))), 4),
        "check_counts_sum_to_sample_size": bool(counts.sum() == size),
    })

result = pd.DataFrame(rows)
summary_df = pd.DataFrame(summary)
if not summary_df["check_counts_sum_to_sample_size"].all():
    raise RuntimeError("Composition totals failed verification.")

csv_path = TABLE_DIR / "20_training_vintage_composition.csv"
report_path = LOG_DIR / "20_training_vintage_composition_report.txt"
result.to_csv(csv_path, index=False)
report = "\n".join([
    "ROUND 2, REVIEWER 2, COMMENT 3: ORIGINATION-YEAR AUDIT",
    f"Generated: {datetime.now().isoformat(timespec='seconds')}",
    f"Source: {DATA}",
    f"Rows scanned: {scanned:,}",
    f"Reconstructed mature training loans: {n_total:,}",
    "Sample design: one NumPy default_rng(42) permutation of rows in source order;",
    "              first 180k, 360k, and 540k rows as nested subsets; full cohort.",
    "Outcome-maturity eligibility: terminal status OR elapsed calendar months",
    "                              from issue month to April 2020 >= term.",
    "This script performs no model fitting and no statistical hypothesis test.",
    "",
    "SUMMARY: maximum absolute origination-year share difference from full cohort",
    summary_df.to_string(index=False),
    "",
    "COMPLETE YEAR COMPOSITION:",
    result.to_string(index=False),
    "",
    "INTERPRETATION:",
    "Similarity in year distributions can reduce concern about simple vintage",
    "imbalance but does not establish that the model-ranking reversal is caused",
    "by training size alone. Seed stability and preprocessing effects remain",
    "separate questions. These results are descriptive, not causal.",
])
report_path.write_text(report + "\n", encoding="utf-8")

print("\nCOMPOSITION CHECK (percentage points):")
print(summary_df.to_string(index=False))
print("\nSaved:", csv_path)
print("Saved:", report_path)
print("STEP 20 COMPLETE. No models were retrained.")
