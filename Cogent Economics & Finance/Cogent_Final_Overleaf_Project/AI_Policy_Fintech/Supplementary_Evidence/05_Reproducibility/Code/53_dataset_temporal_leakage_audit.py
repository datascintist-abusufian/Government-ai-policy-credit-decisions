import argparse
from pathlib import Path
from collections import Counter

import pandas as pd


parser = argparse.ArgumentParser()
parser.add_argument("--root", required=True)
args = parser.parse_args()

root = Path(args.root)

dataset = root / "P2P_Macro_Data.dta"

output = (
    root
    / "Round2_Manuscript_Results"
    / "05_Reproducibility"
    / "Evidence_Checks"
)

output.mkdir(parents=True, exist_ok=True)

if not dataset.exists():
    raise FileNotFoundError(dataset)

print("=" * 65)
print("STEP 53: DATASET-LEVEL TEMPORAL LEAKAGE AUDIT")
print("=" * 65)

# ------------------------------------------------------------
# Identify available columns without loading the full dataset
# ------------------------------------------------------------

reader = pd.read_stata(
    dataset,
    iterator=True,
    convert_categoricals=False
)

first_row = reader.read(1)
available = set(first_row.columns)
reader.close()

wanted = [
    "issue_date",
    "last_date",
    "last_pymnt_d",
    "year",
    "month",
    "quarter",
    "loan_vol",
    "median_dummy",
    "loan_vol6m",
    "loglvol_6m",
    "badloan"
]

present = [c for c in wanted if c in available]
missing = [c for c in wanted if c not in available]

print("\nAvailable audit columns:")
print(present)

print("\nMissing audit columns:")
print(missing)

if "issue_date" not in present:
    raise ValueError("issue_date is missing.")

if "last_date" not in present:
    raise ValueError("last_date is missing.")

# ------------------------------------------------------------
# Inspect the complete dataset in memory-safe chunks
# ------------------------------------------------------------

counts = Counter()

iterator = pd.read_stata(
    dataset,
    columns=present,
    chunksize=100000,
    iterator=True,
    convert_categoricals=False
)

for chunk in iterator:

    n = len(chunk)
    counts["total_rows"] += n

    issue = pd.to_datetime(
        chunk["issue_date"],
        errors="coerce"
    )

    last = pd.to_datetime(
        chunk["last_date"],
        errors="coerce"
    )

    valid_dates = issue.notna() & last.notna()

    different_month = (
        valid_dates
        & (
            issue.dt.to_period("M")
            != last.dt.to_period("M")
        )
    )

    counts["valid_date_pairs"] += int(
        valid_dates.sum()
    )

    counts["different_issue_last_month"] += int(
        different_month.sum()
    )

    # Compare observed year/month/quarter with
    # origination date and last-payment-derived date.

    date_components = {
        "year": (
            issue.dt.year,
            last.dt.year
        ),

        "month": (
            issue.dt.month,
            last.dt.month
        ),

        "quarter": (
            issue.dt.quarter,
            last.dt.quarter
        )
    }

    for feature, (issue_value, last_value) in date_components.items():

        if feature not in chunk.columns:
            continue

        recorded = pd.to_numeric(
            chunk[feature],
            errors="coerce"
        )

        eligible = recorded.notna() & valid_dates

        eligible_different = (
            eligible & different_month
        )

        counts[f"{feature}_eligible"] += int(
            eligible.sum()
        )

        counts[f"{feature}_matches_issue"] += int(
            (eligible & (recorded == issue_value)).sum()
        )

        counts[f"{feature}_matches_last"] += int(
            (eligible & (recorded == last_value)).sum()
        )

        counts[f"{feature}_different_eligible"] += int(
            eligible_different.sum()
        )

        counts[f"{feature}_different_matches_issue"] += int(
            (
                eligible_different
                & (recorded == issue_value)
            ).sum()
        )

        counts[f"{feature}_different_matches_last"] += int(
            (
                eligible_different
                & (recorded == last_value)
            ).sum()
        )

    if counts["total_rows"] % 500000 < n:
        print(
            f"Rows examined: {counts['total_rows']:,}"
        )

print("\n===== DATASET AUDIT RESULTS =====")

print(
    "Total rows:",
    f"{counts['total_rows']:,}"
)

print(
    "Valid issue/last-date pairs:",
    f"{counts['valid_date_pairs']:,}"
)

print(
    "Issue and last dates differ by month:",
    f"{counts['different_issue_last_month']:,}"
)

# ------------------------------------------------------------
# Calculate matching percentages
# ------------------------------------------------------------

results = []

for feature in ["year", "month", "quarter"]:

    if feature not in present:
        continue

    n_all = counts[f"{feature}_eligible"]

    n_diff = counts[
        f"{feature}_different_eligible"
    ]

    issue_matches = counts[
        f"{feature}_matches_issue"
    ]

    last_matches = counts[
        f"{feature}_matches_last"
    ]

    issue_diff = counts[
        f"{feature}_different_matches_issue"
    ]

    last_diff = counts[
        f"{feature}_different_matches_last"
    ]

    result = {
        "feature": feature,
        "eligible_rows": n_all,
        "matches_issue_date": issue_matches,
        "matches_last_date": last_matches,
        "issue_match_percent": (
            100 * issue_matches / n_all
            if n_all else float("nan")
        ),
        "last_match_percent": (
            100 * last_matches / n_all
            if n_all else float("nan")
        ),
        "different_date_rows": n_diff,
        "different_date_matches_issue": issue_diff,
        "different_date_matches_last": last_diff,
        "different_date_issue_match_percent": (
            100 * issue_diff / n_diff
            if n_diff else float("nan")
        ),
        "different_date_last_match_percent": (
            100 * last_diff / n_diff
            if n_diff else float("nan")
        )
    }

    results.append(result)

    print(f"\nFEATURE: {feature}")

    print(
        "Matches origination date:",
        f"{result['issue_match_percent']:.2f}%"
    )

    print(
        "Matches last-payment-derived date:",
        f"{result['last_match_percent']:.2f}%"
    )

    print(
        "Among different-date records, "
        "matches origination:",
        f"{result['different_date_issue_match_percent']:.2f}%"
    )

    print(
        "Among different-date records, "
        "matches last date:",
        f"{result['different_date_last_match_percent']:.2f}%"
    )

# ------------------------------------------------------------
# Save results
# ------------------------------------------------------------

csv_path = (
    output
    / "53_temporal_feature_date_comparison.csv"
)

pd.DataFrame(results).to_csv(
    csv_path,
    index=False
)

summary_path = (
    output
    / "53_temporal_feature_audit_summary.txt"
)

with open(summary_path, "w", encoding="utf-8") as f:

    f.write("STEP 53 TEMPORAL FEATURE AUDIT\n\n")

    for key, value in counts.items():
        f.write(f"{key}: {value}\n")

    f.write("\nAVAILABLE COLUMNS\n")
    f.write(", ".join(present))

    f.write("\n\nMISSING COLUMNS\n")
    f.write(", ".join(missing))

    f.write(
        "\n\nThis is a source-data consistency audit. "
        "It does not independently certify the exact "
        "historical feature-generation procedure.\n"
    )

print("\nSaved CSV:", csv_path)
print("Saved summary:", summary_path)

print("\nSTEP 53 COMPLETE")
print("Original dataset and model results unchanged.")
