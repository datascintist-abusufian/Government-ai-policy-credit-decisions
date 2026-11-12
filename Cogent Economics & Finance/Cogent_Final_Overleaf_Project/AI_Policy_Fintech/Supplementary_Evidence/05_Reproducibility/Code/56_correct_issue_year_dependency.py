import argparse
import re
from pathlib import Path

import pandas as pd


parser = argparse.ArgumentParser()
parser.add_argument("--root", required=True)
args = parser.parse_args()

root = Path(args.root)

dataset = root / "P2P_Macro_Data.dta"
stata = root / "P2P_Macro_Codes.do"

evidence = (
    root / "Round2_Manuscript_Results"
    / "05_Reproducibility"
    / "Evidence_Checks"
)

register_path = (
    evidence / "55_indirect_feature_dependency_register.csv"
)

for path in [dataset, stata, register_path]:
    if not path.exists():
        raise FileNotFoundError(path)

print("=" * 65)
print("STEP 56: CORRECT ISSUE_YEAR DEPENDENCY")
print("=" * 65)

# ------------------------------------------------------------
# 1. Examine Stata source
# ------------------------------------------------------------

print("\n===== STATA SOURCE CONSTRUCTION =====")

source = stata.read_text(
    encoding="utf-8-sig",
    errors="replace"
).splitlines()

relevant = []

for number, line in enumerate(source, 1):

    if (
        re.search(r"\bissue_daily\b", line)
        or re.search(r"\bissue_year\b", line)
    ):
        relevant.append((number, line.strip()))

for number, line in relevant:
    print(f"Line {number}: {line}")

issue_year_uses_function = any(
    re.search(
        r"\b(?:gen|generate)\s+issue_year\s*="
        r"\s*year\s*\(\s*issue_daily\s*\)",
        line,
        re.IGNORECASE
    )
    for _, line in relevant
)

issue_daily_uses_issue_date = any(
    re.search(
        r"\b(?:gen|generate)\s+issue_daily\s*=",
        line,
        re.IGNORECASE
    )
    and re.search(
        r"\bissue_date\b",
        line,
        re.IGNORECASE
    )
    for _, line in relevant
)

print(
    "\nissue_year uses Stata year() function:",
    issue_year_uses_function
)

print(
    "issue_daily constructed using issue_date:",
    issue_daily_uses_issue_date
)

# ------------------------------------------------------------
# 2. Verify actual dataset values
# ------------------------------------------------------------

print("\n===== DATASET VALIDATION =====")

total = 0
valid = 0
matches_issue = 0
matches_last = 0

different_year = 0
different_matches_issue = 0

iterator = pd.read_stata(
    dataset,
    columns=[
        "issue_date",
        "last_date",
        "issue_year"
    ],
    chunksize=100000,
    iterator=True,
    convert_categoricals=False
)

for chunk in iterator:

    total += len(chunk)

    issue = pd.to_datetime(
        chunk["issue_date"],
        errors="coerce"
    )

    last = pd.to_datetime(
        chunk["last_date"],
        errors="coerce"
    )

    recorded = pd.to_numeric(
        chunk["issue_year"],
        errors="coerce"
    )

    eligible = (
        issue.notna()
        & last.notna()
        & recorded.notna()
    )

    issue_match = (
        eligible
        & (recorded == issue.dt.year)
    )

    last_match = (
        eligible
        & (recorded == last.dt.year)
    )

    different = (
        eligible
        & (issue.dt.year != last.dt.year)
    )

    valid += int(eligible.sum())

    matches_issue += int(issue_match.sum())
    matches_last += int(last_match.sum())

    different_year += int(different.sum())

    different_matches_issue += int(
        (different & issue_match).sum()
    )

    if total % 500000 < len(chunk):
        print(f"Rows examined: {total:,}")

issue_match_pct = (
    100 * matches_issue / valid
    if valid else 0
)

last_match_pct = (
    100 * matches_last / valid
    if valid else 0
)

print("\nTotal loans:", f"{total:,}")
print("Valid observations:", f"{valid:,}")

print(
    "issue_year matches origination year:",
    f"{issue_match_pct:.4f}%"
)

print(
    "issue_year matches last-payment year:",
    f"{last_match_pct:.4f}%"
)

print(
    "Different origination/payment years:",
    f"{different_year:,}"
)

print(
    "Matches origination among different years:",
    f"{different_matches_issue:,}"
)

# ------------------------------------------------------------
# 3. Correct dependency classification if verified
# ------------------------------------------------------------

print("\n===== CORRECTED DEPENDENCY REGISTER =====")

register = pd.read_csv(
    register_path,
    dtype=str
).fillna("")

if len(register) != 150:
    raise ValueError(
        "Expected 150 original feature records."
    )

verified = (
    issue_year_uses_function
    and issue_daily_uses_issue_date
    and valid > 0
    and matches_issue == valid
    and different_year > 0
    and different_matches_issue == different_year
)

register["correction_note"] = ""

mask = register["feature"] == "issue_year"

if mask.sum() != 1:
    raise ValueError(
        "Expected exactly one issue_year record."
    )

if verified:

    register.loc[
        mask,
        "payment_history_dependency"
    ] = "False"

    register.loc[
        mask,
        "correction_note"
    ] = (
        "Corrected false positive: year() is a "
        "Stata function. issue_year is derived "
        "from the origination date."
    )

    print(
        "PASS: issue_year payment-history "
        "dependency classified as false positive."
    )

else:

    register.loc[
        mask,
        "correction_note"
    ] = (
        "Correction not applied; source or "
        "dataset verification incomplete."
    )

    print(
        "WARNING: issue_year correction "
        "not independently verified."
    )

output = (
    evidence
    / "56_corrected_payment_history_dependency_register.csv"
)

register.to_csv(
    output,
    index=False
)

remaining = register[
    register["payment_history_dependency"]
    .str.lower() == "true"
]

remaining145 = remaining[
    remaining["included_candidate_145"]
    .str.lower() == "true"
]

remaining133 = remaining[
    remaining["included_candidate_133"]
    .str.lower() == "true"
]

print("\n===== FINAL RESULTS =====")

print(
    "Payment-history-dependent features:",
    len(remaining)
)

print(
    "Remaining in 145-feature candidate:",
    len(remaining145)
)

print(
    "Remaining in 133-feature candidate:",
    len(remaining133)
)

print("\nIdentified features:")

for name in remaining["feature"]:
    print(" -", name)

print("\nSaved:", output)

print("\nSTEP 56 COMPLETE")
print("Original experiments unchanged.")
