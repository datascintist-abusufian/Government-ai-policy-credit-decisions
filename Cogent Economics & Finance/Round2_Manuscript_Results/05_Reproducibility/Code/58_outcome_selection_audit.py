import argparse
import ast
import re
from pathlib import Path
from collections import defaultdict

import pandas as pd


parser = argparse.ArgumentParser()
parser.add_argument("--root", required=True)
args = parser.parse_args()

root = Path(args.root)

dataset = root / "P2P_Macro_Data.dta"

notebook = root / "Economic and Finance.ipynb"

tables = (
    root
    / "Round2_Manuscript_Results"
    / "01_Tables"
)

evidence = (
    root
    / "Round2_Manuscript_Results"
    / "05_Reproducibility"
    / "Evidence_Checks"
)

tables.mkdir(parents=True, exist_ok=True)
evidence.mkdir(parents=True, exist_ok=True)

for path in [dataset, notebook]:
    if not path.exists():
        raise FileNotFoundError(path)

print("=" * 70)
print("STEP 58: OUTCOME-SELECTION AND REPRESENTATION AUDIT")
print("=" * 70)

# ------------------------------------------------------------
# 1. Recover terminal statuses from the original notebook
# ------------------------------------------------------------

import json

nb = json.loads(
    notebook.read_text(encoding="utf-8")
)

source = "".join(nb["cells"][27]["source"])

match = re.search(
    r"terminal_statuses\s*=\s*(\{.*?\})",
    source,
    re.DOTALL
)

if match is None:
    raise RuntimeError(
        "Could not recover original terminal-status definition."
    )

terminal_statuses = set(
    ast.literal_eval(match.group(1))
)

print("\nOriginal terminal statuses:")

for status in sorted(terminal_statuses):
    print(" -", status)

# ------------------------------------------------------------
# 2. Define original chronological evaluation periods
# ------------------------------------------------------------

windows = {
    "validation": (
        pd.Timestamp("2018-07-01"),
        pd.Timestamp("2019-04-01")
    ),
    "test": (
        pd.Timestamp("2019-04-01"),
        pd.Timestamp("2020-01-01")
    )
}

monthly = defaultdict(
    lambda: {
        "all_loans": 0,
        "terminal_selected": 0,
        "selected_badloan": 0
    }
)

statuses = defaultdict(int)

total_scanned = 0

# ------------------------------------------------------------
# 3. Read the original Stata dataset in chunks
# ------------------------------------------------------------

iterator = pd.read_stata(
    dataset,
    columns=[
        "issue_date",
        "loan_status",
        "badloan"
    ],
    chunksize=100000,
    iterator=True,
    convert_categoricals=True
)

for chunk in iterator:

    total_scanned += len(chunk)

    issue = pd.to_datetime(
        chunk["issue_date"],
        errors="coerce"
    )

    status = (
        chunk["loan_status"]
        .astype("string")
        .str.strip()
    )

    badloan = pd.to_numeric(
        chunk["badloan"],
        errors="coerce"
    )

    for cohort, (start, end) in windows.items():

        in_window = (
            (issue >= start)
            & (issue < end)
        )

        if not in_window.any():
            continue

        cohort_status = status.loc[in_window]

        selected = cohort_status.isin(
            terminal_statuses
        )

        cohort_badloan = badloan.loc[in_window]

        cohort_month = (
            issue.loc[in_window]
            .dt.to_period("M")
            .astype(str)
        )

        frame = pd.DataFrame({
            "issue_month": cohort_month.to_numpy(),
            "loan_status": cohort_status.to_numpy(),
            "terminal_selected": selected.to_numpy(),
            "selected_badloan": (
                selected
                & (cohort_badloan == 1)
            ).to_numpy()
        })

        group = frame.groupby(
            "issue_month",
            sort=True
        ).agg(
            all_loans=(
                "terminal_selected",
                "size"
            ),
            terminal_selected=(
                "terminal_selected",
                "sum"
            ),
            selected_badloan=(
                "selected_badloan",
                "sum"
            )
        )

        for month, row in group.iterrows():

            key = (cohort, month)

            monthly[key]["all_loans"] += int(
                row["all_loans"]
            )

            monthly[key]["terminal_selected"] += int(
                row["terminal_selected"]
            )

            monthly[key]["selected_badloan"] += int(
                row["selected_badloan"]
            )

        status_group = frame.groupby(
            "loan_status",
            dropna=False
        ).size()

        for loan_status, count in status_group.items():

            statuses[
                (
                    cohort,
                    str(loan_status)
                )
            ] += int(count)

    if total_scanned % 500000 < len(chunk):

        print(
            f"Rows examined: {total_scanned:,}"
        )

print("\nTotal source rows:", f"{total_scanned:,}")

if total_scanned != 2703430:
    raise RuntimeError(
        "Source row count differs from original experiment."
    )

# ------------------------------------------------------------
# 4. Construct monthly representation table
# ------------------------------------------------------------

monthly_rows = []

for (cohort, month), values in sorted(monthly.items()):

    n_all = values["all_loans"]

    n_selected = values["terminal_selected"]

    n_bad = values["selected_badloan"]

    monthly_rows.append({

        "cohort": cohort,

        "issue_month": month,

        "all_issued_loans": n_all,

        "terminal_selected_loans": n_selected,

        "excluded_loans": n_all - n_selected,

        "selected_fraction": (
            n_selected / n_all
            if n_all else float("nan")
        ),

        "selected_badloan_fraction": (
            n_bad / n_selected
            if n_selected else float("nan")
        )
    })

monthly_df = pd.DataFrame(monthly_rows)

# ------------------------------------------------------------
# 5. Calculate cohort-level summaries
# ------------------------------------------------------------

summary_rows = []

for cohort in windows:

    group = monthly_df[
        monthly_df["cohort"] == cohort
    ]

    n_all = int(
        group["all_issued_loans"].sum()
    )

    n_selected = int(
        group["terminal_selected_loans"].sum()
    )

    selected_bad = sum(
        monthly[(cohort, month)]["selected_badloan"]
        for month in group["issue_month"]
    )

    summary_rows.append({

        "cohort": cohort,

        "all_issued_loans": n_all,

        "terminal_selected_loans": n_selected,

        "excluded_loans": n_all - n_selected,

        "selection_percent": (
            100 * n_selected / n_all
            if n_all else float("nan")
        ),

        "selected_badloan_percent": (
            100 * selected_bad / n_selected
            if n_selected else float("nan")
        )
    })

summary_df = pd.DataFrame(summary_rows)

# ------------------------------------------------------------
# 6. Verify original evaluation sample counts
# ------------------------------------------------------------

print("\n===== COHORT SUMMARY =====")

print(
    summary_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

expected_selected = {
    "validation": 27374,
    "test": 41145
}

expected_test_all = 401169

for cohort, expected in expected_selected.items():

    observed = int(
        summary_df.loc[
            summary_df["cohort"] == cohort,
            "terminal_selected_loans"
        ].iloc[0]
    )

    if observed != expected:

        raise RuntimeError(
            f"{cohort}: expected {expected} "
            f"selected loans, found {observed}. "
            "The status-selection reconstruction "
            "must be investigated."
        )

test_all = int(
    summary_df.loc[
        summary_df["cohort"] == "test",
        "all_issued_loans"
    ].iloc[0]
)

if test_all != expected_test_all:

    raise RuntimeError(
        f"Expected {expected_test_all} test-period loans; "
        f"found {test_all}."
    )

print("\nPASS: Original evaluation counts reproduced.")

# ------------------------------------------------------------
# 7. Show monthly and status distributions
# ------------------------------------------------------------

print("\n===== MONTHLY REPRESENTATION =====")

print(
    monthly_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

status_rows = []

for (cohort, status), n in sorted(statuses.items()):

    status_rows.append({

        "cohort": cohort,

        "loan_status": status,

        "count": n,

        "terminal_selected": (
            status in terminal_statuses
        )
    })

status_df = pd.DataFrame(status_rows)

print("\n===== LOAN-STATUS DISTRIBUTION =====")

print(
    status_df.to_string(index=False)
)

# ------------------------------------------------------------
# 8. Save supporting tables
# ------------------------------------------------------------

monthly_path = (
    tables
    / "58_monthly_outcome_selection.csv"
)

summary_path = (
    tables
    / "58_outcome_selection_summary.csv"
)

status_path = (
    tables
    / "58_loan_status_distribution.csv"
)

monthly_df.to_csv(
    monthly_path,
    index=False
)

summary_df.to_csv(
    summary_path,
    index=False
)

status_df.to_csv(
    status_path,
    index=False
)

report_path = (
    evidence
    / "58_outcome_selection_report.txt"
)

with open(
    report_path,
    "w",
    encoding="utf-8"
) as f:

    f.write("STEP 58 OUTCOME-SELECTION AUDIT\n\n")

    f.write(
        summary_df.to_string(index=False)
    )

    f.write("\n\n")

    f.write(
        monthly_df.to_string(index=False)
    )

    f.write(
        "\n\nLIMITATIONS\n"
        "Recorded terminal statuses are used as "
        "the selection indicator. This does not "
        "establish when each terminal outcome "
        "occurred or whether the recorded status "
        "was known at the stated follow-up cutoff.\n"
        "The selected badloan fraction is not "
        "a fixed-horizon default probability.\n"
        "Selection differences are descriptive "
        "and do not themselves identify or remove "
        "selection bias.\n"
    )

print("\n===== SAVED FILES =====")

print(monthly_path)
print(summary_path)
print(status_path)
print(report_path)

print("\nSTEP 58 COMPLETE")
print("Original data and experimental results unchanged.")
