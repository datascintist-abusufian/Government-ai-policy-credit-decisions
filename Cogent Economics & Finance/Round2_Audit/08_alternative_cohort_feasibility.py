from pathlib import Path
from collections import Counter
from datetime import datetime
import pandas as pd
import numpy as np

ROOT = Path(r"E:\Governance-Policy\Cogent Economics & Finance")

DATA = ROOT / "P2P_Macro_Data.dta"
AUDIT = ROOT / "Round2_Audit"
TABLES = ROOT / "Round2_Manuscript_Results" / "01_Tables"

CUTOFF = pd.Timestamp("2020-04-01")

CHUNK_SIZE = 200000

COLUMNS = [
    "issue_date",
    "term",
    "loan_status",
    "badloan"
]

if not DATA.exists():
    raise FileNotFoundError(DATA)

print("=" * 70)
print("ROUND 2 - STEP 8")
print("ALTERNATIVE COHORT FEASIBILITY AUDIT")
print("=" * 70)

print("Dataset:", DATA)
print("Administrative cutoff:", CUTOFF.date())
print("Chunk size:", CHUNK_SIZE)

counts = Counter()
total_scanned = 0

# ------------------------------------------------------------
# 1. Read data in manageable chunks
# ------------------------------------------------------------

with pd.read_stata(
    DATA,
    columns=COLUMNS,
    convert_categoricals=False,
    chunksize=CHUNK_SIZE
) as reader:

    for chunk_number, df in enumerate(reader, start=1):

        total_scanned += len(df)

        issue = pd.to_datetime(
            df["issue_date"],
            errors="coerce"
        )

        term = pd.to_numeric(
            df["term"]
            .fillna("")
            .astype(str)
            .str.extract(r"(\d+)", expand=False),
            errors="coerce"
        )

        status = (
            df["loan_status"]
            .fillna("Missing")
            .astype(str)
            .str.strip()
        )

        bad = pd.to_numeric(
            df["badloan"],
            errors="coerce"
        )

        # Contractual follow-up in complete calendar months
        followup = (
            (CUTOFF.year - issue.dt.year) * 12
            +
            (CUTOFF.month - issue.dt.month)
        )

        contractually_eligible = (
            followup >= term
        )

        # Only study years and recognised contract lengths
        valid = (
            issue.notna()
            &
            (issue >= pd.Timestamp("2012-08-01"))
            &
            (issue < pd.Timestamp("2020-01-01"))
            &
            term.isin([36, 60])
        )

        if not valid.any():
            continue

        label = np.select(
            [
                bad.eq(1),
                bad.eq(0)
            ],
            [
                "Adverse",
                "Non_adverse"
            ],
            default="Missing_or_other"
        )

        work = pd.DataFrame({
            "issue_year": issue.dt.year,
            "term_months": term,
            "contractually_eligible": contractually_eligible,
            "loan_status": status,
            "label": label
        })

        work = work.loc[valid].copy()

        grouped = work.groupby(
            [
                "issue_year",
                "term_months",
                "contractually_eligible",
                "loan_status",
                "label"
            ],
            dropna=False
        ).size()

        for key, n in grouped.items():

            counts[tuple(key)] += int(n)

        print(
            f"Chunk {chunk_number:02d} | "
            f"Processed {total_scanned:,} records",
            flush=True
        )

# ------------------------------------------------------------
# 2. Reconstruct aggregated vintage dataset
# ------------------------------------------------------------

rows = []

for key, n in counts.items():

    year, term, eligible, status, label = key

    rows.append({
        "issue_year": int(year),
        "term_months": int(term),
        "contractually_eligible": bool(eligible),
        "loan_status": status,
        "label": label,
        "n": int(n)
    })

detail = pd.DataFrame(rows)

if detail.empty:
    raise RuntimeError(
        "No eligible records found. Check the input columns."
    )

# ------------------------------------------------------------
# 3. Vintage and contractual eligibility summary
# ------------------------------------------------------------

detail["adverse_n"] = np.where(
    detail["label"].eq("Adverse"),
    detail["n"],
    0
)

detail["non_adverse_n"] = np.where(
    detail["label"].eq("Non_adverse"),
    detail["n"],
    0
)

detail["unknown_label_n"] = np.where(
    detail["label"].eq("Missing_or_other"),
    detail["n"],
    0
)

vintage = (
    detail.groupby(
        [
            "issue_year",
            "term_months",
            "contractually_eligible"
        ],
        as_index=False
    )
    .agg(
        total_loans=("n", "sum"),
        adverse_n=("adverse_n", "sum"),
        non_adverse_n=("non_adverse_n", "sum"),
        unknown_label_n=("unknown_label_n", "sum")
    )
)

vintage["adverse_rate_pct"] = (
    100
    *
    vintage["adverse_n"]
    /
    vintage["total_loans"]
)

vintage = vintage.sort_values(
    [
        "issue_year",
        "term_months",
        "contractually_eligible"
    ]
)

vintage_path = (
    TABLES /
    "08_contractual_maturity_by_vintage.csv"
)

vintage.to_csv(
    vintage_path,
    index=False
)

# ------------------------------------------------------------
# 4. Evaluate candidate chronological model split
# ------------------------------------------------------------

candidate_groups = {
    "Training": (2012, 2014),
    "Validation": (2015, 2015),
    "Test": (2016, 2016)
}

candidate_rows = []

for group, (start_year, end_year) in candidate_groups.items():

    selected = detail.loc[
        detail["issue_year"].between(
            start_year,
            end_year
        )
        &
        detail["term_months"].eq(36)
        &
        detail["contractually_eligible"]
    ].copy()

    n_total = int(selected["n"].sum())

    n_bad = int(selected["adverse_n"].sum())

    n_good = int(selected["non_adverse_n"].sum())

    n_unknown = int(selected["unknown_label_n"].sum())

    current_n = int(
        selected.loc[
            selected["loan_status"].eq("Current"),
            "n"
        ].sum()
    )

    candidate_rows.append({
        "split": group,
        "start_year": start_year,
        "end_year": end_year,
        "term_months": 36,
        "n_total": n_total,
        "n_adverse": n_bad,
        "n_non_adverse": n_good,
        "n_unknown_labels": n_unknown,
        "n_current_status": current_n,
        "adverse_rate_pct": (
            100 * n_bad / n_total
            if n_total else np.nan
        ),
        "all_contractually_eligible": True
    })

candidate = pd.DataFrame(candidate_rows)

candidate_path = (
    TABLES /
    "08_alternative_36month_split_feasibility.csv"
)

candidate.to_csv(
    candidate_path,
    index=False
)

# ------------------------------------------------------------
# 5. Save audit report
# ------------------------------------------------------------

report_path = (
    AUDIT /
    "08_alternative_cohort_feasibility_report.txt"
)

report_lines = [
    "ROUND 2 - REVIEWER 2 COMMENT 1",
    "Alternative cohort feasibility audit",
    "",
    f"Generated: {datetime.now()}",
    f"Dataset: {DATA}",
    f"Total records scanned: {total_scanned:,}",
    f"Administrative reference cutoff: {CUTOFF.date()}",
    "",
    "PURPOSE:",
    "Investigate cohort selection independent",
    "of realised loan status.",
    "",
    "ELIGIBILITY DEFINITION:",
    "Months from origination to cutoff >= contract term.",
    "",
    "IMPORTANT LIMITATIONS:",
    "This is an eligibility feasibility analysis.",
    "It does not prove that all final statuses were",
    "first observed by the administrative cutoff.",
    "The badloan label is not a verified fixed-36-month",
    "outcome and must not be described as such.",
    "",
    "ALTERNATIVE CHRONOLOGICAL SPLIT:",
    candidate.to_string(index=False),
    "",
    "CONTRACTUAL MATURITY BY VINTAGE:",
    vintage.to_string(index=False)
]

report_path.write_text(
    "\n".join(report_lines),
    encoding="utf-8"
)

# ------------------------------------------------------------
# 6. Display findings
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("ALTERNATIVE 36-MONTH COHORT")
print("=" * 70)

print(candidate.to_string(index=False))

print("\n" + "=" * 70)
print("VINTAGE ELIGIBILITY")
print("=" * 70)

print(
    vintage[
        [
            "issue_year",
            "term_months",
            "contractually_eligible",
            "total_loans",
            "adverse_n",
            "adverse_rate_pct"
        ]
    ].to_string(index=False)
)

print("\n" + "=" * 70)
print("OUTPUTS SAVED")
print("=" * 70)

print(vintage_path)
print(candidate_path)
print(report_path)

print("\nSTEP 8 COMPLETED SUCCESSFULLY")