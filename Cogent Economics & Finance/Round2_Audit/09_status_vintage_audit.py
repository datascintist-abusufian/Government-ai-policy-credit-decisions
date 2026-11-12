from pathlib import Path
from collections import Counter
import pandas as pd
import numpy as np

ROOT = Path(r"E:\Governance-Policy\Cogent Economics & Finance")

DATA = ROOT / "P2P_Macro_Data.dta"

AUDIT = ROOT / "Round2_Audit"

TABLES = ROOT / "Round2_Manuscript_Results" / "01_Tables"

CHUNK = 200000

COLS = [
    "issue_date",
    "term",
    "loan_status",
    "badloan",
    "last_pymnt_d",
    "next_pymnt_d",
    "last_credit_pull_d"
]

if not DATA.exists():
    raise FileNotFoundError(DATA)

print("=" * 70)
print("STEP 9: LOAN-STATUS VINTAGE AUDIT")
print("=" * 70)

print("Investigating 36-month loans originated in 2012-2016")

counts = Counter()
date_counts = Counter()

total_scanned = 0
total_selected = 0


def parse_month(s):

    raw = s.fillna("").astype(str).str.strip()

    result = pd.to_datetime(
        raw,
        format="%b-%y",
        errors="coerce"
    )

    other = result.isna() & raw.ne("")

    if other.any():

        result.loc[other] = pd.to_datetime(
            raw.loc[other],
            format="%b-%Y",
            errors="coerce"
        )

    return result


with pd.read_stata(
    DATA,
    columns=COLS,
    convert_categoricals=False,
    chunksize=CHUNK
) as reader:

    for chunk_no, df in enumerate(reader, 1):

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

        mask = (
            (issue >= pd.Timestamp("2012-08-01"))
            &
            (issue < pd.Timestamp("2017-01-01"))
            &
            term.eq(36)
        )

        if not mask.any():
            print(f"Chunk {chunk_no}: {total_scanned:,} processed")
            continue

        selected = df.loc[mask].copy()

        selected_issue = issue.loc[mask]

        selected["issue_year"] = selected_issue.dt.year

        selected["issue_month"] = (
            selected_issue.dt.strftime("%Y-%m")
        )

        selected["loan_status"] = (
            selected["loan_status"]
            .fillna("Missing")
            .astype(str)
            .str.strip()
        )

        bad = pd.to_numeric(
            selected["badloan"],
            errors="coerce"
        )

        selected["label"] = np.select(
            [
                bad.eq(1),
                bad.eq(0)
            ],
            [
                "Adverse",
                "Non_adverse"
            ],
            default="Unknown"
        )

        total_selected += len(selected)

        grouped = selected.groupby(
            [
                "issue_year",
                "issue_month",
                "loan_status",
                "label"
            ],
            dropna=False
        ).size()

        for key, n in grouped.items():
            counts[tuple(key)] += int(n)

        # Examine date-field years specifically for
        # 2016 loans that are recorded as Current

        current_2016 = selected[
            selected["issue_year"].eq(2016)
            &
            selected["loan_status"].eq("Current")
        ]

        for field in [
            "last_pymnt_d",
            "next_pymnt_d",
            "last_credit_pull_d"
        ]:

            dates = parse_month(current_2016[field])

            years = (
                dates.dt.year
                .fillna(-1)
                .astype(int)
            )

            year_counts = years.value_counts()

            for year, n in year_counts.items():

                date_counts[
                    (field, int(year))
                ] += int(n)

        print(
            f"Chunk {chunk_no:02d} | "
            f"Total scanned: {total_scanned:,} | "
            f"Selected: {total_selected:,}",
            flush=True
        )


# ============================================================
# Build aggregated records
# ============================================================

rows = []

for key, n in counts.items():

    year, month, status, label = key

    rows.append({
        "issue_year": int(year),
        "issue_month": month,
        "loan_status": status,
        "label": label,
        "n": int(n)
    })

detail = pd.DataFrame(rows)

if detail.empty:
    raise RuntimeError("No matching observations found.")


# ============================================================
# A. Loan status by year
# ============================================================

status_table = (
    detail.groupby(
        ["issue_year", "loan_status"],
        as_index=False
    )["n"].sum()
)

year_totals = (
    status_table.groupby("issue_year")["n"]
    .sum()
    .rename("year_total")
    .reset_index()
)

status_table = status_table.merge(
    year_totals,
    on="issue_year"
)

status_table["percent"] = (
    100
    *
    status_table["n"]
    /
    status_table["year_total"]
)

status_path = (
    TABLES /
    "09_status_by_vintage_PRELIMINARY.csv"
)

status_table.to_csv(status_path, index=False)


# ============================================================
# B. Loan status against adverse label
# ============================================================

label_table = (
    detail.groupby(
        ["issue_year", "loan_status", "label"],
        as_index=False
    )["n"].sum()
)

label_path = (
    TABLES /
    "09_status_label_crosscheck_PRELIMINARY.csv"
)

label_table.to_csv(label_path, index=False)


# ============================================================
# C. Monthly distribution for 2016
# ============================================================

monthly = detail[
    detail["issue_year"].eq(2016)
].copy()

monthly_total = (
    monthly.groupby("issue_month")["n"]
    .sum()
    .rename("total")
)

monthly_current = (
    monthly[
        monthly["loan_status"].eq("Current")
    ]
    .groupby("issue_month")["n"]
    .sum()
    .rename("current")
)

monthly_table = pd.concat(
    [monthly_total, monthly_current],
    axis=1
).fillna(0).reset_index()

monthly_table["current_pct"] = (
    100
    *
    monthly_table["current"]
    /
    monthly_table["total"]
)

monthly_path = (
    TABLES /
    "09_2016_monthly_current_share_PRELIMINARY.csv"
)

monthly_table.to_csv(monthly_path, index=False)


# ============================================================
# D. Date-field diagnostics for 2016 Current loans
# ============================================================

date_rows = []

for (field, year), n in date_counts.items():

    date_rows.append({
        "date_field": field,
        "recorded_year": (
            "Missing_or_unparsed"
            if year == -1
            else str(year)
        ),
        "n": int(n)
    })

date_table = pd.DataFrame(date_rows)

date_table = date_table.sort_values(
    ["date_field", "recorded_year"]
)

date_path = (
    TABLES /
    "09_2016_current_date_diagnostics_PRELIMINARY.csv"
)

date_table.to_csv(date_path, index=False)


# ============================================================
# E. Audit report
# ============================================================

report_path = (
    AUDIT /
    "09_status_vintage_audit_report.txt"
)

report = [
    "ROUND 2 - REVIEWER 2 COMMENT 1",
    "2016 loan-status vintage audit",
    "",
    f"Total dataset records scanned: {total_scanned:,}",
    f"Selected 2012-2016 36-month loans: {total_selected:,}",
    "",
    "IMPORTANT:",
    "Recorded status is not necessarily an as-of-April-2020 status.",
    "Date fields are diagnostics, not proof of status history.",
    "",
    "STATUS BY YEAR:",
    status_table.to_string(index=False),
    "",
    "2016 MONTHLY CURRENT SHARE:",
    monthly_table.to_string(index=False),
    "",
    "2016 CURRENT LOAN DATE DIAGNOSTICS:",
    date_table.to_string(index=False),
    "",
    "STATUS/LABEL CROSSCHECK:",
    label_table.to_string(index=False)
]

report_path.write_text(
    "\n".join(report),
    encoding="utf-8"
)


# ============================================================
# Display results
# ============================================================

print("\n" + "=" * 70)
print("STATUS BY ORIGINATION YEAR")
print("=" * 70)

print(status_table.to_string(index=False))

print("\n" + "=" * 70)
print("2016 MONTHLY CURRENT SHARE")
print("=" * 70)

print(monthly_table.to_string(index=False))

print("\n" + "=" * 70)
print("2016 CURRENT LOAN DATE DIAGNOSTICS")
print("=" * 70)

print(date_table.to_string(index=False))

print("\n" + "=" * 70)
print("FILES SAVED")
print("=" * 70)

print(status_path)
print(label_path)
print(monthly_path)
print(date_path)
print(report_path)

print("\nSTEP 9 COMPLETE")