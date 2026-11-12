from pathlib import Path
from collections import Counter
import pandas as pd
import numpy as np

ROOT = Path(r"E:\Governance-Policy\Cogent Economics & Finance")
DATA = ROOT / "P2P_Macro_Data.dta"

AUDIT = ROOT / "Round2_Audit"
TABLES = ROOT / "Round2_Manuscript_Results" / "01_Tables"

COLUMNS = [
    "issue_date",
    "term",
    "loan_status",
    "badloan",
    "last_pymnt_d",
    "next_pymnt_d",
    "last_credit_pull_d"
]

CHUNK_SIZE = 200000

date_fields = [
    "last_pymnt_d",
    "next_pymnt_d",
    "last_credit_pull_d"
]

if not DATA.exists():
    raise FileNotFoundError(DATA)


def parse_month(series):

    raw = series.fillna("").astype(str).str.strip()

    result = pd.to_datetime(
        raw,
        format="%b-%y",
        errors="coerce"
    )

    missing = result.isna() & raw.ne("")

    if missing.any():
        result.loc[missing] = pd.to_datetime(
            raw.loc[missing],
            format="%b-%Y",
            errors="coerce"
        )

    return result


print("=" * 70)
print("STEP 10: HISTORICAL SNAPSHOT CONSISTENCY")
print("=" * 70)

print("Reading selected variables only.")

date_counts = Counter()
current_counts = Counter()
bad0_counts = Counter()
bad1_counts = Counter()

total_scanned = 0

with pd.read_stata(
    DATA,
    columns=COLUMNS,
    convert_categoricals=False,
    chunksize=CHUNK_SIZE
) as reader:

    for chunk_number, df in enumerate(reader, 1):

        total_scanned += len(df)

        issue = pd.to_datetime(
            df["issue_date"],
            errors="coerce"
        )

        vintage = issue.dt.year

        term = pd.to_numeric(
            df["term"]
            .fillna("")
            .astype(str)
            .str.extract(r"(\d+)", expand=False),
            errors="coerce"
        )

        status = (
            df["loan_status"]
            .fillna("")
            .astype(str)
            .str.strip()
        )

        # Focus only on recorded Current loans
        # with 36-month contractual terms.

        mask = (
            vintage.between(2012, 2019)
            &
            term.eq(36)
            &
            status.eq("Current")
        )

        if not mask.any():
            print(
                f"Chunk {chunk_number:02d}: "
                f"{total_scanned:,} processed"
            )
            continue

        selected = df.loc[mask].copy()

        selected["vintage"] = vintage.loc[mask].astype(int)

        selected["badloan_numeric"] = pd.to_numeric(
            selected["badloan"],
            errors="coerce"
        )

        for year, group in selected.groupby("vintage"):

            year = int(year)

            current_counts[year] += len(group)

            bad0_counts[year] += int(
                group["badloan_numeric"].eq(0).sum()
            )

            bad1_counts[year] += int(
                group["badloan_numeric"].eq(1).sum()
            )

            for field in date_fields:

                dates = parse_month(group[field])

                date_years = (
                    dates.dt.year
                    .fillna(-1)
                    .astype(int)
                )

                frequencies = date_years.value_counts()

                for date_year, n in frequencies.items():

                    date_counts[
                        (year, field, int(date_year))
                    ] += int(n)

        print(
            f"Chunk {chunk_number:02d} | "
            f"Processed {total_scanned:,} rows",
            flush=True
        )


# ============================================================
# A. Full observation-year distribution
# ============================================================

rows = []

for (vintage, field, recorded_year), n in date_counts.items():

    rows.append({
        "origination_year": vintage,
        "date_field": field,
        "recorded_year": (
            "Missing_or_unparsed"
            if recorded_year == -1
            else str(recorded_year)
        ),
        "n": n
    })

distribution = pd.DataFrame(rows).sort_values(
    ["origination_year", "date_field", "recorded_year"]
)

distribution_path = (
    TABLES /
    "10_current_snapshot_year_distribution_PRELIMINARY.csv"
)

distribution.to_csv(
    distribution_path,
    index=False
)


# ============================================================
# B. Summary by origination year
# ============================================================

summary_rows = []

for year in sorted(current_counts):

    row = {
        "origination_year": year,
        "current_loans": current_counts[year],
        "badloan_0": bad0_counts[year],
        "badloan_1": bad1_counts[year]
    }

    for field in date_fields:

        options = [
            (date_year, n)
            for (vintage, date_field, date_year), n
            in date_counts.items()
            if vintage == year and date_field == field
        ]

        if options:

            most_common_year, most_common_n = max(
                options,
                key=lambda x: x[1]
            )

            row[field + "_modal_year"] = (
                "Missing"
                if most_common_year == -1
                else most_common_year
            )

            row[field + "_modal_share_pct"] = round(
                100 * most_common_n / current_counts[year],
                2
            )

    summary_rows.append(row)

summary = pd.DataFrame(summary_rows)

summary_path = (
    TABLES /
    "10_current_snapshot_summary_PRELIMINARY.csv"
)

summary.to_csv(
    summary_path,
    index=False
)


# ============================================================
# C. Readable audit report
# ============================================================

report_path = (
    AUDIT /
    "10_snapshot_consistency_report.txt"
)

report = [
    "ROUND 2 - REVIEWER 2 COMMENT 1",
    "Historical observation-date consistency",
    "",
    f"Dataset: {DATA}",
    f"Total source records scanned: {total_scanned:,}",
    "",
    "Scope: 36-month loans recorded as Current.",
    "",
    "IMPORTANT:",
    "Last-credit-pull and payment dates are",
    "proxies for recorded data recency.",
    "They are not verified historical",
    "loan-status snapshot dates.",
    "",
    "SUMMARY:",
    summary.to_string(index=False),
    "",
    "FULL DATE-YEAR DISTRIBUTION:",
    distribution.to_string(index=False)
]

report_path.write_text(
    "\n".join(report),
    encoding="utf-8"
)


# ============================================================
# D. Display results
# ============================================================

print("\n" + "=" * 70)
print("OBSERVATION-DATE SUMMARY")
print("=" * 70)

print(summary.to_string(index=False))

print("\n" + "=" * 70)
print("FILES SAVED")
print("=" * 70)

print(distribution_path)
print(summary_path)
print(report_path)

print("\nSTEP 10 COMPLETED SUCCESSFULLY")