from pathlib import Path
from collections import Counter
from datetime import datetime
import pandas as pd
import numpy as np
import sys
import traceback

ROOT = Path(r"E:\Governance-Policy\Cogent Economics & Finance")

DATA = ROOT / "P2P_Macro_Data.dta"

AUDIT = ROOT / "Round2_Audit"

TABLES = ROOT / "Round2_Manuscript_Results" / "01_Tables"

STATISTICS = ROOT / "Round2_Manuscript_Results" / "03_Statistical_Results"

for folder in [AUDIT, TABLES, STATISTICS]:
    folder.mkdir(parents=True, exist_ok=True)

CUTOFF = pd.Timestamp("2020-04-01")

TERMINAL = {
    "Fully Paid",
    "Charged Off",
    "Default"
}

EXPECTED = {
    "Validation": {
        "total": 370994,
        "mature": 27374
    },
    "Test": {
        "total": 401169,
        "mature": 41145
    }
}

COLUMNS = [
    "issue_date",
    "loan_status",
    "last_pymnt_d",
    "term",
    "badloan"
]

CHUNK_SIZE = 150000


def parse_payment_month(series):

    raw = (
        series.fillna("")
        .astype(str)
        .str.strip()
    )

    result = pd.Series(
        pd.NaT,
        index=series.index,
        dtype="datetime64[ns]"
    )

    formats = [
        (r"^[A-Za-z]{3}-\d{2}$", "%b-%y"),
        (r"^[A-Za-z]{3}-\d{4}$", "%b-%Y"),
        (r"^\d{4}-\d{2}$", "%Y-%m"),
        (r"^\d{4}-\d{2}-\d{2}$", "%Y-%m-%d")
    ]

    for pattern, date_format in formats:

        mask = raw.str.match(pattern, na=False)

        if mask.any():

            result.loc[mask] = pd.to_datetime(
                raw.loc[mask],
                format=date_format,
                errors="coerce"
            )

    return result


def assign_split(issue):

    split = np.full(
        len(issue),
        "Outside",
        dtype=object
    )

    split[
        (issue >= pd.Timestamp("2012-08-01"))
        &
        (issue < pd.Timestamp("2018-07-01"))
    ] = "Train"

    split[
        (issue >= pd.Timestamp("2018-07-01"))
        &
        (issue < pd.Timestamp("2019-04-01"))
    ] = "Validation"

    split[
        (issue >= pd.Timestamp("2019-04-01"))
        &
        (issue < pd.Timestamp("2020-01-01"))
    ] = "Test"

    return split


def main():

    if not DATA.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA}"
        )

    print("=" * 65)
    print("ROUND 2 - STEP 5")
    print("TERMINAL-STATUS TIMING AUDIT")
    print("=" * 65)

    print("Dataset:", DATA)
    print("Dataset size GB:", round(
        DATA.stat().st_size / 1024**3, 3
    ))

    print("Administrative cutoff:", CUTOFF.date())
    print("Chunk size:", CHUNK_SIZE)

    counts = Counter()
    total_source_rows = 0

    print("\nReading original dataset in chunks...\n")

    with pd.read_stata(
        DATA,
        columns=COLUMNS,
        convert_categoricals=False,
        chunksize=CHUNK_SIZE
    ) as reader:

        for chunk_number, df in enumerate(reader, 1):

            total_source_rows += len(df)

            issue = pd.to_datetime(
                df["issue_date"],
                errors="coerce"
            )

            split = assign_split(issue)

            status = (
                df["loan_status"]
                .fillna("")
                .astype(str)
                .str.strip()
            )

            last_payment = parse_payment_month(
                df["last_pymnt_d"]
            )

            adverse = pd.to_numeric(
                df["badloan"],
                errors="coerce"
            )

            # All chronological observations
            for group in ["Train", "Validation", "Test"]:

                group_mask = split == group

                counts[(group, "all")] += int(
                    group_mask.sum()
                )

            terminal_mask = status.isin(TERMINAL)

            valid = (
                np.isin(
                    split,
                    ["Train", "Validation", "Test"]
                )
                &
                terminal_mask.to_numpy()
            )

            if valid.any():

                work = pd.DataFrame({
                    "split": split[valid],
                    "loan_status": status[valid].to_numpy(),
                    "last_payment": last_payment[valid].to_numpy(),
                    "adverse": adverse[valid].to_numpy()
                })

                before = (
                    work["last_payment"] < CUTOFF
                )

                during = (
                    work["last_payment"].dt.to_period("M")
                    ==
                    CUTOFF.to_period("M")
                )

                after = (
                    work["last_payment"]
                    >= pd.Timestamp("2020-05-01")
                )

                work["timing"] = np.select(
                    [
                        before,
                        during,
                        after
                    ],
                    [
                        "Before_April_2020",
                        "During_April_2020",
                        "After_April_2020"
                    ],
                    default="Missing_or_Unparsed"
                )

                grouped = (
                    work.groupby(
                        ["split", "loan_status", "timing"]
                    )
                    .size()
                )

                for key, n in grouped.items():
                    counts[
                        ("timing",) + tuple(key)
                    ] += int(n)

                for group in [
                    "Train",
                    "Validation",
                    "Test"
                ]:

                    group_mask = work["split"].eq(group)

                    counts[
                        (group, "terminal")
                    ] += int(group_mask.sum())

                    counts[
                        (group, "terminal_adverse")
                    ] += int(
                        (
                            group_mask
                            &
                            work["adverse"].eq(1)
                        ).sum()
                    )

            print(
                f"Chunk {chunk_number:02d} | "
                f"Processed {total_source_rows:,} rows",
                flush=True
            )

    # --------------------------------------------------------
    # Detailed terminal-status timing table
    # --------------------------------------------------------

    detail_rows = []

    for key, n in counts.items():

        if key[0] != "timing":
            continue

        _, group, status, timing = key

        detail_rows.append({
            "cohort": group,
            "loan_status": status,
            "last_payment_timing": timing,
            "n": n
        })

    details = pd.DataFrame(detail_rows)

    details = details.sort_values(
        [
            "cohort",
            "loan_status",
            "last_payment_timing"
        ]
    )

    detail_path = (
        TABLES /
        "05_terminal_payment_timing_by_status.csv"
    )

    details.to_csv(
        detail_path,
        index=False
    )

    # --------------------------------------------------------
    # Summary table
    # --------------------------------------------------------

    timing_categories = [
        "Before_April_2020",
        "During_April_2020",
        "After_April_2020",
        "Missing_or_Unparsed"
    ]

    summary_rows = []

    for group in ["Train", "Validation", "Test"]:

        total = counts[(group, "all")]

        terminal = counts[(group, "terminal")]

        adverse_n = counts[
            (group, "terminal_adverse")
        ]

        row = {
            "cohort": group,
            "total_loans": total,
            "terminal_status_loans": terminal,
            "terminal_share_pct": (
                100 * terminal / total
                if total else np.nan
            ),
            "adverse_terminal_n": adverse_n,
            "adverse_terminal_pct": (
                100 * adverse_n / terminal
                if terminal else np.nan
            )
        }

        for timing in timing_categories:

            row[timing] = int(
                details.loc[
                    (details["cohort"] == group)
                    &
                    (
                        details["last_payment_timing"]
                        == timing
                    ),
                    "n"
                ].sum()
            )

        row["timing_count_check"] = (
            sum(row[t] for t in timing_categories)
            == terminal
        )

        if group in EXPECTED:

            row["expected_total"] = (
                EXPECTED[group]["total"]
            )

            row["expected_mature"] = (
                EXPECTED[group]["mature"]
            )

            row["expected_counts_match"] = (
                total == EXPECTED[group]["total"]
                and
                terminal == EXPECTED[group]["mature"]
            )

        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)

    summary_path = (
        TABLES /
        "05_terminal_payment_timing_summary.csv"
    )

    summary.to_csv(
        summary_path,
        index=False
    )

    # --------------------------------------------------------
    # Save audit interpretation
    # --------------------------------------------------------

    report_path = (
        AUDIT /
        "05_terminal_date_audit_report.txt"
    )

    report_lines = [
        "ROUND 2 - REVIEWER 2 COMMENT 1",
        "Terminal-status timing audit",
        "",
        f"Date generated: {datetime.now()}",
        f"Dataset: {DATA}",
        f"Source records: {total_source_rows:,}",
        f"Cutoff: {CUTOFF.date()}",
        "",
        "IMPORTANT METHODOLOGICAL NOTE:",
        "last_pymnt_d records the last payment month.",
        "It is NOT a verified terminal-event date.",
        "",
        "A payment after April 2020 is inconsistent",
        "with an April 2020 observation cutoff.",
        "",
        "A payment before April 2020 does NOT prove",
        "that final loan status was already known",
        "at the administrative cutoff.",
        "",
        "The Default status is included here to",
        "reproduce the published maturity rule.",
        "Its classification as terminal requires",
        "separate substantive justification.",
        "",
        "COHORT SUMMARY:",
        summary.to_string(index=False),
        "",
        "DETAILED STATUS TABLE:",
        details.to_string(index=False),
        ""
    ]

    report_path.write_text(
        "\n".join(report_lines),
        encoding="utf-8"
    )

    print("\n" + "=" * 65)
    print("COHORT SUMMARY")
    print("=" * 65)

    print(summary.to_string(index=False))

    print("\n" + "=" * 65)
    print("OUTPUTS SAVED")
    print("=" * 65)

    print(detail_path)
    print(summary_path)
    print(report_path)

    print("\nSTEP 5 COMPLETED SUCCESSFULLY")


if __name__ == "__main__":

    try:
        main()

    except Exception:

        traceback.print_exc()
        sys.exit(1)