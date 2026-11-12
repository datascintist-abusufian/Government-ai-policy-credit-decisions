from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd

ROOT = Path(r"E:\Governance-Policy\Cogent Economics & Finance")

DATA = ROOT / "P2P_Macro_Data.dta"

AUDIT = ROOT / "Round2_Audit"

TABLES = ROOT / "Round2_Manuscript_Results" / "01_Tables"

CUTOFF = pd.Timestamp("2020-04-01")

CHUNK_SIZE = 200000

RANDOM_SEED = 42

EXPECTED_TRAIN_N = 1304058
EXPECTED_TERMINAL_N = 874826
EXPECTED_CURRENT_2014_2016 = 311541

COLUMNS = [
    "issue_d",
    "loan_status",
    "term",
    "badloan"
]

if not DATA.exists():
    raise FileNotFoundError(DATA)

print("=" * 72)
print("ROUND 2 - STEP 13")
print("EXACT TRAINING-SAMPLE COMPOSITION AUDIT")
print("=" * 72)

print("Dataset:", DATA)
print("Random seed:", RANDOM_SEED)
print("Chunk size:", CHUNK_SIZE)

# ------------------------------------------------------------
# 1. Reconstruct original training eligibility
# ------------------------------------------------------------

flag_names = [
    "terminal",
    "nonterminal",
    "current",
    "current_2014",
    "current_2016",
    "current_2014_or_2016",
    "current_2014_or_2016_badloan0",
    "badloan_0",
    "badloan_1",
    "missing_badloan"
]

flag_parts = []

total_scanned = 0
total_selected = 0

with pd.read_stata(
    DATA,
    columns=COLUMNS,
    convert_categoricals=False,
    chunksize=CHUNK_SIZE
) as reader:

    for chunk_number, df in enumerate(reader, 1):

        total_scanned += len(df)

        # Match Notebook Cell 27 origination parsing
        issue = pd.to_datetime(
            df["issue_d"]
            .astype(str)
            .str.strip(),
            format="%b-%y",
            errors="coerce"
        )

        # Match Notebook Cell 27 term parsing
        term = pd.to_numeric(
            df["term"]
            .astype(str)
            .str.extract(r"(\d+)")[0],
            errors="coerce"
        )

        status = (
            df["loan_status"]
            .astype(str)
            .str.strip()
        )

        bad = pd.to_numeric(
            df["badloan"],
            errors="coerce"
        )

        # Exact training chronology
        is_train = (
            (issue >= pd.Timestamp("2012-08-01"))
            &
            (issue <= pd.Timestamp("2018-06-01"))
        )

        # Exact reconstructed maturity rule
        terminal = status.isin([
            "Fully Paid",
            "Charged Off",
            "Default"
        ])

        followup_months = (
            (CUTOFF.year - issue.dt.year) * 12
            +
            (CUTOFF.month - issue.dt.month)
        )

        contractual = followup_months >= term

        mature = terminal | contractual

        # Training observations in original file order
        selected_mask = (is_train & mature).to_numpy()

        if not selected_mask.any():
            print(
                f"Chunk {chunk_number:02d} | "
                f"Scanned {total_scanned:,} rows",
                flush=True
            )
            continue

        selected_status = status[selected_mask]

        selected_issue = issue[selected_mask]

        selected_terminal = terminal[selected_mask]

        selected_bad = bad[selected_mask]

        selected_current = selected_status.eq("Current")

        selected_2014 = selected_issue.dt.year.eq(2014)

        selected_2016 = selected_issue.dt.year.eq(2016)

        selected_term36 = term[selected_mask].eq(36)

        current_2014 = (
            selected_current
            &
            selected_2014
            &
            selected_term36
        )

        current_2016 = (
            selected_current
            &
            selected_2016
            &
            selected_term36
        )

        current_combined = current_2014 | current_2016

        # Compact Boolean matrix: one row per training loan
        flags = np.column_stack([
            selected_terminal.to_numpy(),
            (~selected_terminal).to_numpy(),
            selected_current.to_numpy(),
            current_2014.to_numpy(),
            current_2016.to_numpy(),
            current_combined.to_numpy(),
            (
                current_combined
                &
                selected_bad.eq(0)
            ).to_numpy(),
            selected_bad.eq(0).to_numpy(),
            selected_bad.eq(1).to_numpy(),
            selected_bad.isna().to_numpy()
        ]).astype(np.uint8)

        flag_parts.append(flags)

        total_selected += len(flags)

        print(
            f"Chunk {chunk_number:02d} | "
            f"Scanned {total_scanned:,} | "
            f"Training mature {total_selected:,}",
            flush=True
        )

# ------------------------------------------------------------
# 2. Preserve original training row order
# ------------------------------------------------------------

if not flag_parts:
    raise RuntimeError("No mature training observations found.")

all_flags = np.concatenate(flag_parts, axis=0)

del flag_parts

n_full = len(all_flags)

print("\nReconstructed mature training size:", f"{n_full:,}")

if n_full != EXPECTED_TRAIN_N:
    raise RuntimeError(
        f"Training cohort mismatch: "
        f"observed={n_full:,}, "
        f"expected={EXPECTED_TRAIN_N:,}. "
        "Sampling audit stopped."
    )

# ------------------------------------------------------------
# 3. Reproduce exact NumPy sampling order
# ------------------------------------------------------------

rng = np.random.default_rng(RANDOM_SEED)

permutation = rng.permutation(n_full)

training_sizes = [
    180000,
    360000,
    540000,
    n_full
]

rows = []

for n_train in training_sizes:

    sample_idx = permutation[:n_train]

    subset = all_flags[sample_idx]

    observed = subset.sum(axis=0, dtype=np.int64)

    row = {
        "training_n": int(n_train),
        "sampling_seed": RANDOM_SEED
    }

    for column_name, count in zip(flag_names, observed):

        row[column_name + "_n"] = int(count)

        row[column_name + "_pct"] = round(
            100 * count / n_train,
            4
        )

    row["total_flagged_current_2014_2016"] = int(
        observed[5]
    )

    row["all_badloan_values_accounted_for"] = (
        int(observed[7] + observed[8] + observed[9])
        == n_train
    )

    rows.append(row)

results = pd.DataFrame(rows)

# ------------------------------------------------------------
# 4. Cross-check previously established counts
# ------------------------------------------------------------

full = results.loc[
    results["training_n"] == n_full
].iloc[0]

checks = {
    "full_training_size": (
        n_full == EXPECTED_TRAIN_N
    ),
    "full_terminal_count": (
        int(full["terminal_n"])
        == EXPECTED_TERMINAL_N
    ),
    "full_current_2014_2016_count": (
        int(full["current_2014_or_2016_n"])
        == EXPECTED_CURRENT_2014_2016
    ),
    "current_2014_2016_all_badloan0": (
        int(full["current_2014_or_2016_n"])
        ==
        int(
            full["current_2014_or_2016_badloan0_n"]
        )
    )
}

checks_passed = all(checks.values())

# ------------------------------------------------------------
# 5. Save detailed CSV
# ------------------------------------------------------------

csv_path = (
    TABLES /
    "13_exact_training_sample_composition_PRELIMINARY.csv"
)

results.to_csv(
    csv_path,
    index=False
)

# ------------------------------------------------------------
# 6. Save readable audit report
# ------------------------------------------------------------

report_path = (
    AUDIT /
    "13_model_input_composition_audit.txt"
)

report_lines = [
    "ROUND 2 - REVIEWER 2 COMMENT 1",
    "Exact modelling sample composition",
    "",
    f"Generated: {datetime.now()}",
    f"Dataset: {DATA}",
    f"Rows scanned: {total_scanned:,}",
    f"Training observations: {n_full:,}",
    f"Random seed: {RANDOM_SEED}",
    f"NumPy version: {np.__version__}",
    f"Pandas version: {pd.__version__}",
    "",
    "RECONSTRUCTION:",
    "1. Original training chronological period",
    "2. Original outcome-maturity eligibility rule",
    "3. Original dataset row order",
    "4. NumPy random generator with seed 42",
    "5. Nested subsets: 180k, 360k, 540k, full",
    "",
    "CHECKS:",
]

for name, passed in checks.items():
    report_lines.append(
        f"{name}: {'PASS' if passed else 'FAIL'}"
    )

report_lines.extend([
    "",
    "SAMPLE COMPOSITION:",
    results.to_string(index=False),
    "",
    "IMPORTANT INTERPRETATION:",
    "These counts identify the composition of the",
    "deterministically reconstructed training samples.",
    "",
    "Current status does not prove an incorrect label.",
    "It indicates potentially incomplete outcome follow-up.",
    "",
    "The full training sample includes every eligible",
    "mature observation.",
    "",
    "All three classifiers use the same sampling indices",
    "at a given training size.",
    "",
    "Preprocessing in the original notebook was fitted",
    "on the full mature training cohort.",
    "",
    "This analysis does not retrain any classifier.",
    ""
])

report_path.write_text(
    "\n".join(report_lines),
    encoding="utf-8"
)

# ------------------------------------------------------------
# 7. Display concise output
# ------------------------------------------------------------

display_columns = [
    "training_n",
    "terminal_n",
    "nonterminal_n",
    "current_n",
    "current_2014_n",
    "current_2016_n",
    "current_2014_or_2016_n",
    "current_2014_or_2016_pct",
    "missing_badloan_n"
]

print("\n" + "=" * 72)
print("EXACT TRAINING SAMPLE COMPOSITION")
print("=" * 72)

print(
    results[display_columns].to_string(index=False)
)

print("\n" + "=" * 72)
print("VERIFICATION")
print("=" * 72)

for name, passed in checks.items():
    print(
        name,
        "PASS" if passed else "FAIL"
    )

print("\nOutputs saved:")
print(csv_path)
print(report_path)

if not checks_passed:
    raise RuntimeError(
        "One or more expected counts did not reconcile. "
        "Inspect the saved audit report."
    )

print("\nSTEP 13 COMPLETED SUCCESSFULLY")