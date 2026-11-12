from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

ROOT = Path(r"E:\Governance-Policy\Cogent Economics & Finance")

OLD = ROOT / "revision_outputs"

REPRO = (
    ROOT
    / "Round2_Manuscript_Results"
    / "05_Reproducibility"
)

REPRO.mkdir(parents=True, exist_ok=True)

if not OLD.exists():
    raise FileNotFoundError(OLD)

print("=" * 72)
print("ROUND 2 - STEP 16")
print("EXISTING STATISTICAL RESULTS VERIFICATION")
print("=" * 72)

# ------------------------------------------------------------
# A. Priority statistical outputs
# ------------------------------------------------------------

PRIORITY = [
    "cell27_training_size_sensitivity.csv",
    "cell27_training_size_deltas.csv",
    "cell28_paired_bootstrap_model_comparisons.csv",
    "cell28_test_performance_point_estimates.csv",
    "cell29_before_after_calibration.csv",
    "cell29_test_calibration_change.csv",
    "cell30_recalibrated_threshold_analysis.csv",
    "cell30_recalibrated_test_operating_points.csv",
    "cell31A_490_threshold_audit.csv",
    "cell31A_geographic_stress_summary.csv",
    "cell31B_supported_slice_results.csv",
    "cell31B_unemployment_group_performance.csv",
    "cell31B_unemployment_threshold_performance.csv",
    "cell31B_high_unemployment_slice_support.csv",
    "cell19_overall_cif_horizons.csv",
    "cell21_overall_horizon_ci.csv",
    "cell22_bootstrap_cif_summary.csv",
    "cell24_suspicious_candidate_features.csv",
    "cell26_locked_feature_specification.csv"
]

# ------------------------------------------------------------
# B. Inventory actual CSV contents
# ------------------------------------------------------------

schema_rows = []
preview_cache = {}

csv_files = sorted(OLD.glob("*.csv"))

for path in csv_files:

    try:
        df = pd.read_csv(path)

        schema_rows.append({
            "filename": path.name,
            "rows": len(df),
            "columns": len(df.columns),
            "column_names": "; ".join(
                map(str, df.columns)
            ),
            "read_status": "PASS"
        })

        if path.name in PRIORITY:
            preview_cache[path.name] = df

    except Exception as exc:

        schema_rows.append({
            "filename": path.name,
            "rows": -1,
            "columns": -1,
            "column_names": "",
            "read_status": (
                f"ERROR: {type(exc).__name__}: {exc}"
            )
        })

schema_df = pd.DataFrame(schema_rows)

schema_path = (
    REPRO /
    "16_existing_statistical_table_schema.csv"
)

schema_df.to_csv(
    schema_path,
    index=False
)

# ------------------------------------------------------------
# C. Inspect saved NumPy arrays
# ------------------------------------------------------------

array_rows = []

EXPECTED_VAL = 27374
EXPECTED_TEST = 41145

for path in sorted(OLD.glob("*.npy")):

    try:
        array = np.load(
            path,
            mmap_mode="r",
            allow_pickle=False
        )

        n = int(array.shape[0])

        if "_val" in path.name:
            expected = EXPECTED_VAL

        elif "_test" in path.name:
            expected = EXPECTED_TEST

        else:
            expected = None

        if expected is None:
            check = "NOT_APPLICABLE"
        else:
            check = (
                "PASS" if n == expected else "FAIL"
            )

        array_rows.append({
            "filename": path.name,
            "shape": str(array.shape),
            "dtype": str(array.dtype),
            "observations": n,
            "expected_observations": expected,
            "dimension_check": check
        })

    except Exception as exc:

        array_rows.append({
            "filename": path.name,
            "shape": "",
            "dtype": "",
            "observations": -1,
            "expected_observations": None,
            "dimension_check": (
                f"ERROR: {type(exc).__name__}: {exc}"
            )
        })

array_df = pd.DataFrame(array_rows)

array_path = (
    REPRO /
    "16_saved_prediction_array_integrity.csv"
)

array_df.to_csv(
    array_path,
    index=False
)

# ------------------------------------------------------------
# D. Prepare detailed statistical report
# ------------------------------------------------------------

report_lines = [
    "ROUND 2 - STEP 16",
    "Existing statistical results inspection",
    "",
    f"Generated: {datetime.now()}",
    f"Source: {OLD}",
    "",
    f"CSV files inspected: {len(csv_files)}",
    f"NumPy arrays inspected: {len(array_rows)}",
    "",
    "IMPORTANT:",
    "Successful file reading and matching array lengths",
    "do not prove that the statistical analyses are valid.",
    "",
    "=" * 72,
    "PRIORITY STATISTICAL RESULTS",
    "=" * 72
]

for filename in PRIORITY:

    report_lines.extend([
        "",
        "-" * 72,
        filename,
        "-" * 72
    ])

    if filename not in preview_cache:

        report_lines.append(
            "MISSING OR UNREADABLE"
        )
        continue

    df = preview_cache[filename]

    report_lines.append(
        f"Rows: {len(df)}"
    )

    report_lines.append(
        f"Columns: {len(df.columns)}"
    )

    report_lines.append(
        "Column names:"
    )

    report_lines.append(
        ", ".join(map(str, df.columns))
    )

    report_lines.append("")

    report_lines.append(
        "First 15 records:"
    )

    with pd.option_context(
        "display.max_columns", None,
        "display.width", 300,
        "display.max_colwidth", 40
    ):
        report_lines.append(
            df.head(15).to_string(index=False)
        )

# ------------------------------------------------------------
# E. Array-integrity summary
# ------------------------------------------------------------

report_lines.extend([
    "",
    "=" * 72,
    "SAVED PREDICTION ARRAY INTEGRITY",
    "=" * 72,
    "",
    array_df.to_string(index=False),
    "",
    "=" * 72,
    "NEXT ACTION",
    "=" * 72,
    "",
    "Review statistical content against Reviewer 2.",
    "Identify missing or methodologically inadequate",
    "experiments before running additional models."
])

report_path = (
    REPRO /
    "16_existing_statistical_results_report.txt"
)

report_path.write_text(
    "\n".join(report_lines),
    encoding="utf-8"
)

# ------------------------------------------------------------
# F. Display concise results
# ------------------------------------------------------------

print("\nCSV files inspected:", len(csv_files))

print("Priority files requested:", len(PRIORITY))

print("Priority files readable:", len(preview_cache))

print("\nSaved prediction arrays:", len(array_df))

print("\nARRAY DIMENSION CHECKS:")

print(
    array_df["dimension_check"]
    .value_counts(dropna=False)
    .to_string()
)

print("\n" + "=" * 72)
print("OUTPUTS SAVED")
print("=" * 72)

print(schema_path)
print(array_path)
print(report_path)

print("\nSTEP 16 COMPLETED SUCCESSFULLY")