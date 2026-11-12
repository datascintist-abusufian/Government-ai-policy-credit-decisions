import argparse
import csv
import json
import sys
import zipfile

from pathlib import Path, PurePosixPath
from collections import Counter
from importlib.metadata import version, PackageNotFoundError

import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument("--root", required=True)
args = parser.parse_args()

root = Path(args.root)

repro = (
    root
    / "Round2_Manuscript_Results"
    / "05_Reproducibility"
)

evidence = repro / "Evidence_Checks"
tables = root / "Round2_Manuscript_Results" / "01_Tables"
revision = root / "revision_outputs"

archive = (
    evidence
    / "44_Cogent_replication_candidate_NOT_VERIFIED.zip"
)

evidence.mkdir(parents=True, exist_ok=True)

print("=" * 72)
print("STEP 60: REPRODUCIBILITY PACKAGE READINESS AUDIT")
print("=" * 72)

# ------------------------------------------------------------
# 1. Check the essential local files
# ------------------------------------------------------------

required = {
    "Original Stata dataset": root / "P2P_Macro_Data.dta",
    "Main modelling notebook": root / "Economic and Finance.ipynb",
    "Original analysis notebook": root / "Fintech_5_11_2026.ipynb",
    "Stata feature construction": root / "P2P_Macro_Codes.do",
    "Stata label construction": root / "Labels.do",
    "Original feature manifest":
        revision / "cell26_locked_feature_specification.csv",
    "Original training sensitivity":
        revision / "cell27_training_size_sensitivity.csv",
    "Original model comparisons":
        revision / "cell28_test_performance_point_estimates.csv",
    "Original calibration results":
        revision / "cell29_before_after_calibration.csv",
    "Original validation labels":
        revision / "cell27_y_val.npy",
    "Original test labels":
        revision / "cell27_y_test.npy",
    "145-feature sensitivity":
        tables / "57_145_feature_quarantine_sensitivity.csv",
    "Outcome-selection summary":
        tables / "58_outcome_selection_summary.csv",
    "Population balance":
        tables / "59_selected_vs_excluded_numeric_balance.csv",
    "Corrected dependency register":
        evidence /
        "56_corrected_payment_history_dependency_register.csv",
}

print("\n===== LOCAL EVIDENCE CHECK =====")

records = []

for label, path in required.items():

    exists = path.is_file()

    print(
        f"{'FOUND' if exists else 'MISSING'}: {label}"
    )

    records.append({
        "artifact": label,
        "local_exists": exists,
        "local_path": str(path),
        "archive_contains_matching_filename": False
    })

missing_local = [
    record["artifact"]
    for record in records
    if not record["local_exists"]
]

# ------------------------------------------------------------
# 2. Verify original prediction-array dimensions
# ------------------------------------------------------------

print("\n===== ORIGINAL PREDICTION ARRAYS =====")

array_checks = {
    "cell27_y_val.npy": 27374,
    "cell27_y_test.npy": 41145
}

array_results = {}

for name, expected in array_checks.items():

    path = revision / name

    if not path.is_file():

        print("MISSING:", name)

        array_results[name] = {
            "status": "MISSING"
        }

        continue

    data = np.load(
        path,
        mmap_mode="r",
        allow_pickle=False
    )

    observed = int(data.shape[0])

    passed = (
        data.ndim == 1
        and observed == expected
    )

    print(
        f"{name}: shape={data.shape}; "
        f"expected={expected}; "
        f"{'PASS' if passed else 'FAIL'}"
    )

    array_results[name] = {
        "status": "PASS" if passed else "FAIL",
        "shape": list(data.shape),
        "expected_rows": expected
    }

# ------------------------------------------------------------
# 3. Check candidate ZIP integrity
# ------------------------------------------------------------

print("\n===== REPLICATION ARCHIVE =====")

archive_summary = {
    "exists": archive.is_file(),
    "integrity": "NOT_CHECKED",
    "file_count": 0,
    "python_file_count": 0,
    "stata_file_count": 0,
    "data_file_count": 0,
    "python_compile_failures": [],
    "unsafe_archive_names": [],
}

archive_names = []

if archive.is_file():

    print("FOUND:", archive)

    with zipfile.ZipFile(archive, "r") as z:

        bad_member = z.testzip()

        archive_summary["integrity"] = (
            "PASS" if bad_member is None else "FAIL"
        )

        members = [
            info
            for info in z.infolist()
            if not info.is_dir()
        ]

        archive_names = [
            info.filename.replace("\\", "/")
            for info in members
        ]

        archive_summary["file_count"] = len(members)

        suffix_counts = Counter(
            PurePosixPath(name).suffix.lower()
            for name in archive_names
        )

        archive_summary["python_file_count"] = (
            suffix_counts[".py"]
        )

        archive_summary["stata_file_count"] = (
            suffix_counts[".do"]
        )

        archive_summary["data_file_count"] = (
            suffix_counts[".dta"]
        )

        # Check archive path safety without extracting files.

        for name in archive_names:

            path = PurePosixPath(name)

            if (
                path.is_absolute()
                or ".." in path.parts
                or (
                    len(name) >= 2
                    and name[1] == ":"
                )
            ):

                archive_summary[
                    "unsafe_archive_names"
                ].append(name)

        # Compile source code without running it.

        for name in archive_names:

            if not name.lower().endswith(".py"):
                continue

            try:

                source = z.read(name)

                compile(
                    source,
                    name,
                    "exec"
                )

            except Exception as exc:

                archive_summary[
                    "python_compile_failures"
                ].append({
                    "file": name,
                    "error": str(exc)
                })

        archived_basenames = {
            PurePosixPath(name).name
            for name in archive_names
        }

        for record in records:

            basename = Path(
                record["local_path"]
            ).name

            record[
                "archive_contains_matching_filename"
            ] = (
                basename in archived_basenames
            )

    print(
        "ZIP integrity:",
        archive_summary["integrity"]
    )

    print(
        "Archived files:",
        archive_summary["file_count"]
    )

    print(
        "Python scripts:",
        archive_summary["python_file_count"]
    )

    print(
        "Stata scripts:",
        archive_summary["stata_file_count"]
    )

    print(
        "Stata datasets:",
        archive_summary["data_file_count"]
    )

    print(
        "Python compilation failures:",
        len(
            archive_summary[
                "python_compile_failures"
            ]
        )
    )

    print(
        "Unsafe archive paths:",
        len(
            archive_summary[
                "unsafe_archive_names"
            ]
        )
    )

    print("\n===== ARCHIVE CONTENT PREVIEW =====")

    for name in archive_names[:35]:
        print(name)

else:

    print(
        "MISSING: Candidate replication ZIP."
    )

# ------------------------------------------------------------
# 4. Record current software environment
# ------------------------------------------------------------

print("\n===== SOFTWARE ENVIRONMENT =====")

environment = {
    "python": sys.version.split()[0]
}

for package in [
    "numpy",
    "pandas",
    "scipy",
    "scikit-learn",
    "statsmodels",
    "matplotlib"
]:

    try:
        environment[package] = version(package)

    except PackageNotFoundError:
        environment[package] = "NOT_INSTALLED"

for name, value in environment.items():
    print(f"{name}: {value}")

# ------------------------------------------------------------
# 5. Save machine-readable audit
# ------------------------------------------------------------

audit = {
    "purpose": "Replication readiness preflight",
    "clean_replay_executed": False,
    "archive": str(archive),
    "archive_summary": archive_summary,
    "local_missing_files": missing_local,
    "prediction_array_checks": array_results,
    "software_environment": environment,
    "interpretation": (
        "ZIP integrity, source compilation and file "
        "presence do not establish that the original "
        "experiments can be independently reproduced."
    )
}

json_path = (
    evidence
    / "60_replication_readiness_audit.json"
)

csv_path = (
    evidence
    / "60_replication_file_inventory.csv"
)

with open(
    json_path,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        audit,
        f,
        indent=2
    )

with open(
    csv_path,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "artifact",
            "local_exists",
            "local_path",
            "archive_contains_matching_filename"
        ]
    )

    writer.writeheader()
    writer.writerows(records)

# ------------------------------------------------------------
# 6. Summary
# ------------------------------------------------------------

print("\n===== STEP 60 SUMMARY =====")

print(
    "Missing local artifacts:",
    len(missing_local)
)

for item in missing_local:
    print(" -", item)

print(
    "Archive integrity:",
    archive_summary["integrity"]
)

print(
    "Archive compilation failures:",
    len(
        archive_summary[
            "python_compile_failures"
        ]
    )
)

for failure in archive_summary[
    "python_compile_failures"
][:10]:

    print(
        " -",
        failure["file"],
        failure["error"]
    )

print("\nSaved:", json_path)
print("Saved:", csv_path)

print(
    "\nIMPORTANT: A successful audit does not "
    "constitute a clean independent replay."
)

print("\nSTEP 60 COMPLETE")
print("Original experiments unchanged.")
