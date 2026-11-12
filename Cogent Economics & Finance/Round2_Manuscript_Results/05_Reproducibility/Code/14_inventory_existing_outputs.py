from pathlib import Path
from collections import Counter
from datetime import datetime
import csv
import os
import re

ROOT = Path(r"E:\Governance-Policy\Cogent Economics & Finance")

RESULTS = ROOT / "Round2_Manuscript_Results"

REPRO = RESULTS / "05_Reproducibility"

CODE = REPRO / "Code"

REPRO.mkdir(parents=True, exist_ok=True)
CODE.mkdir(parents=True, exist_ok=True)

SOURCES = {
    "Round1_Results": ROOT / "revision_outputs",
    "Round2_Audit": ROOT / "Round2_Audit",
    "Round2_Results": RESULTS
}

SKIP_DIRECTORIES = {
    "__pycache__",
    ".git",
    ".ipynb_checkpoints",
    ".venv",
    "venv",
    "node_modules"
}

REVIEW_TOPICS = {
    "01_Outcome_Maturity": [
        "cohort", "matur", "terminal", "followup",
        "censor", "snapshot", "outcome",
        "selection", "cell26"
    ],
    "02_Review_All": [
        "review_all", "reviewall", "r5",
        "review_capacity", "capacity"
    ],
    "03_Multiple_Seeds": [
        "seed", "stability", "training_size",
        "training_size_sensitivity", "cell27"
    ],
    "04_Data_Leakage": [
        "leakage", "leak", "preprocess",
        "temporal", "feature_audit"
    ],
    "05_Origination_Features": [
        "origination", "predictor", "feature",
        "availability", "cell24", "cell25"
    ],
    "06_Brier_Baseline": [
        "brier", "baseline", "null_model"
    ],
    "07_Cluster_Uncertainty": [
        "cluster", "bootstrap", "confidence",
        "uncertainty", "cell28"
    ],
    "08_Subgroups": [
        "subgroup", "fairness", "disparity",
        "group_analysis"
    ],
    "09_Governance_Rules": [
        "governance", "policy", "decision_rule",
        "threshold", "cell31"
    ],
    "10_Model_Comparison": [
        "comparison", "model_performance",
        "auc", "training_size", "cell28"
    ],
    "11_Reproducibility": [
        "reproducib", "environment",
        "requirements", "manifest"
    ],
    "12_Target_Definition": [
        "target", "badloan", "default",
        "label", "status"
    ],
    "13_Competing_Risks": [
        "competing", "survival", "hazard",
        "cumulative_incidence"
    ],
    "14_Calibration": [
        "calibration", "platt", "isotonic",
        "reliability", "cell30"
    ],
    "15_Review_Costs": [
        "cost", "utility", "review",
        "economic"
    ],
    "16_Regulatory": [
        "regulatory", "regulation",
        "compliance"
    ]
}

ALLOWED_EXTENSIONS = {
    ".csv", ".tsv", ".txt", ".json",
    ".npy", ".npz", ".pkl", ".joblib",
    ".png", ".jpg", ".jpeg", ".svg",
    ".pdf", ".xlsx", ".xls",
    ".html", ".md", ".py", ".ps1",
    ".parquet", ".feather", ".ipynb"
}

print("=" * 72)
print("ROUND 2 - STEP 14")
print("EXISTING RESEARCH OUTPUT INVENTORY")
print("=" * 72)

inventory = []
evidence = []
missing_sources = []

for source_name, source_path in SOURCES.items():

    if not source_path.exists():

        print(f"WARNING: Missing source folder: {source_path}")
        missing_sources.append(str(source_path))
        continue

    print(f"\nScanning: {source_name}")

    source_count = 0

    for current_dir, dirs, files in os.walk(source_path):

        dirs[:] = [
            d for d in dirs
            if d not in SKIP_DIRECTORIES
        ]

        for filename in files:

            path = Path(current_dir) / filename

            if path.suffix.lower() not in ALLOWED_EXTENSIONS:
                continue

            try:
                metadata = path.stat()
            except OSError:
                continue

            relative = str(path.relative_to(ROOT))

            record = {
                "source": source_name,
                "relative_path": relative,
                "filename": filename,
                "extension": path.suffix.lower(),
                "size_bytes": metadata.st_size,
                "size_mb": round(
                    metadata.st_size / (1024 ** 2), 4
                ),
                "modified": datetime.fromtimestamp(
                    metadata.st_mtime
                ).isoformat(timespec="seconds")
            }

            inventory.append(record)

            source_count += 1

            searchable_name = (
                filename + " " + str(path.parent.name)
            ).lower()

            for topic, keywords in REVIEW_TOPICS.items():

                matched = [
                    keyword
                    for keyword in keywords
                    if keyword in searchable_name
                ]

                if matched:

                    evidence.append({
                        "review_topic": topic,
                        "source": source_name,
                        "filename": filename,
                        "relative_path": relative,
                        "matched_keywords": "; ".join(matched),
                        "verification_status":
                            "Candidate only - inspect before reuse"
                    })

    print(f"Files identified: {source_count:,}")

# ============================================================
# A. Save complete inventory
# ============================================================

inventory_path = (
    REPRO / "14_existing_output_inventory.csv"
)

inventory_columns = [
    "source",
    "relative_path",
    "filename",
    "extension",
    "size_bytes",
    "size_mb",
    "modified"
]

with inventory_path.open(
    "w",
    newline="",
    encoding="utf-8-sig"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=inventory_columns
    )

    writer.writeheader()
    writer.writerows(inventory)

# ============================================================
# B. Save reviewer evidence candidates
# ============================================================

evidence_path = (
    REPRO / "14_reviewer2_evidence_candidates.csv"
)

evidence_columns = [
    "review_topic",
    "source",
    "filename",
    "relative_path",
    "matched_keywords",
    "verification_status"
]

with evidence_path.open(
    "w",
    newline="",
    encoding="utf-8-sig"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=evidence_columns
    )

    writer.writeheader()
    writer.writerows(evidence)

# ============================================================
# C. Summarise available evidence
# ============================================================

topic_counts = Counter(
    row["review_topic"]
    for row in evidence
)

source_counts = Counter(
    row["source"]
    for row in inventory
)

extension_counts = Counter(
    row["extension"]
    for row in inventory
)

report_lines = [
    "ROUND 2 - STEP 14",
    "Existing statistical output inventory",
    "",
    f"Generated: {datetime.now()}",
    f"Project: {ROOT}",
    "",
    f"Total files inventoried: {len(inventory):,}",
    f"Potential reviewer evidence matches: {len(evidence):,}",
    "",
    "IMPORTANT:",
    "This inventory checks filenames and metadata only.",
    "A matching filename does NOT prove that an analysis",
    "is correct, complete or suitable for publication.",
    "",
    "=" * 65,
    "FILES BY SOURCE",
    "=" * 65
]

for source_name in SOURCES:

    report_lines.append(
        f"{source_name}: {source_counts[source_name]:,}"
    )

report_lines.extend([
    "",
    "=" * 65,
    "FILES BY TYPE",
    "=" * 65
])

for extension, count in sorted(extension_counts.items()):

    report_lines.append(
        f"{extension}: {count:,}"
    )

report_lines.extend([
    "",
    "=" * 65,
    "POTENTIAL EVIDENCE BY REVIEW TOPIC",
    "=" * 65
])

for topic in REVIEW_TOPICS:

    report_lines.append(
        f"\n{topic}: {topic_counts[topic]} candidate matches"
    )

    examples = [
        row["relative_path"]
        for row in evidence
        if row["review_topic"] == topic
    ]

    for example in examples[:8]:

        report_lines.append(
            "  " + example
        )

    if len(examples) > 8:

        report_lines.append(
            f"  ... and {len(examples) - 8} additional matches"
        )

if missing_sources:

    report_lines.extend([
        "",
        "MISSING SOURCE FOLDERS:",
        *missing_sources
    ])

report_lines.extend([
    "",
    "=" * 65,
    "NEXT ACTION",
    "=" * 65,
    "Inspect identified outputs before deciding",
    "which experiments must be rerun.",
    "",
    "Do not treat filename matches as validated results."
])

report_path = (
    REPRO / "14_existing_outputs_report.txt"
)

report_path.write_text(
    "\n".join(report_lines),
    encoding="utf-8"
)

# ============================================================
# D. Display audit summary
# ============================================================

print("\n" + "=" * 72)
print("REVIEWER 2 - POTENTIAL EXISTING EVIDENCE")
print("=" * 72)

for topic in REVIEW_TOPICS:

    print(
        f"{topic:32s} "
        f"{topic_counts[topic]:5d} candidate files"
    )

print("\n" + "=" * 72)
print("OUTPUTS SAVED")
print("=" * 72)

print(inventory_path)
print(evidence_path)
print(report_path)

print("\nIMPORTANT: No existing files were modified.")

print("\nSTEP 14 COMPLETED SUCCESSFULLY")