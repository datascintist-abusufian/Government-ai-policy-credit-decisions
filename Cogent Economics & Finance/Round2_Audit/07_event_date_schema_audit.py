from pathlib import Path
import pandas as pd
import re

ROOT = Path(r"E:\Governance-Policy\Cogent Economics & Finance")

DATA = ROOT / "P2P_Macro_Data.dta"

AUDIT = ROOT / "Round2_Audit"

TABLES = ROOT / "Round2_Manuscript_Results" / "01_Tables"

if not DATA.exists():
    raise FileNotFoundError(DATA)

print("=" * 70)
print("STEP 7: EVENT-DATE VARIABLE AUDIT")
print("=" * 70)

print("\nReading dataset metadata and one observation only...")

with pd.read_stata(
    DATA,
    convert_categoricals=False,
    iterator=True
) as reader:

    sample = reader.read(1)

    try:
        labels = reader.variable_labels()
    except Exception:
        labels = {}

print("\nDataset opened successfully.")

columns = list(sample.columns)

print("Total variables:", len(columns))

# ------------------------------------------------------------
# Identify potentially relevant variables
# ------------------------------------------------------------

pattern = re.compile(
    r"date|time|month|"
    r"issue|last_pymnt|next_pymnt|"
    r"payment|charge|default|"
    r"status|settle|hardship|"
    r"delinq|recover|matur|"
    r"event|observation|snapshot",
    flags=re.IGNORECASE
)

all_rows = []
candidate_rows = []

for column in columns:

    dtype = str(sample[column].dtype)

    label = labels.get(column, "") or ""

    record = {
        "column_name": column,
        "data_type": dtype,
        "variable_label": str(label)
    }

    all_rows.append(record)

    if pattern.search(column) or pattern.search(str(label)):
        candidate_rows.append(record)

all_df = pd.DataFrame(all_rows)

candidate_df = pd.DataFrame(candidate_rows)

# ------------------------------------------------------------
# Save complete variable inventory
# ------------------------------------------------------------

all_path = (
    TABLES /
    "07_complete_dataset_variable_inventory.csv"
)

all_df.to_csv(
    all_path,
    index=False
)

# ------------------------------------------------------------
# Save event-date candidate variables
# ------------------------------------------------------------

candidate_path = (
    TABLES /
    "07_event_date_candidate_variables.csv"
)

candidate_df.to_csv(
    candidate_path,
    index=False
)

# ------------------------------------------------------------
# Create readable report
# ------------------------------------------------------------

report_path = (
    AUDIT /
    "07_event_date_schema_report.txt"
)

report = [
    "ROUND 2 - REVIEWER 2 COMMENT 1",
    "Event-date variable availability audit",
    "",
    f"Dataset: {DATA}",
    f"Total variables: {len(columns)}",
    f"Candidate variables: {len(candidate_rows)}",
    "",
    "IMPORTANT:",
    "Column names alone do not establish whether",
    "a variable contains a genuine event date.",
    "",
    "CANDIDATE VARIABLES:",
    "",
    candidate_df.to_string(index=False)
]

report_path.write_text(
    "\n".join(report),
    encoding="utf-8"
)

# ------------------------------------------------------------
# Display findings
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("POTENTIAL EVENT-DATE VARIABLES")
print("=" * 70)

print(candidate_df.to_string(index=False))

print("\n" + "=" * 70)
print("OUTPUTS SAVED")
print("=" * 70)

print(all_path)
print(candidate_path)
print(report_path)

print("\nSTEP 7 COMPLETED SUCCESSFULLY")