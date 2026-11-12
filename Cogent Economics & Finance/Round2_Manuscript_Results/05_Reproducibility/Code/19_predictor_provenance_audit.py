from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

ROOT = Path(r"E:\Governance-Policy\Cogent Economics & Finance")

DATA = ROOT / "P2P_Macro_Data.dta"

OLD = ROOT / "revision_outputs"

RESULTS = ROOT / "Round2_Manuscript_Results"

TABLES = RESULTS / "01_Tables"

STATS = RESULTS / "03_Statistical_Results"

LOGS = (
    RESULTS /
    "05_Reproducibility" /
    "Execution_Logs"
)

for folder in [TABLES, STATS, LOGS]:
    folder.mkdir(parents=True, exist_ok=True)

FEATURE_SOURCE = (
    OLD /
    "cell26_locked_feature_specification.csv"
)

if not DATA.exists():
    raise FileNotFoundError(DATA)

if not FEATURE_SOURCE.exists():
    raise FileNotFoundError(FEATURE_SOURCE)

print("=" * 75)
print("ROUND 2 - STEP 19")
print("PREDICTOR AVAILABILITY AND LEAKAGE AUDIT")
print("=" * 75)

# ============================================================
# 1. Read locked feature specification
# ============================================================

features = pd.read_csv(FEATURE_SOURCE)

required = {"feature", "type"}

if not required.issubset(features.columns):
    raise ValueError(
        "Feature specification missing required columns."
    )

features["feature"] = (
    features["feature"].astype(str).str.strip()
)

if features["feature"].duplicated().any():
    raise ValueError(
        "Duplicate features found in locked specification."
    )

print("\nLocked features:", len(features))

# ============================================================
# 2. Match the actual training feature exclusions
# ============================================================

HIGH_CARDINALITY_EXCLUSIONS = {
    "emp_title",
    "desc",
    "title"
}

features["used_in_model"] = (
    ~features["feature"].isin(
        HIGH_CARDINALITY_EXCLUSIONS
    )
)

print(
    "Features used in original models:",
    int(features["used_in_model"].sum())
)

# ============================================================
# 3. Read Stata metadata only
# ============================================================

print("\nReading original Stata variable metadata...")

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

source_columns = set(sample.columns)

print("Source dataset variables:", len(source_columns))

# ============================================================
# 4. Define preliminary timing-risk rules
# ============================================================

POST_ORIGINATION = {
    "out_prncp":
        "Outstanding principal is updated after origination",

    "out_prncp_inv":
        "Investor outstanding principal is updated after origination",

    "total_pymnt":
        "Payments received to date",

    "total_pymnt_inv":
        "Investor payments received to date",

    "total_rec_prncp":
        "Principal repayments received to date",

    "total_rec_int":
        "Interest repayments received to date",

    "total_rec_late_fee":
        "Late fees received after origination",

    "recoveries":
        "Post-charge-off recoveries",

    "collection_recovery_fee":
        "Post-charge-off collection fee",

    "last_pymnt_d":
        "Last payment date is post-origination information",

    "last_pymnt_amnt":
        "Last payment amount is post-origination information",

    "next_pymnt_d":
        "Next scheduled payment may reflect subsequent servicing",

    "last_credit_pull_d":
        "Most recent credit pull may occur after origination",

    "last_fico_range_high":
        "Updated credit score may reflect post-origination information",

    "last_fico_range_low":
        "Updated credit score may reflect post-origination information",

    "loan_status":
        "Recorded loan outcome status",

    "badloan":
        "Outcome target",

    "last_date":
        "Constructed date with unverified timing",

    "daily_date":
        "Constructed date with unverified timing",

    "debt_settlement_flag":
        "Debt-settlement information is post-origination",

    "pymnt_plan":
        "Servicing payment-plan status may change after origination"
}

POST_PREFIXES = (
    "hardship_",
    "settlement_",
    "debt_settlement_",
    "last_fico_",
    "loan_status_"
)

CONDITIONAL_FIELDS = {
    "funded_amnt":
        "Verify availability at the intended decision point",

    "funded_amnt_inv":
        "Investor funding may be known only after loan funding",

    "int_rate":
        "Interest rate depends on the underwriting decision stage",

    "grade":
        "Platform-assigned grade depends on decision timing",

    "sub_grade":
        "Platform-assigned subgrade depends on decision timing",

    "issue_d":
        "Origination date requires a defined prediction timestamp",

    "issue_date":
        "Derived origination date requires temporal verification",

    "issue_year":
        "Derived origination year requires temporal verification",

    "issue_daily":
        "Derived origination date requires temporal verification",

    "unemployment":
        "Macroeconomic observation period requires verification",

    "unem_rate":
        "Unemployment rate must be aligned to origination",

    "gdp":
        "Economic data publication timing requires verification",

    "inflation":
        "Economic data publication timing requires verification",

    "cpi":
        "Economic data publication timing requires verification"
}

CONDITIONAL_PREFIXES = (
    "macro_",
    "unem_",
    "gdp_",
    "cpi_",
    "inflation_",
    "fedfunds_",
    "treasury_"
)


def assess_timing(name, label):

    name = str(name).strip()

    label_lower = str(label).lower()

    if name in POST_ORIGINATION:

        return (
            "HIGH_RISK_POST_ORIGINATION",
            POST_ORIGINATION[name]
        )

    if name.startswith(POST_PREFIXES):

        return (
            "HIGH_RISK_POST_ORIGINATION",
            "Variable name indicates subsequent "
            "servicing, settlement or outcome information"
        )

    if "post charge off" in label_lower:

        return (
            "HIGH_RISK_POST_ORIGINATION",
            "Dataset label explicitly references post-charge-off data"
        )

    if "payments received to date" in label_lower:

        return (
            "HIGH_RISK_POST_ORIGINATION",
            "Dataset label references cumulative repayments"
        )

    if name in CONDITIONAL_FIELDS:

        return (
            "TIMING_REQUIRES_VERIFICATION",
            CONDITIONAL_FIELDS[name]
        )

    if name.startswith(CONDITIONAL_PREFIXES):

        return (
            "TIMING_REQUIRES_VERIFICATION",
            "Verify historical information availability "
            "and publication lag"
        )

    return (
        "NOT_FLAGGED_UNVERIFIED",
        "No obvious timing concern identified by "
        "automated screening; manual validation required"
    )


# ============================================================
# 5. Audit every locked feature
# ============================================================

rows = []

for _, record in features.iterrows():

    name = record["feature"]

    label = labels.get(name, "") or ""

    source_exists = name in source_columns

    flag, reason = assess_timing(name, label)

    if not source_exists:

        flag = "MISSING_FROM_SOURCE"

        reason = (
            "Feature not found in original Stata schema"
        )

    rows.append({
        "feature": name,
        "declared_type": record["type"],
        "used_in_model": bool(
            record["used_in_model"]
        ),
        "exists_in_source": source_exists,
        "source_dtype": (
            str(sample[name].dtype)
            if source_exists else ""
        ),
        "source_variable_label": str(label),
        "timing_flag": flag,
        "reason": reason
    })

audit = pd.DataFrame(rows)

# ============================================================
# 6. Identify potentially problematic model inputs
# ============================================================

flagged = audit.loc[
    audit["used_in_model"]
    &
    audit["timing_flag"].ne(
        "NOT_FLAGGED_UNVERIFIED"
    )
].copy()

flagged = flagged.sort_values(
    ["timing_flag", "feature"]
)

summary = (
    audit.groupby(
        ["used_in_model", "timing_flag"],
        as_index=False
    )
    .agg(
        feature_count=("feature", "size")
    )
)

# ============================================================
# 7. Save reproducible results
# ============================================================

full_path = (
    TABLES /
    "19_full_predictor_provenance_audit.csv"
)

flagged_path = (
    TABLES /
    "19_flagged_predictors_PRELIMINARY.csv"
)

summary_path = (
    STATS /
    "19_predictor_timing_risk_summary.csv"
)

audit.to_csv(
    full_path,
    index=False
)

flagged.to_csv(
    flagged_path,
    index=False
)

summary.to_csv(
    summary_path,
    index=False
)

# ============================================================
# 8. Generate readable audit report
# ============================================================

report_path = (
    LOGS /
    "19_predictor_provenance_report.txt"
)

report = [
    "ROUND 2 - STEP 19",
    "Predictor provenance and timing audit",
    "",
    f"Generated: {datetime.now()}",
    f"Source dataset: {DATA}",
    f"Locked feature specification: {FEATURE_SOURCE}",
    "",
    f"Locked features: {len(audit)}",
    f"Model input features: {int(audit['used_in_model'].sum())}",
    f"Flagged model inputs: {len(flagged)}",
    "",
    "IMPORTANT METHODOLOGICAL NOTE:",
    "This is an automated screening audit.",
    "It does not independently establish the",
    "timestamp at which each predictor was measured.",
    "",
    "HIGH_RISK_POST_ORIGINATION:",
    "Candidate predictors requiring investigation",
    "because their names or labels indicate",
    "post-origination information.",
    "",
    "TIMING_REQUIRES_VERIFICATION:",
    "Candidate predictors requiring an explicit",
    "prediction timestamp or source-date check.",
    "",
    "NOT_FLAGGED_UNVERIFIED:",
    "Features without an obvious automated",
    "timing warning remain unverified.",
    "",
    "=" * 72,
    "SUMMARY",
    "=" * 72,
    summary.to_string(index=False),
    "",
    "=" * 72,
    "FLAGGED MODEL INPUTS",
    "=" * 72,
    flagged.to_string(index=False),
    "",
    "=" * 72,
    "COMPLETE FEATURE AUDIT",
    "=" * 72,
    audit.to_string(index=False)
]

report_path.write_text(
    "\n".join(report),
    encoding="utf-8"
)

# ============================================================
# 9. Display findings
# ============================================================

print("\n" + "=" * 75)
print("PREDICTOR TIMING SUMMARY")
print("=" * 75)

print(summary.to_string(index=False))

print("\n" + "=" * 75)
print("FLAGGED MODEL INPUTS")
print("=" * 75)

print(
    flagged[
        [
            "feature",
            "timing_flag",
            "reason"
        ]
    ].to_string(index=False)
)

print("\n" + "=" * 75)
print("OUTPUTS SAVED")
print("=" * 75)

print(full_path)
print(flagged_path)
print(summary_path)
print(report_path)

print("\nSTEP 19 COMPLETED SUCCESSFULLY")