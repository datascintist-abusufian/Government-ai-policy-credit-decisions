import argparse
import csv
import re
from pathlib import Path
from collections import defaultdict


parser = argparse.ArgumentParser()
parser.add_argument("--root", required=True)
args = parser.parse_args()

root = Path(args.root)

evidence = (
    root
    / "Round2_Manuscript_Results"
    / "05_Reproducibility"
    / "Evidence_Checks"
)

stata_file = root / "P2P_Macro_Codes.do"

manifest150 = evidence / "48_exact_150_feature_manifest.csv"
manifest145 = evidence / "54_145_feature_direct_leakage_quarantine.csv"
manifest133 = evidence / "54_133_feature_timing_risk_quarantine.csv"

required = [stata_file, manifest150, manifest145, manifest133]

for path in required:
    if not path.exists():
        raise FileNotFoundError(path)


def load_features(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        return [
            row["feature"]
            for row in csv.DictReader(f)
        ]


features150 = load_features(manifest150)
features145 = load_features(manifest145)
features133 = load_features(manifest133)

assert len(features150) == 150
assert len(features145) == 145
assert len(features133) == 133

print("=" * 70)
print("STEP 55: INDIRECT TEMPORAL LEAKAGE AUDIT")
print("=" * 70)

# ------------------------------------------------------------
# Read Stata source
# ------------------------------------------------------------

stata_text = stata_file.read_text(
    encoding="utf-8-sig",
    errors="replace"
)

# Join Stata statements continued with ///
statements = []
buffer = ""
start_line = None

for number, raw in enumerate(stata_text.splitlines(), start=1):

    line = raw.strip()

    if not line or line.startswith("*"):
        continue

    if start_line is None:
        start_line = number

    continuation = "///" in line

    if continuation:
        line = line.split("///", 1)[0]

    buffer += " " + line

    if not continuation:
        statements.append((start_line, buffer.strip()))
        buffer = ""
        start_line = None

if buffer:
    statements.append((start_line, buffer.strip()))

# ------------------------------------------------------------
# Identify generated and replaced variables
# ------------------------------------------------------------

assignment_pattern = re.compile(
    r"\b(?:gen|generate|egen|replace)\s+"
    r"([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)",
    re.IGNORECASE
)

assignments = defaultdict(list)
targets = set()

for line_number, statement in statements:

    match = assignment_pattern.search(statement)

    if match:
        target = match.group(1)
        targets.add(target)

        assignments[target].append({
            "line": line_number,
            "statement": statement
        })

# Include intermediate generated variables in dependency matching.
vocabulary = (
    set(features150)
    | targets
    | {
        "last_pymnt_d",
        "last_date",
        "daily_date",
        "loan_status",
        "badloan",
        "issue_date"
    }
)

dependencies = defaultdict(set)

token_pattern = re.compile(
    r"[A-Za-z_][A-Za-z0-9_]*"
)

for target, definitions in assignments.items():

    for definition in definitions:

        tokens = set(
            token_pattern.findall(
                definition["statement"]
            )
        )

        dependencies[target].update(
            (tokens & vocabulary) - {target}
        )

# ------------------------------------------------------------
# Trace transitive dependencies
# ------------------------------------------------------------

def trace_dependency(seed):

    affected = {seed}

    changed = True

    while changed:

        changed = False

        for target, sources in dependencies.items():

            if target in affected:
                continue

            if sources & affected:

                affected.add(target)
                changed = True

    return affected


payment_derived = trace_dependency("last_pymnt_d")

outcome_derived = trace_dependency("loan_status")

# ------------------------------------------------------------
# Compare feature specifications
# ------------------------------------------------------------

print("\n===== PAYMENT-HISTORY DEPENDENCIES =====")

affected_original = sorted(
    set(features150) & payment_derived
)

for name in affected_original:

    print(f"\nFEATURE: {name}")

    print(
        "Dependencies:",
        sorted(dependencies.get(name, set()))
    )

    for definition in assignments.get(name, []):

        print(
            f"Stata line {definition['line']}: "
            f"{definition['statement'][:220]}"
        )

print("\n===== FEATURE SET COMPARISON =====")

rows = []

for name in features150:

    rows.append({
        "feature": name,

        "payment_history_dependency": (
            name in payment_derived
        ),

        "loan_status_dependency": (
            name in outcome_derived
        ),

        "included_original_150": True,

        "included_candidate_145": (
            name in features145
        ),

        "included_candidate_133": (
            name in features133
        ),

        "stata_source_dependencies": ";".join(
            sorted(dependencies.get(name, set()))
        )
    })

remaining145 = sorted(
    set(features145) & payment_derived
)

remaining133 = sorted(
    set(features133) & payment_derived
)

outcome150 = sorted(
    set(features150) & outcome_derived
)

print(
    "Original 150-feature payment-dependent variables:",
    len(affected_original)
)

print(
    "Payment-dependent variables remaining in 145:",
    len(remaining145)
)

print(
    "Payment-dependent variables remaining in 133:",
    len(remaining133)
)

print(
    "Source-code outcome-dependent variables in 150:",
    len(outcome150)
)

print("\nRemaining in 145:")
print(remaining145)

print("\nRemaining in 133:")
print(remaining133)

print("\nOutcome-dependent variables:")
print(outcome150)

# ------------------------------------------------------------
# Save audit
# ------------------------------------------------------------

evidence.mkdir(parents=True, exist_ok=True)

csv_path = (
    evidence
    / "55_indirect_feature_dependency_register.csv"
)

with open(
    csv_path,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=list(rows[0].keys())
    )

    writer.writeheader()
    writer.writerows(rows)

summary_path = (
    evidence
    / "55_indirect_leakage_audit_summary.txt"
)

with open(
    summary_path,
    "w",
    encoding="utf-8"
) as f:

    f.write("STEP 55 INDIRECT LEAKAGE AUDIT\n\n")

    f.write(
        f"Original payment-dependent features: "
        f"{affected_original}\n"
    )

    f.write(
        f"Remaining in 145: {remaining145}\n"
    )

    f.write(
        f"Remaining in 133: {remaining133}\n"
    )

    f.write(
        f"Outcome-dependent features: {outcome150}\n"
    )

    f.write(
        "\nThis is a static source-code dependency audit. "
        "It cannot establish that the supplied Stata script "
        "generated every stored field or certify that "
        "other variables were available at decision time.\n"
    )

print("\nSaved:", csv_path)
print("Saved:", summary_path)

print("\nSTEP 55 COMPLETE")
print("Original dataset, notebooks and results unchanged.")
