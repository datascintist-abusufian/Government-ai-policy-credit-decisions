from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd

ROOT = Path(r"E:\Governance-Policy\Cogent Economics & Finance")

OLD = ROOT / "revision_outputs"

RESULTS = ROOT / "Round2_Manuscript_Results"

TABLES = RESULTS / "01_Tables"

STATS = RESULTS / "03_Statistical_Results"

LOGS = RESULTS / "05_Reproducibility" / "Execution_Logs"

for folder in [TABLES, STATS, LOGS]:
    folder.mkdir(parents=True, exist_ok=True)

SOURCE = OLD / "cell30_recalibrated_threshold_analysis.csv"

TEST_SOURCE = (
    OLD / "cell30_recalibrated_test_operating_points.csv"
)

LABEL_PATHS = {
    "Validation": OLD / "cell27_y_val.npy",
    "Test": OLD / "cell27_y_test.npy"
}

for path in [SOURCE, TEST_SOURCE, *LABEL_PATHS.values()]:
    if not path.exists():
        raise FileNotFoundError(path)

print("=" * 75)
print("ROUND 2 - STEP 17")
print("DECISION POLICY VS TRIVIAL BASELINES")
print("=" * 75)

# ============================================================
# 1. Load original policy results
# ============================================================

policy = pd.read_csv(SOURCE)

test_policy = pd.read_csv(TEST_SOURCE)

labels = {}

for cohort, path in LABEL_PATHS.items():

    y = np.load(path, allow_pickle=False)

    if y.ndim != 1:
        raise ValueError(
            f"{cohort}: outcome array must be one-dimensional"
        )

    if not np.isin(y, [0, 1]).all():
        raise ValueError(
            f"{cohort}: outcome array is not binary"
        )

    labels[cohort] = y.astype(np.int8)

    print(
        f"{cohort}: N={len(y):,}, "
        f"adverse={int(y.sum()):,}"
    )

# ============================================================
# 2. Recalculate policy and baseline costs
# ============================================================

rows = []

checks = []

for _, record in policy.iterrows():

    cohort = str(record["cohort"]).strip()

    if cohort not in labels:
        raise ValueError(f"Unexpected cohort: {cohort}")

    y = labels[cohort]

    n = len(y)

    adverse = int(y.sum())

    nonadverse = n - adverse

    tp = int(record["tp"])
    fp = int(record["fp"])
    tn = int(record["tn"])
    fn = int(record["fn"])

    fn_cost = float(record["fn_cost"])
    fp_cost = float(record["fp_cost"])

    ratio = str(record["fn_fp_ratio"])

    threshold = float(record["threshold"])

    # --------------------------------------------------------
    # A. Verify confusion matrix
    # --------------------------------------------------------

    confusion_valid = (
        tp + fn == adverse
        and
        tn + fp == nonadverse
        and
        tp + fp + tn + fn == n
    )

    if not confusion_valid:
        raise RuntimeError(
            f"Confusion matrix mismatch: {cohort}, {ratio}"
        )

    # --------------------------------------------------------
    # B. Cost calculations
    # --------------------------------------------------------

    # Original model-selected policy
    model_cost = fn_cost * fn + fp_cost * fp

    # No-review proxy: classify everyone as non-adverse
    no_review_cost = fn_cost * adverse

    # Review-all proxy: classify everyone as adverse
    review_all_cost = fp_cost * nonadverse

    # Best trivial classification baseline
    best_baseline_cost = min(
        no_review_cost,
        review_all_cost
    )

    if no_review_cost <= review_all_cost:
        best_baseline = "No_review"
    else:
        best_baseline = "Review_all_proxy"

    # Best among three available policies
    policies = {
        "Model_selected": model_cost,
        "No_review": no_review_cost,
        "Review_all_proxy": review_all_cost
    }

    best_policy = min(
        policies,
        key=policies.get
    )

    best_policy_cost = policies[best_policy]

    # --------------------------------------------------------
    # C. Audit original reported values
    # --------------------------------------------------------

    reported_cost = float(record["cost"])

    reported_no_review = float(
        record["no_review_cost"]
    )

    reported_net_benefit = float(
        record["net_benefit"]
    )

    cost_check = np.isclose(
        model_cost,
        reported_cost,
        atol=1e-7
    )

    no_review_check = np.isclose(
        no_review_cost,
        reported_no_review,
        atol=1e-7
    )

    net_benefit_check = np.isclose(
        no_review_cost - model_cost,
        reported_net_benefit,
        atol=1e-7
    )

    if not all([
        cost_check,
        no_review_check,
        net_benefit_check
    ]):
        raise RuntimeError(
            f"Reported-cost inconsistency: {cohort}, {ratio}"
        )

    # --------------------------------------------------------
    # D. Save comprehensive comparison
    # --------------------------------------------------------

    rows.append({
        "cohort": cohort,
        "fn_fp_ratio": ratio,
        "n": n,
        "adverse_n": adverse,
        "nonadverse_n": nonadverse,

        "fn_cost": fn_cost,
        "fp_cost": fp_cost,

        "selected_threshold": threshold,

        "model_review_n": tp + fp,
        "model_review_rate": (tp + fp) / n,

        "model_cost": model_cost,
        "no_review_cost": no_review_cost,
        "review_all_proxy_cost": review_all_cost,

        "best_trivial_baseline": best_baseline,
        "best_trivial_baseline_cost": best_baseline_cost,

        "model_saving_vs_no_review":
            no_review_cost - model_cost,

        "model_saving_vs_review_all":
            review_all_cost - model_cost,

        "model_saving_vs_best_trivial_baseline":
            best_baseline_cost - model_cost,

        "model_beats_both_baselines":
            model_cost < best_baseline_cost,

        "best_policy_among_three": best_policy,
        "lowest_cost_among_three": best_policy_cost,

        "cost_per_loan": model_cost / n,
        "best_baseline_cost_per_loan":
            best_baseline_cost / n
    })

    checks.append({
        "cohort": cohort,
        "fn_fp_ratio": ratio,

        "confusion_matrix_valid": confusion_valid,
        "model_cost_valid": bool(cost_check),
        "no_review_cost_valid": bool(no_review_check),
        "net_benefit_valid": bool(net_benefit_check)
    })

# ============================================================
# 3. Save results
# ============================================================

comparison = pd.DataFrame(rows)

comparison = comparison.sort_values(
    ["cohort", "fn_cost"]
)

checks_df = pd.DataFrame(checks)

comparison_path = (
    TABLES / "17_policy_vs_trivial_baselines.csv"
)

checks_path = (
    STATS / "17_policy_cost_verification.csv"
)

comparison.to_csv(
    comparison_path,
    index=False
)

checks_df.to_csv(
    checks_path,
    index=False
)

# ============================================================
# 4. Cross-check separate test operating-point table
# ============================================================

test_checks = []

for _, record in test_policy.iterrows():

    ratio = str(record["fn_fp_ratio"])

    match = comparison[
        (comparison["cohort"] == "Test")
        &
        (comparison["fn_fp_ratio"] == ratio)
    ]

    if len(match) != 1:
        raise RuntimeError(
            f"Test table mismatch for ratio {ratio}"
        )

    selected = match.iloc[0]

    checks_ok = all([
        np.isclose(
            float(record["cost"]),
            selected["model_cost"]
        ),
        np.isclose(
            float(record["no_review_cost"]),
            selected["no_review_cost"]
        ),
        np.isclose(
            float(record["threshold"]),
            selected["selected_threshold"]
        )
    ])

    test_checks.append({
        "ratio": ratio,
        "separate_test_table_consistent": checks_ok
    })

test_checks_df = pd.DataFrame(test_checks)

test_check_path = (
    STATS / "17_test_table_crosscheck.csv"
)

test_checks_df.to_csv(
    test_check_path,
    index=False
)

# ============================================================
# 5. Generate readable audit report
# ============================================================

report_path = (
    LOGS / "17_policy_benchmark_report.txt"
)

report = [
    "ROUND 2 - REVIEWER 2 COMMENT 2",
    "Model-based policy versus trivial baselines",
    "",
    f"Generated: {datetime.now()}",
    "",
    "COST FRAMEWORK:",
    "False-negative cost = ratio-dependent",
    "False-positive cost = 1 unless otherwise specified",
    "",
    "NO-REVIEW PROXY:",
    "Classify every loan as non-adverse.",
    "",
    "REVIEW-ALL PROXY:",
    "Classify every loan as adverse.",
    "",
    "IMPORTANT LIMITATION:",
    "These are classification-cost benchmarks.",
    "They do not independently model human review",
    "staffing costs, review effectiveness or capacity.",
    "",
    "POLICY COMPARISON:",
    comparison.to_string(index=False),
    "",
    "ORIGINAL COST VERIFICATION:",
    checks_df.to_string(index=False),
    "",
    "SEPARATE TEST TABLE CONSISTENCY:",
    test_checks_df.to_string(index=False)
]

report_path.write_text(
    "\n".join(report),
    encoding="utf-8"
)

# ============================================================
# 6. Display results
# ============================================================

display_cols = [
    "cohort",
    "fn_fp_ratio",
    "model_cost",
    "no_review_cost",
    "review_all_proxy_cost",
    "model_saving_vs_best_trivial_baseline",
    "best_policy_among_three"
]

print("\n" + "=" * 75)
print("POLICY COMPARISON")
print("=" * 75)

print(
    comparison[display_cols].to_string(index=False)
)

print("\n" + "=" * 75)
print("VERIFICATION")
print("=" * 75)

print(checks_df.to_string(index=False))

print("\nSEPARATE TEST TABLE CHECKS:")
print(test_checks_df.to_string(index=False))

print("\nOutputs saved:")
print(comparison_path)
print(checks_path)
print(test_check_path)
print(report_path)

if not test_checks_df[
    "separate_test_table_consistent"
].all():
    raise RuntimeError(
        "Separate test operating-point table mismatch."
    )

print("\nSTEP 17 COMPLETED SUCCESSFULLY")