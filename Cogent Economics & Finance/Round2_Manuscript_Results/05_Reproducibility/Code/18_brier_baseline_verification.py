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

SOURCE = OLD / "cell27_training_size_sensitivity.csv"

CALIBRATION_SOURCE = (
    OLD / "cell29_before_after_calibration.csv"
)

LABEL_PATHS = {
    "Validation": OLD / "cell27_y_val.npy",
    "Test": OLD / "cell27_y_test.npy"
}

CALIBRATED_PATHS = {
    "Validation":
        OLD / "cell29_histgb_full_val_recalibrated.npy",
    "Test":
        OLD / "cell29_histgb_full_test_recalibrated.npy"
}

required = [
    SOURCE,
    CALIBRATION_SOURCE,
    *LABEL_PATHS.values(),
    *CALIBRATED_PATHS.values()
]

for path in required:
    if not path.exists():
        raise FileNotFoundError(path)

print("=" * 75)
print("ROUND 2 - STEP 18")
print("BRIER-SCORE BASELINE VERIFICATION")
print("=" * 75)

# ============================================================
# 1. Load saved results and labels
# ============================================================

original = pd.read_csv(SOURCE)

calibration = pd.read_csv(CALIBRATION_SOURCE)

labels = {}

for cohort, path in LABEL_PATHS.items():

    y = np.load(
        path,
        allow_pickle=False
    ).astype(np.int8)

    if y.ndim != 1:
        raise ValueError(
            f"{cohort}: Labels must be one-dimensional."
        )

    if not np.isin(y, [0, 1]).all():
        raise ValueError(
            f"{cohort}: Labels must be binary."
        )

    labels[cohort] = y

    print(
        f"{cohort}: N={len(y):,}, "
        f"adverse rate={y.mean():.6f}"
    )

validation_rate = float(
    labels["Validation"].mean()
)

# ============================================================
# 2. Helper functions
# ============================================================

def validate_predictions(p, y, name):

    if p.ndim != 1:
        raise ValueError(
            f"{name}: Predictions must be one-dimensional."
        )

    if len(p) != len(y):
        raise ValueError(
            f"{name}: Prediction length mismatch."
        )

    if not np.isfinite(p).all():
        raise ValueError(
            f"{name}: Non-finite predictions detected."
        )

    if ((p < 0) | (p > 1)).any():
        raise ValueError(
            f"{name}: Predictions outside [0, 1]."
        )


def brier(y, probabilities):

    y = np.asarray(y, dtype=np.float64)

    p = np.asarray(
        probabilities,
        dtype=np.float64
    )

    return float(
        np.mean((p - y) ** 2)
    )


def constant_brier(y, probability):

    return float(
        np.mean(
            (
                np.asarray(y, dtype=np.float64)
                -
                float(probability)
            ) ** 2
        )
    )


def skill_score(model_brier, baseline_brier):

    if baseline_brier <= 0:
        return np.nan

    return float(
        1 - model_brier / baseline_brier
    )


# ============================================================
# 3. Verify all original model predictions
# ============================================================

model_rows = []

verification_rows = []

for _, record in original.iterrows():

    training_n = int(record["training_n"])

    model = str(record["model"])

    training_rate = float(
        record["train_event_rate"]
    )

    if not 0 <= training_rate <= 1:
        raise ValueError(
            f"Invalid training prevalence: {training_rate}"
        )

    for cohort, suffix in [
        ("Validation", "val"),
        ("Test", "test")
    ]:

        y = labels[cohort]

        prediction_path = (
            OLD /
            f"cell27_{training_n}_{model}_{suffix}_pred.npy"
        )

        if not prediction_path.exists():
            raise FileNotFoundError(prediction_path)

        p = np.load(
            prediction_path,
            allow_pickle=False
        )

        validate_predictions(
            p,
            y,
            prediction_path.name
        )

        model_brier = brier(y, p)

        # Baseline 1:
        # Constant adverse probability from the
        # model's own training sample.

        train_baseline = constant_brier(
            y,
            training_rate
        )

        # Baseline 2:
        # Constant probability estimated from
        # the validation cohort.

        val_baseline = constant_brier(
            y,
            validation_rate
        )

        # Baseline 3:
        # Cohort-specific prevalence.
        # For test, this uses test outcomes and
        # is an oracle/descriptive benchmark.

        cohort_rate = float(y.mean())

        oracle_baseline = constant_brier(
            y,
            cohort_rate
        )

        reported_column = (
            "val_brier"
            if cohort == "Validation"
            else "test_brier"
        )

        reported_brier = float(
            record[reported_column]
        )

        verification_pass = bool(
            np.isclose(
                model_brier,
                reported_brier,
                atol=2e-6,
                rtol=0
            )
        )

        verification_rows.append({
            "training_n": training_n,
            "model": model,
            "cohort": cohort,
            "recalculated_brier": model_brier,
            "reported_brier": reported_brier,
            "absolute_difference": abs(
                model_brier - reported_brier
            ),
            "verification_pass": verification_pass
        })

        model_rows.append({
            "training_n": training_n,
            "model": model,
            "cohort": cohort,
            "n": len(y),
            "observed_adverse_rate": cohort_rate,
            "training_adverse_rate": training_rate,
            "validation_adverse_rate": validation_rate,

            "model_brier": model_brier,

            "training_prevalence_baseline_brier":
                train_baseline,

            "validation_prevalence_baseline_brier":
                val_baseline,

            "oracle_cohort_prevalence_baseline_brier":
                oracle_baseline,

            "bss_vs_training_prevalence":
                skill_score(
                    model_brier,
                    train_baseline
                ),

            "bss_vs_validation_prevalence":
                skill_score(
                    model_brier,
                    val_baseline
                ),

            "bss_vs_oracle_prevalence":
                skill_score(
                    model_brier,
                    oracle_baseline
                ),

            "beats_training_prevalence_baseline":
                model_brier < train_baseline,

            "beats_validation_prevalence_baseline":
                model_brier < val_baseline
        })

# ============================================================
# 4. Verify recalibrated full-training HistGB
# ============================================================

calibration_rows = []

FULL_TRAIN_N = 1304058

train_rate_matches = original.loc[
    (original["training_n"] == FULL_TRAIN_N)
    &
    (original["model"] == "HistGB"),
    "train_event_rate"
]

if len(train_rate_matches) != 1:
    raise RuntimeError(
        "Full-training HistGB prevalence not uniquely found."
    )

full_train_rate = float(
    train_rate_matches.iloc[0]
)

for cohort, path in CALIBRATED_PATHS.items():

    y = labels[cohort]

    p = np.load(
        path,
        allow_pickle=False
    )

    validate_predictions(
        p,
        y,
        path.name
    )

    actual_brier = brier(y, p)

    match = calibration.loc[
        (calibration["cohort"] == cohort)
        &
        (calibration["stage"] == "After")
    ]

    if len(match) != 1:
        raise RuntimeError(
            f"{cohort}: Calibration result not unique."
        )

    reported = float(
        match.iloc[0]["brier"]
    )

    verified = bool(
        np.isclose(
            actual_brier,
            reported,
            atol=2e-6,
            rtol=0
        )
    )

    training_baseline = constant_brier(
        y,
        full_train_rate
    )

    validation_baseline = constant_brier(
        y,
        validation_rate
    )

    oracle_baseline = constant_brier(
        y,
        float(y.mean())
    )

    calibration_rows.append({
        "model": "HistGB_Full_Recalibrated",
        "training_n": FULL_TRAIN_N,
        "cohort": cohort,
        "n": len(y),

        "recalibrated_brier": actual_brier,

        "reported_recalibrated_brier": reported,

        "reported_brier_verified": verified,

        "training_prevalence_baseline_brier":
            training_baseline,

        "validation_prevalence_baseline_brier":
            validation_baseline,

        "oracle_cohort_prevalence_baseline_brier":
            oracle_baseline,

        "bss_vs_training_prevalence":
            skill_score(
                actual_brier,
                training_baseline
            ),

        "bss_vs_validation_prevalence":
            skill_score(
                actual_brier,
                validation_baseline
            ),

        "bss_vs_oracle_prevalence":
            skill_score(
                actual_brier,
                oracle_baseline
            ),

        "beats_training_prevalence_baseline":
            actual_brier < training_baseline
    })

# ============================================================
# 5. Build result tables
# ============================================================

model_results = pd.DataFrame(model_rows)

verification = pd.DataFrame(verification_rows)

calibration_results = pd.DataFrame(
    calibration_rows
)

# ============================================================
# 6. Save all statistical results
# ============================================================

model_path = (
    TABLES /
    "18_brier_baseline_model_comparison.csv"
)

calibration_path = (
    TABLES /
    "18_recalibrated_histgb_brier_skill.csv"
)

verification_path = (
    STATS /
    "18_brier_recalculation_verification.csv"
)

model_results.to_csv(
    model_path,
    index=False
)

calibration_results.to_csv(
    calibration_path,
    index=False
)

verification.to_csv(
    verification_path,
    index=False
)

# ============================================================
# 7. Prepare readable execution report
# ============================================================

report_path = (
    LOGS /
    "18_brier_baseline_audit_report.txt"
)

test_results = model_results.loc[
    model_results["cohort"] == "Test"
].copy()

test_results = test_results.sort_values(
    ["training_n", "model"]
)

display_cols = [
    "training_n",
    "model",
    "model_brier",
    "training_prevalence_baseline_brier",
    "bss_vs_training_prevalence",
    "beats_training_prevalence_baseline"
]

all_checks_pass = bool(
    verification["verification_pass"].all()
    and
    calibration_results[
        "reported_brier_verified"
    ].all()
)

report = [
    "ROUND 2 - REVIEWER 2 COMMENT 6",
    "Brier-score baseline verification",
    "",
    f"Generated: {datetime.now()}",
    f"NumPy version: {np.__version__}",
    f"Pandas version: {pd.__version__}",
    "",
    "PRIMARY BASELINE:",
    "Constant probability equal to each",
    "model's training-sample adverse rate.",
    "",
    "SECONDARY BASELINE:",
    "Constant validation-cohort adverse rate.",
    "",
    "ORACLE BASELINE:",
    "Constant observed cohort adverse rate.",
    "The test-cohort version uses test outcomes.",
    "It is descriptive and not a deployable forecast.",
    "",
    "BRIER SKILL SCORE:",
    "1 - model_brier / baseline_brier",
    "",
    "TEST RESULTS:",
    test_results[
        display_cols
    ].to_string(index=False),
    "",
    "RECALIBRATED HISTGB:",
    calibration_results.to_string(index=False),
    "",
    "ORIGINAL SCORE VERIFICATION:",
    verification.to_string(index=False),
    "",
    "ALL VERIFICATION CHECKS:",
    "PASS" if all_checks_pass else "FAIL",
    "",
    "IMPORTANT LIMITATION:",
    "These results apply to the original",
    "outcome-selected evaluation cohorts.",
    "They do not resolve maturity-selection bias."
]

report_path.write_text(
    "\n".join(report),
    encoding="utf-8"
)

# ============================================================
# 8. Display summary
# ============================================================

print("\n" + "=" * 75)
print("TEST BRIER SCORES VS TRAINING BASELINE")
print("=" * 75)

print(
    test_results[
        display_cols
    ].to_string(index=False)
)

print("\n" + "=" * 75)
print("RECALIBRATED HISTGB")
print("=" * 75)

print(
    calibration_results[
        [
            "cohort",
            "recalibrated_brier",
            "training_prevalence_baseline_brier",
            "bss_vs_training_prevalence",
            "beats_training_prevalence_baseline"
        ]
    ].to_string(index=False)
)

print("\nVerification:",
      "PASS" if all_checks_pass else "FAIL")

print("\nOutputs saved:")

print(model_path)
print(calibration_path)
print(verification_path)
print(report_path)

if not all_checks_pass:
    raise RuntimeError(
        "One or more Brier values failed verification. "
        "Inspect the saved audit report."
    )

print("\nSTEP 18 COMPLETED SUCCESSFULLY")