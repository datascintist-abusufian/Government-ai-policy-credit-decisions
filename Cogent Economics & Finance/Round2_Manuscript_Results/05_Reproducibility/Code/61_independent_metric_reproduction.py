import argparse
import csv
import io
import re
import zipfile
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score, brier_score_loss


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

archive = (
    evidence
    / "44_Cogent_replication_candidate_NOT_VERIFIED.zip"
)

if not archive.exists():
    raise FileNotFoundError(archive)

# Original reported 150-feature test ROC-AUC values

expected_auc = {
    (180000, "Logistic_SGD"): 0.646062,
    (180000, "HistGB"): 0.697660,
    (180000, "MLP"): 0.718945,

    (360000, "Logistic_SGD"): 0.672210,
    (360000, "HistGB"): 0.687550,
    (360000, "MLP"): 0.700299,

    (540000, "Logistic_SGD"): 0.665383,
    (540000, "HistGB"): 0.674609,
    (540000, "MLP"): 0.694318,

    (1304058, "Logistic_SGD"): 0.699310,
    (1304058, "HistGB"): 0.675016,
    (1304058, "MLP"): 0.646276,
}


def identify_model(name):

    normal = re.sub(
        r"[^a-z0-9]",
        "",
        name.lower()
    )

    if "logistic" in normal or normal == "sgd":
        return "Logistic_SGD"

    if "hist" in normal:
        return "HistGB"

    if "mlp" in normal:
        return "MLP"

    return None


print("=" * 72)
print("STEP 61: INDEPENDENT MODEL METRIC REPRODUCTION")
print("=" * 72)

results = []
found = {}

with zipfile.ZipFile(archive, "r") as z:

    names = [
        n for n in z.namelist()
        if not n.endswith("/")
    ]

    # Find original test labels

    labels = [
        name for name in names
        if Path(name).name == "cell27_y_test.npy"
    ]

    if len(labels) != 1:
        raise RuntimeError(
            "Expected one archived cell27_y_test.npy; "
            f"found {len(labels)}."
        )

    y = np.load(
        io.BytesIO(z.read(labels[0])),
        allow_pickle=False
    )

    if y.shape != (41145,):
        raise RuntimeError(
            f"Unexpected test-label shape: {y.shape}"
        )

    if set(np.unique(y).tolist()) != {0, 1}:
        raise RuntimeError(
            "Test labels must contain binary 0/1 values."
        )

    print("\nArchived test labels:", len(y))
    print("Observed adverse-outcome rate:", y.mean())

    # Locate original model predictions

    pattern = re.compile(
        r"^cell27_(\d+)_(.+)_test_pred\.npy$",
        re.IGNORECASE
    )

    for name in names:

        filename = Path(name).name

        match = pattern.match(filename)

        if not match:
            continue

        n_train = int(match.group(1))

        model = identify_model(
            match.group(2)
        )

        key = (n_train, model)

        if key not in expected_auc:
            continue

        if key in found:
            raise RuntimeError(
                f"Duplicate prediction array for {key}"
            )

        found[key] = name

    print("\nExpected prediction arrays:", len(expected_auc))
    print("Located prediction arrays:", len(found))

    missing = sorted(
        set(expected_auc) - set(found)
    )

    if missing:

        print("\nMISSING ARRAYS:")

        for key in missing:
            print(" -", key)

        print("\nAvailable cell27 test predictions:")

        for name in names:
            if (
                "cell27_" in name
                and "_test_pred.npy" in name
            ):
                print(" -", name)

        raise RuntimeError(
            "The archive does not contain all "
            "12 expected prediction arrays."
        )

    # Recalculate metrics

    print("\n===== RECOMPUTED TEST METRICS =====")

    for key in sorted(expected_auc):

        n_train, model = key

        prediction = np.load(
            io.BytesIO(z.read(found[key])),
            allow_pickle=False
        )

        if prediction.ndim == 2 and prediction.shape[1] == 2:
            prediction = prediction[:, 1]

        if prediction.shape != y.shape:
            raise RuntimeError(
                f"Prediction shape mismatch for {key}: "
                f"{prediction.shape}"
            )

        if not np.isfinite(prediction).all():
            raise RuntimeError(
                f"Non-finite predictions for {key}"
            )

        if (
            (prediction < 0).any()
            or (prediction > 1).any()
        ):
            raise RuntimeError(
                f"Invalid probability range for {key}"
            )

        observed_auc = float(
            roc_auc_score(y, prediction)
        )

        observed_brier = float(
            brier_score_loss(y, prediction)
        )

        reported_auc = expected_auc[key]

        difference = abs(
            observed_auc - reported_auc
        )

        passed = difference <= 0.00001

        results.append({
            "training_n": n_train,
            "model": model,
            "test_n": len(y),
            "recomputed_auc": observed_auc,
            "reported_auc": reported_auc,
            "absolute_difference": difference,
            "recomputed_brier": observed_brier,
            "pass": passed,
            "archived_prediction_file": found[key]
        })

        print(
            f"n={n_train:>8,} | "
            f"{model:<13} | "
            f"recomputed={observed_auc:.6f} | "
            f"reported={reported_auc:.6f} | "
            f"{'PASS' if passed else 'FAIL'}"
        )

# Save results

output = (
    evidence
    / "61_independent_archived_auc_reproduction.csv"
)

with open(
    output,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=list(results[0].keys())
    )

    writer.writeheader()
    writer.writerows(results)

passed_count = sum(
    row["pass"] for row in results
)

print("\n===== STEP 61 SUMMARY =====")

print("Model comparisons:", len(results))
print("ROC-AUC matches:", passed_count)

print("\nSaved:", output)

if passed_count != len(expected_auc):

    raise RuntimeError(
        "One or more original ROC-AUC values "
        "could not be reproduced."
    )

print("\nPASS: All 12 archived ROC-AUC values reproduced.")

print(
    "LIMITATION: Recomputing archived prediction metrics "
    "does not independently reproduce model training."
)

print("\nSTEP 61 COMPLETE")
print("Original files unchanged.")
