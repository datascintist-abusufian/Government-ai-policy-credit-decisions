import argparse
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
import pandas as pd


parser = argparse.ArgumentParser()
parser.add_argument("--root", required=True)
args = parser.parse_args()

root = Path(args.root)

dataset = root / "P2P_Macro_Data.dta"

tables = (
    root
    / "Round2_Manuscript_Results"
    / "01_Tables"
)

evidence = (
    root
    / "Round2_Manuscript_Results"
    / "05_Reproducibility"
    / "Evidence_Checks"
)

tables.mkdir(parents=True, exist_ok=True)
evidence.mkdir(parents=True, exist_ok=True)

if not dataset.exists():
    raise FileNotFoundError(dataset)

print("=" * 70)
print("STEP 59: SELECTED VS EXCLUDED POPULATION BALANCE")
print("=" * 70)

# ------------------------------------------------------------
# 1. Identify available source variables
# ------------------------------------------------------------

numeric_candidates = [
    "loan_amnt",
    "annual_inc",
    "dti",
    "fico_range_low",
    "fico_range_high",
    "int_rate",
    "installment"
]

categorical_candidates = [
    "grade",
    "term",
    "purpose",
    "home_ownership",
    "verification_status"
]

with pd.read_stata(
    dataset,
    iterator=True,
    convert_categoricals=False
) as reader:
    available = set(reader.read(1).columns)

numeric_features = [
    x for x in numeric_candidates
    if x in available
]

categorical_features = [
    x for x in categorical_candidates
    if x in available
]

print("\nAvailable numeric features:")
print(numeric_features)

print("\nAvailable categorical features:")
print(categorical_features)

if not numeric_features and not categorical_features:
    raise RuntimeError(
        "No predeclared comparison variables are available."
    )

# ------------------------------------------------------------
# 2. Define original chronological cohorts
# ------------------------------------------------------------

windows = {
    "validation": (
        pd.Timestamp("2018-07-01"),
        pd.Timestamp("2019-04-01")
    ),
    "test": (
        pd.Timestamp("2019-04-01"),
        pd.Timestamp("2020-01-01")
    )
}

terminal = {
    "Fully Paid",
    "Charged Off",
    "Default"
}

columns = (
    ["issue_date", "loan_status"]
    + numeric_features
    + categorical_features
)

# Accumulate descriptive statistics without retaining
# the full 2.7-million-row dataset in memory.

sample_counts = Counter()

numeric_stats = defaultdict(
    lambda: {
        "n": 0,
        "valid": 0,
        "sum": 0.0,
        "sum_sq": 0.0
    }
)

categorical_counts = Counter()

total_scanned = 0

# ------------------------------------------------------------
# 3. Process source dataset
# ------------------------------------------------------------

iterator = pd.read_stata(
    dataset,
    columns=columns,
    chunksize=100000,
    iterator=True,
    convert_categoricals=True
)

for chunk in iterator:

    total_scanned += len(chunk)

    issue = pd.to_datetime(
        chunk["issue_date"],
        errors="coerce"
    )

    status = (
        chunk["loan_status"]
        .astype("string")
        .str.strip()
    )

    for cohort, (start, end) in windows.items():

        in_window = (
            (issue >= start)
            & (issue < end)
        )

        if not in_window.any():
            continue

        subset = chunk.loc[in_window]

        selected = status.loc[in_window].isin(
            terminal
        )

        for group in ["selected", "excluded"]:

            group_mask = (
                selected
                if group == "selected"
                else ~selected
            )

            data = subset.loc[group_mask]

            n = len(data)

            sample_counts[(cohort, group)] += n

            if n == 0:
                continue

            # --------------------------------------------
            # Continuous-variable statistics
            # --------------------------------------------

            for feature in numeric_features:

                values = pd.to_numeric(
                    data[feature],
                    errors="coerce"
                ).to_numpy(dtype=float)

                values = values[
                    np.isfinite(values)
                ]

                key = (cohort, group, feature)

                stat = numeric_stats[key]

                stat["n"] += n
                stat["valid"] += len(values)

                if len(values):

                    stat["sum"] += float(
                        values.sum(dtype=np.float64)
                    )

                    stat["sum_sq"] += float(
                        np.square(values).sum(
                            dtype=np.float64
                        )
                    )

            # --------------------------------------------
            # Categorical distributions
            # --------------------------------------------

            for feature in categorical_features:

                values = (
                    data[feature]
                    .astype("string")
                    .fillna("<MISSING>")
                    .value_counts(dropna=False)
                )

                for category, count in values.items():

                    categorical_counts[
                        (
                            cohort,
                            group,
                            feature,
                            str(category)
                        )
                    ] += int(count)

    if total_scanned % 500000 < len(chunk):

        print(
            f"Rows examined: {total_scanned:,}"
        )

print(
    "\nTotal source rows:",
    f"{total_scanned:,}"
)

if total_scanned != 2703430:
    raise RuntimeError(
        "Unexpected source dataset size."
    )

# ------------------------------------------------------------
# 4. Verify original cohort sizes
# ------------------------------------------------------------

expected = {
    ("validation", "selected"): 27374,
    ("validation", "excluded"): 343620,
    ("test", "selected"): 41145,
    ("test", "excluded"): 360024
}

for key, expected_n in expected.items():

    observed = sample_counts[key]

    if observed != expected_n:

        raise RuntimeError(
            f"Count mismatch for {key}: "
            f"expected {expected_n}, observed {observed}"
        )

print("\nPASS: All original cohort counts reproduced.")

# ------------------------------------------------------------
# 5. Calculate means, standard deviations and SMDs
# ------------------------------------------------------------

def summarise(stat):

    n = stat["valid"]

    if n == 0:
        return np.nan, np.nan

    mean = stat["sum"] / n

    variance = max(
        0.0,
        stat["sum_sq"] / n - mean ** 2
    )

    return mean, np.sqrt(variance)


numeric_rows = []

for cohort in windows:

    for feature in numeric_features:

        a = numeric_stats[
            (cohort, "selected", feature)
        ]

        b = numeric_stats[
            (cohort, "excluded", feature)
        ]

        mean_a, sd_a = summarise(a)
        mean_b, sd_b = summarise(b)

        pooled_sd = np.sqrt(
            (sd_a ** 2 + sd_b ** 2) / 2
        )

        smd = (
            (mean_a - mean_b) / pooled_sd
            if np.isfinite(pooled_sd)
            and pooled_sd > 0
            else np.nan
        )

        numeric_rows.append({

            "cohort": cohort,

            "feature": feature,

            "selected_n": a["valid"],

            "excluded_n": b["valid"],

            "selected_mean": mean_a,

            "excluded_mean": mean_b,

            "selected_sd": sd_a,

            "excluded_sd": sd_b,

            "standardized_mean_difference": smd,

            "absolute_smd": abs(smd),

            "selected_missing_percent": (
                100 * (a["n"] - a["valid"]) / a["n"]
                if a["n"] else np.nan
            ),

            "excluded_missing_percent": (
                100 * (b["n"] - b["valid"]) / b["n"]
                if b["n"] else np.nan
            )
        })

numeric_df = pd.DataFrame(numeric_rows)

# ------------------------------------------------------------
# 6. Calculate categorical representation differences
# ------------------------------------------------------------

categorical_rows = []

for cohort in windows:

    n_selected = sample_counts[
        (cohort, "selected")
    ]

    n_excluded = sample_counts[
        (cohort, "excluded")
    ]

    for feature in categorical_features:

        categories = {
            category
            for (
                c,
                group,
                f,
                category
            ) in categorical_counts
            if c == cohort and f == feature
        }

        for category in sorted(categories):

            a = categorical_counts[
                (
                    cohort,
                    "selected",
                    feature,
                    category
                )
            ]

            b = categorical_counts[
                (
                    cohort,
                    "excluded",
                    feature,
                    category
                )
            ]

            selected_share = (
                a / n_selected
            )

            excluded_share = (
                b / n_excluded
            )

            categorical_rows.append({

                "cohort": cohort,

                "feature": feature,

                "category": category,

                "selected_count": a,

                "excluded_count": b,

                "selected_percent": (
                    100 * selected_share
                ),

                "excluded_percent": (
                    100 * excluded_share
                ),

                "percentage_point_difference": (
                    100 * (
                        selected_share
                        - excluded_share
                    )
                )
            })

categorical_df = pd.DataFrame(
    categorical_rows
)

# ------------------------------------------------------------
# 7. Save results
# ------------------------------------------------------------

numeric_path = (
    tables
    / "59_selected_vs_excluded_numeric_balance.csv"
)

categorical_path = (
    tables
    / "59_selected_vs_excluded_categorical_balance.csv"
)

report_path = (
    evidence
    / "59_population_balance_summary.txt"
)

numeric_df.to_csv(
    numeric_path,
    index=False
)

categorical_df.to_csv(
    categorical_path,
    index=False
)

# ------------------------------------------------------------
# 8. Display summary
# ------------------------------------------------------------

print("\n===== NUMERIC POPULATION DIFFERENCES =====")

print(
    numeric_df[
        [
            "cohort",
            "feature",
            "selected_mean",
            "excluded_mean",
            "standardized_mean_difference",
            "selected_missing_percent",
            "excluded_missing_percent"
        ]
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

print("\n===== LARGEST CATEGORICAL DIFFERENCES =====")

largest = (
    categorical_df
    .assign(
        absolute_difference=lambda d:
        d["percentage_point_difference"].abs()
    )
    .sort_values(
        "absolute_difference",
        ascending=False
    )
    .head(15)
)

print(
    largest[
        [
            "cohort",
            "feature",
            "category",
            "selected_percent",
            "excluded_percent",
            "percentage_point_difference"
        ]
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

with open(
    report_path,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "STEP 59 POPULATION REPRESENTATIVENESS AUDIT\n\n"
    )

    f.write(
        "Selected vs excluded loan characteristics.\n"
        "SMD compares group means using their pooled "
        "standard deviation.\n\n"
    )

    f.write(
        numeric_df.to_string(index=False)
    )

    f.write("\n\nCATEGORICAL DIFFERENCES\n")

    f.write(
        categorical_df.to_string(index=False)
    )

    f.write(
        "\n\nLIMITATIONS\n"
        "These are unadjusted descriptive comparisons.\n"
        "They do not adjust for origination month.\n"
        "They do not establish causal selection effects.\n"
        "They do not recover missing future outcomes.\n"
        "They cannot produce an unbiased estimate of "
        "population-wide default risk.\n"
        "Some descriptive predictors may not have "
        "been available at an application decision.\n"
    )

print("\n===== SAVED FILES =====")

print(numeric_path)
print(categorical_path)
print(report_path)

print("\nSTEP 59 COMPLETE")

print(
    "Original dataset and experimental results unchanged."
)
