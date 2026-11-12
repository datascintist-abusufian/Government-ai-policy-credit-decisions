r"""Create three publication-ready figures from verified Cogent Round 2 aggregates.

Run from the project directory:
    python .\Round2_Manuscript_Results\05_Reproducibility\Code\35_create_missing_figures.py

Outputs: Round2_Manuscript_Results/02_Figures/*.png (600 dpi), *.pdf,
         and three audit CSVs containing the plotted values.

This script plots *previously computed* model outputs; it does not refit models.
It does not modify the source notebook, prediction arrays or manuscript.
"""

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def find_results_root():
    here = Path(__file__).resolve()
    # Correct placement: <root>/05_Reproducibility/Code/35_create_missing_figures.py
    for directory in [here.parent, *here.parents, Path.cwd()]:
        if directory.name == "Round2_Manuscript_Results":
            return directory
        candidate = directory / "Round2_Manuscript_Results"
        if candidate.is_dir():
            return candidate
    raise SystemExit(
        "Round2_Manuscript_Results was not found. Run the script from the "
        "'Cogent Economics & Finance' project directory, or place it in "
        "Round2_Manuscript_Results/05_Reproducibility/Code."
    )


OUT = find_results_root() / "02_Figures"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "legend.fontsize": 9,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def write_csv(filename, headers, rows):
    path = OUT / filename
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)


def save_figure(fig, stem):
    png = OUT / (stem + ".png")
    pdf = OUT / (stem + ".pdf")
    fig.savefig(png, dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("Saved:", png)
    print("Saved:", pdf)


# FIGURE 4. Original 150-predictor, seed-42 test AUCs.
# These are four nested training sizes, NOT independently resampled pipelines.
train_rows = [
    (180000, 0.646062, 0.697660, 0.718945),
    (360000, 0.672210, 0.687550, 0.700299),
    (540000, 0.665383, 0.674609, 0.694318),
    (1304058, 0.699310, 0.675016, 0.646276),
]
write_csv(
    "Figure_04_Source_Training_Size_AUC.csv",
    ["training_n", "logistic_test_auc", "histgb_test_auc", "mlp_test_auc"],
    train_rows,
)
fig, ax = plt.subplots(figsize=(7.2, 4.5))
xs = [row[0] for row in train_rows]
for idx, label, marker in [(1, "Logistic SGD", "o"), (2, "HistGB", "s"), (3, "MLP", "^")]:
    ax.plot(xs, [row[idx] for row in train_rows], marker=marker,
            linewidth=2.0, markersize=6, label=label)
ax.set_xticks(xs, ["180,000", "360,000", "540,000", "1,304,058"])
ax.set_xlabel("Mature training loans (n)")
ax.set_ylabel("Test ROC-AUC")
ax.set_ylim(0.60, 0.76)
ax.set_title("Model discrimination across training sizes")
ax.grid(axis="y", alpha=0.3)
ax.legend(frameon=False)
fig.text(0.12, 0.015,
         "Seed 42; original 150-feature specification; selected 2019 test cohort (n = 41,145).",
         fontsize=8)
fig.subplots_adjust(bottom=0.20)
save_figure(fig, "Figure_04_Training_Size_Test_AUC")


# FIGURE 5. Recalibrated HistGB reliability, 10%-wide score bins.
# The top-end 0.5-1.0 bins contain just three additional observations and
# are omitted from the displayed series because their bin means were not
# validated for this script. The five shown bins contain 41,142 of 41,145 loans.
cal_rows = [
    ("0-10%", 26759, 0.055636, 0.100265),
    ("10-20%", 12649, 0.135146, 0.220571),
    ("20-30%", 1511, 0.233370, 0.309729),
    ("30-40%", 199, 0.330410, 0.361809),
    ("40-50%", 24, 0.435292, 0.500000),
]
assert sum(row[1] for row in cal_rows) == 41142
write_csv(
    "Figure_05_Source_Calibration_Bins.csv",
    ["risk_band", "n", "mean_predicted", "observed_rate"],
    cal_rows,
)
fig, ax = plt.subplots(figsize=(6.3, 5.4))
ax.plot([0, 0.53], [0, 0.53], linestyle="--", linewidth=1.6,
        label="Perfect calibration")
ax.plot([row[2] for row in cal_rows], [row[3] for row in cal_rows],
        marker="o", markersize=6, linewidth=2, label="Recalibrated HistGB")
for i, (band, n, p, y) in enumerate(cal_rows):
    offsets = [(8, 7), (7, 7), (7, -17), (7, 7), (-100, 7)]
    ax.annotate(f"n={n:,}", xy=(p, y), xytext=offsets[i],
                textcoords="offset points", fontsize=8)
ax.set_xlim(0, 0.54)
ax.set_ylim(0, 0.56)
ax.set_xlabel("Mean predicted adverse-outcome probability")
ax.set_ylabel("Observed adverse-outcome proportion")
ax.set_title("Test-set reliability after validation recalibration")
ax.grid(alpha=0.25)
ax.legend(loc="upper left", frameon=False)
fig.text(0.13, 0.020,
         "Five plotted bins: n = 41,142; three additional loans score >= 0.50 (not shown).",
         fontsize=8)
fig.subplots_adjust(bottom=0.16)
save_figure(fig, "Figure_05_Recalibrated_Test_Reliability")


# FIGURE 6. Decision-rule trade-off for the validation-selected thresholds.
# Flagged loan proportion is NOT actual human review, and costs are hypothetical.
threshold_rows = [
    ("5:1", 0.172966, 0.077458, 0.843485),
    ("10:1", 0.096162, 0.375307, 0.413610),
    ("20:1", 0.032698, 0.850553, 0.048963),
]
write_csv(
    "Figure_06_Source_Threshold_Tradeoff.csv",
    ["fn_to_fp_cost_ratio", "validation_threshold", "test_flag_rate", "test_fnr"],
    threshold_rows,
)
fig, ax = plt.subplots(figsize=(6.8, 4.9))
flags = [100 * row[2] for row in threshold_rows]
misses = [100 * row[3] for row in threshold_rows]
ax.plot(flags, misses, marker="o", markersize=7, linewidth=2)
for i, (ratio, threshold, flag, fnr) in enumerate(threshold_rows):
    offsets = [(9, 8), (9, 9), (-118, 10)]
    ax.annotate(f"{ratio}  (t={threshold:.4f})", xy=(100 * flag, 100 * fnr),
                xytext=offsets[i], textcoords="offset points", fontsize=9)
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.set_xlabel("Test loans flagged for analytical review (%)")
ax.set_ylabel("Test false-negative rate (%)")
ax.set_title("Threshold-policy trade-off under illustrative costs")
ax.grid(alpha=0.3)
fig.text(0.11, 0.015,
         "Selected test cohort (n = 41,145); no observed human-review intervention or cost.",
         fontsize=8)
fig.subplots_adjust(bottom=0.18)
save_figure(fig, "Figure_06_Flag_Rate_vs_False_Negatives")

print("\nComplete. Created 3 PNGs (600 dpi), 3 vector PDFs and 3 source CSVs in:")
print(OUT)
