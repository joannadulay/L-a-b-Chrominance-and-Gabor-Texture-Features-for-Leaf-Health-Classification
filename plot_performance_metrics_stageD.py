"""
plot_performance_metrics_stageD.py

Generates a performance metrics bar chart for the Stage D GBM model,
styled to match the reference figure (bordered percentage labels above
each bar, boxed legend, two-line title) but in blue instead of red,
using Times New Roman.

Run this locally (Windows/your machine) so it picks up the actual
Times New Roman font — Linux sandboxes typically don't have it
installed, so a remote-generated preview would silently substitute a
fallback serif font.

Usage:
    python plot_performance_metrics_stageD.py
Output:
    figures/stageD_performance_metrics.png
    figures/stageD_performance_metrics.pdf
"""

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib as mpl

# ---- Set Times New Roman globally (falls back to default serif if not installed) ----
mpl.rcParams["font.family"] = "Times New Roman"

# ---- Metrics from the Stage D final test run ----
# Accuracy from accuracy_score(); Precision/Recall/F1 from macro avg row
# of classification_report() (macro avg precision=0.9655, recall=0.9645,
# f1-score=0.9645 per your printed report).
METRICS = {
    "Accuracy": 0.9640,
    "Precision": 0.9655,
    "Recall": 0.9645,
    "F1-Score": 0.9645,
}

BAR_COLOR = "#1F4E9C"      # blue, swap for any hex you prefer
LEGEND_LABEL = "Stage D GBM Model"
TITLE = "Performance Metrics of Stage D\nGradient Boosting Machine Model"
CAPTION = "Fig. X Performance Evaluation of Stage D Model"

OUTPUT_DIR = Path("figures")
OUTPUT_BASENAME = "stageD_performance_metrics"


def plot_metrics():
    labels = list(METRICS.keys())
    values = list(METRICS.values())

    fig, ax = plt.subplots(figsize=(6, 5))

    bars = ax.bar(labels, values, color=BAR_COLOR, width=0.55, label=LEGEND_LABEL)

    # Y-axis matches the reference figure's zoomed-in range so differences
    # between near-ceiling metrics are visible rather than flattened.
    ax.set_ylim(0.80, 1.04)
    ax.set_yticks([0.80, 0.84, 0.88, 0.92, 0.96, 1.00, 1.04])
    ax.set_ylabel("Performance Score", fontsize=12)
    ax.set_xlabel("Metrics", fontsize=12)
    ax.set_title(TITLE, fontsize=13, fontweight="bold")

    # Percentage labels in a bordered box above each bar, matching reference style
    for bar, val in zip(bars, values):
        ax.annotate(
            f"{val * 100:.2f}%",
            xy=(bar.get_x() + bar.get_width() / 2, val),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center", va="bottom",
            fontsize=11,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="black", linewidth=0.8),
        )

    ax.legend(loc="upper right", frameon=True, edgecolor="black", fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="-", alpha=0.15)

    fig.text(0.5, -0.04, CAPTION, ha="center", fontsize=11)

    plt.tight_layout()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    png_path = OUTPUT_DIR / f"{OUTPUT_BASENAME}.png"
    pdf_path = OUTPUT_DIR / f"{OUTPUT_BASENAME}.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    print(f"Saved:\n  {png_path}\n  {pdf_path}")

    plt.show()


if __name__ == "__main__":
    plot_metrics()
