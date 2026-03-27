"""
run_evaluation.py
-----------------
Loads the synthetic ground truth CSV, scores every pair using cosine
similarity, sweeps thresholds, and plots a precision/recall curve.

Usage:
    python scripts/run_evaluation.py
"""
import csv
import os
from itertools import combinations

import matplotlib.pyplot as plt

from supplier_cleaner.evaluate import NamePair, score_pairs, sweep_thresholds

DEFAULT_THRESHOLD = 0.85

print("script started")


# ---------------------------------------------------------------------------
# CSV loader
# ---------------------------------------------------------------------------


def load_pairs(csv_path: str) -> list[NamePair]:
    """Load supplier names and generate all pairwise combinations.

    Expects columns: supplier_name, true_name
    Two names are a match if they share the same true_name.

    For 250 names this produces ~31k pairs — large enough for robust
    precision/recall estimation, with a realistic class imbalance
    (most pairs are non-matches, just like real data).

    Args:
        csv_path: Path to the CSV file with supplier_name and true_name columns.

    Returns:
        List of NamePair objects for every pairwise combination.
    """
    rows = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    pairs = []
    for (row_a, row_b) in combinations(rows, 2):
        pairs.append(
            NamePair(
                name_a=row_a["supplier_name"],
                name_b=row_b["supplier_name"],
                match=row_a["true_name"] == row_b["true_name"],
            )
        )
    return pairs


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_precision_recall(results, output_path: str, default_threshold: float) -> None:
    """Plot precision and recall curves across thresholds and save to file.

    Draws both curves on the same axes, adds a vertical line at the
    default threshold, and annotates the precision and recall values
    at that threshold.

    Args:
        results: List of ThresholdResult objects from sweep_thresholds().
        output_path: File path to save the plot image.
        default_threshold: The threshold to highlight on the plot.
    """
    thresholds = [r.threshold for r in results]
    precisions = [r.precision for r in results]
    recalls = [r.recall for r in results]

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(thresholds, precisions, label="Precision", color="#378ADD", linewidth=2)
    ax.plot(thresholds, recalls, label="Recall", color="#1D9E75", linewidth=2)

    # Find the result closest to the default threshold
    closest = min(results, key=lambda r: abs(r.threshold - default_threshold))

    # Vertical line at default threshold
    ax.axvline(
        x=closest.threshold,
        color="#E24B4A",
        linestyle="--",
        linewidth=1.5,
        label=f"Default threshold (τ={default_threshold})",
    )

    # Annotate precision and recall at default threshold
    ax.annotate(
        f"P={closest.precision:.2f}",
        xy=(closest.threshold, closest.precision),
        xytext=(closest.threshold + 0.03, closest.precision - 0.08),
        color="#378ADD",
        fontsize=10,
    )
    ax.annotate(
        f"R={closest.recall:.2f}",
        xy=(closest.threshold, closest.recall),
        xytext=(closest.threshold + 0.03, closest.recall + 0.04),
        color="#1D9E75",
        fontsize=10,
    )

    ax.set_xlabel("Threshold (τ)", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Precision and Recall vs Threshold", fontsize=13)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150)
    print(f"Plot saved to {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    csv_path = "data/supplier_cleaning_synthetic_250_exact.csv"
    plot_path = "data/precision_recall_curve.png"

    print("Loading pairs...")
    pairs = load_pairs(csv_path)
    print(f"  {len(pairs)} pairs loaded")

    print("Scoring pairs (this may take a moment)...")
    scored = score_pairs(pairs)
    print("  Scoring complete")

    print("Sweeping thresholds...")
    results = sweep_thresholds(scored)
    print("  Sweep complete")

    # Print a summary table around the default threshold
    print(f"\n{'Threshold':>10} {'Precision':>10} {'Recall':>10}")
    print("-" * 32)
    for r in results:
        if 0.70 <= r.threshold <= 0.95:
            marker = " <--" if abs(r.threshold - DEFAULT_THRESHOLD) < 0.001 else ""
            print(f"{r.threshold:>10.2f} {r.precision:>10.2f} {r.recall:>10.2f}{marker}")

    plot_precision_recall(results, plot_path, DEFAULT_THRESHOLD)