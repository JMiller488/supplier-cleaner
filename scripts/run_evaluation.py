"""
run_evaluation.py
-----------------
Loads the synthetic ground truth CSV, scores every pair using cosine
similarity, sweeps thresholds, and plots a precision/recall curve.

Usage:
    python scripts/run_evaluation.py
"""
print("script started")
import csv
import os

import matplotlib.pyplot as plt

from supplier_cleaner.evaluate import NamePair, score_pairs, sweep_thresholds

DEFAULT_THRESHOLD = 0.85


# ---------------------------------------------------------------------------
# CSV loader
# ---------------------------------------------------------------------------


def load_pairs(csv_path: str) -> list[NamePair]:
    """Load labelled name pairs from a CSV file.

    Expects columns: name_a, name_b, match
    The match column should contain 'True' or 'False' as strings.

    Args:
        csv_path: Path to the CSV file produced by generate_synthetic.py.

    Returns:
        List of NamePair objects.
    """
    pairs = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pairs.append(
                NamePair(
                    name_a=row["name_a"],
                    name_b=row["name_b"],
                    match=row["match"] == "True",
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
    csv_path = "data/synthetic_pairs.csv"
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