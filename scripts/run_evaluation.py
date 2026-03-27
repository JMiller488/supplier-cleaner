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

from supplier_cleaner.evaluate import (NamePair,score_pairs,score_pairs_tfidf,sweep_thresholds,)

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


def plot_precision_recall(
    results_by_method: dict[str, list],
    output_path: str,
    default_threshold: float,
) -> None:
    """Plot precision and recall curves for one or more scoring methods.

    Each method gets its own pair of precision/recall lines. A vertical
    line marks the default threshold.

    Args:
        results_by_method: Dict mapping method name to list of ThresholdResult.
        output_path: File path to save the plot image.
        default_threshold: The threshold to highlight on the plot.
    """
    method_colours = {
        "Sentence Transformer": "#378ADD",
        "TF-IDF (char n-grams)": "#E8913A",
    }

    fig, ax = plt.subplots(figsize=(10, 6))

    for method_name, results in results_by_method.items():
        colour = method_colours.get(method_name, "#888888")
        thresholds = [r.threshold for r in results]
        precisions = [r.precision for r in results]
        recalls = [r.recall for r in results]

        ax.plot(
            thresholds, precisions,
            label=f"{method_name} — Precision",
            color=colour,
            linewidth=2,
        )
        ax.plot(
            thresholds, recalls,
            label=f"{method_name} — Recall",
            color=colour,
            linewidth=2,
            linestyle="--",
        )

    ax.axvline(
        x=default_threshold,
        color="#E24B4A",
        linestyle=":",
        linewidth=1.5,
        label=f"Default threshold (τ={default_threshold})",
    )

    ax.set_xlabel("Threshold (τ)", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Precision and Recall vs Threshold — Method Comparison", fontsize=13)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9, loc="center left")
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

    # --- Sentence Transformer scoring ---
    print("Scoring pairs with Sentence Transformer...")
    scored_st = score_pairs(pairs)
    print("  Scoring complete")

    print("Sweeping thresholds (Sentence Transformer)...")
    results_st = sweep_thresholds(scored_st)
    print("  Sweep complete")

    # --- TF-IDF scoring ---
    print("Scoring pairs with TF-IDF...")
    scored_tfidf = score_pairs_tfidf(pairs)
    print("  Scoring complete")

    print("Sweeping thresholds (TF-IDF)...")
    results_tfidf = sweep_thresholds(scored_tfidf)
    print("  Sweep complete")

    # --- Summary table ---
    print(f"\n{'Threshold':>10} {'ST Prec':>10} {'ST Rec':>10} {'TFIDF Prec':>12} {'TFIDF Rec':>10}")
    print("-" * 54)
    for r_st, r_tfidf in zip(results_st, results_tfidf):
        if 0.70 <= r_st.threshold <= 0.95:
            marker = " <--" if abs(r_st.threshold - DEFAULT_THRESHOLD) < 0.001 else ""
            print(
                f"{r_st.threshold:>10.2f}"
                f" {r_st.precision:>10.2f} {r_st.recall:>10.2f}"
                f" {r_tfidf.precision:>12.2f} {r_tfidf.recall:>10.2f}"
                f"{marker}"
            )

    # --- Plot ---
    results_by_method = {
        "Sentence Transformer": results_st,
        "TF-IDF (char n-grams)": results_tfidf,
    }
    plot_precision_recall(results_by_method, plot_path, DEFAULT_THRESHOLD)