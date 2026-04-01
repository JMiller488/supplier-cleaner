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

from supplier_cleaner.evaluate import (NamePair,score_pairs,score_pairs_tfidf,score_pairs_levenshtein,sweep_thresholds,)

DEFAULT_THRESHOLD = 0.85


def find_optimal_threshold(results: list) -> object:
    """Return the ThresholdResult with the highest F1 score."""
    return max(results, key=lambda r: r.f1)

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
    optimal_by_method: dict[str, object],
) -> None:
    """Plot precision/recall curves and F1 curves for each scoring method.

    Left subplot: precision (solid) and recall (dashed) vs threshold.
    Right subplot: F1 vs threshold, with a marker at each method's optimal threshold.

    Args:
        results_by_method: Dict mapping method name to list of ThresholdResult.
        output_path: File path to save the plot image.
        optimal_by_method: Dict mapping method name to its optimal ThresholdResult.
    """
    method_colours = {
        "Sentence Transformer": "#378ADD",
        "TF-IDF (char n-grams)": "#E8913A",
        "Levenshtein": "#2ECC71",
    }

    fig, (ax_pr, ax_f1) = plt.subplots(1, 2, figsize=(14, 6))

    for method_name, results in results_by_method.items():
        colour = method_colours.get(method_name, "#888888")
        thresholds = [r.threshold for r in results]
        precisions = [r.precision for r in results]
        recalls = [r.recall for r in results]
        f1s = [r.f1 for r in results]

        ax_pr.plot(thresholds, precisions, label=f"{method_name} — Precision", color=colour, linewidth=2)
        ax_pr.plot(thresholds, recalls, label=f"{method_name} — Recall", color=colour, linewidth=2, linestyle="--")

        ax_f1.plot(thresholds, f1s, label=method_name, color=colour, linewidth=2)

        opt = optimal_by_method[method_name]
        ax_f1.plot(opt.threshold, opt.f1, marker="x", markersize=8, markeredgewidth=2, color=colour)
        ax_f1.axvline(x=opt.threshold, color=colour, linestyle=":", linewidth=1, alpha=0.5)

    ax_pr.set_xlabel("Threshold (τ)", fontsize=12)
    ax_pr.set_ylabel("Score", fontsize=12)
    ax_pr.set_title("Precision and Recall vs Threshold", fontsize=13)
    ax_pr.set_xlim(0, 1)
    ax_pr.set_ylim(0, 1.05)
    ax_pr.legend(fontsize=8, loc="center left")
    ax_pr.grid(True, alpha=0.3)

    ax_f1.set_xlabel("Threshold (τ)", fontsize=12)
    ax_f1.set_ylabel("F1 Score", fontsize=12)
    ax_f1.set_title("F1 vs Threshold  (★ = optimal)", fontsize=13)
    ax_f1.set_xlim(0, 1)
    ax_f1.set_ylim(0, 1.05)
    ax_f1.legend(fontsize=9)
    ax_f1.grid(True, alpha=0.3)

    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150)
    print(f"Plot saved to {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    csv_path = "analysis/data/supplier_cleaning_synthetic_250_exact.csv"
    plot_path = "analysis/results/precision_recall_curve.png"

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

    # --- Levenshtein scoring ---
    print("Scoring pairs with Levenshtein...")
    scored_lev = score_pairs_levenshtein(pairs)
    print("  Scoring complete")

    print("Sweeping thresholds (Levenshtein)...")
    results_lev = sweep_thresholds(scored_lev)
    print("  Sweep complete")

    # --- Optimal thresholds ---
    opt_st = find_optimal_threshold(results_st)
    opt_tfidf = find_optimal_threshold(results_tfidf)
    opt_lev = find_optimal_threshold(results_lev)

    # --- Summary table ---
    print(f"\n{'Method':<28} {'Optimal τ':>10} {'Precision':>10} {'Recall':>10} {'F1':>8}")
    print("-" * 68)
    for label, opt in [
        ("Sentence Transformer", opt_st),
        ("TF-IDF (char n-grams)", opt_tfidf),
        ("Levenshtein", opt_lev),
    ]:
        print(f"{label:<28} {opt.threshold:>10.2f} {opt.precision:>10.2f} {opt.recall:>10.2f} {opt.f1:>8.2f}")

    # --- Plot ---
    results_by_method = {
        "Sentence Transformer": results_st,
        "TF-IDF (char n-grams)": results_tfidf,
        "Levenshtein": results_lev,
    }
    optimal_by_method = {
        "Sentence Transformer": opt_st,
        "TF-IDF (char n-grams)": opt_tfidf,
        "Levenshtein": opt_lev,
    }
    plot_precision_recall(results_by_method, plot_path, optimal_by_method)