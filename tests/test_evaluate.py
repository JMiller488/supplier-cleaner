"""
test_evaluate.py
----------------
Unit tests for evaluate.py.

Tests focus on precision_recall_at_threshold, which contains the core
TP/FP/FN logic. Inputs are constructed by hand so expected outputs
can be verified without running the embedding model.
"""

import pytest

from supplier_cleaner.evaluate import NamePair, ThresholdResult, precision_recall_at_threshold


def make_scored_pairs(similarities: list[float], matches: list[bool]):
    """Helper to build scored pairs without needing real embeddings.

    Creates a NamePair for each (similarity, match) combination and
    pairs it with its similarity score, mimicking the output of score_pairs().

    Args:
        similarities: List of cosine similarity scores.
        matches: List of ground truth match labels.

    Returns:
        List of (NamePair, float) tuples.
    """
    return [
        (NamePair(name_a=f"name_{i}_a", name_b=f"name_{i}_b", match=match), sim)
        for i, (sim, match) in enumerate(zip(similarities, matches))
    ]


def test_perfect_precision_and_recall():
    """All predictions correct: high similarity for matches, low for non-matches.

    Setup:
        - 3 true matches, all with similarity above threshold
        - 3 true non-matches, all with similarity below threshold

    Expected: precision=1.0, recall=1.0
    """
    scored = make_scored_pairs(
        similarities=[0.90, 0.92, 0.95, 0.40, 0.30, 0.20],
        matches=    [True, True, True, False, False, False],
    )
    result = precision_recall_at_threshold(scored, threshold=0.85)
    assert result.precision == pytest.approx(1.0)
    assert result.recall == pytest.approx(1.0)


def test_false_positives_reduce_precision():
    """A non-match pair scores above threshold, creating a false positive.

    Setup:
        - 1 true match above threshold (TP=1)
        - 1 true non-match above threshold (FP=1)
        - 1 true match below threshold (FN=1)

    Expected:
        precision = TP / (TP + FP) = 1 / (1 + 1) = 0.5
        recall    = TP / (TP + FN) = 1 / (1 + 1) = 0.5
    """
    scored = make_scored_pairs(
        similarities=[0.90, 0.90, 0.70],
        matches=    [True, False, True],
    )
    result = precision_recall_at_threshold(scored, threshold=0.85)
    assert result.precision == pytest.approx(0.5)
    assert result.recall == pytest.approx(0.5)


def test_no_predictions_made():
    """All similarities below threshold — model predicts no matches at all.

    Setup:
        - 2 true matches, both below threshold (FN=2)
        - No predictions made (TP=0, FP=0)

    Expected:
        precision = 1.0 (by convention — no predictions means no false claims)
        recall    = 0.0 (missed every true match)
    """
    scored = make_scored_pairs(
        similarities=[0.60, 0.70],
        matches=    [True, True],
    )
    result = precision_recall_at_threshold(scored, threshold=0.85)
    assert result.precision == pytest.approx(1.0)
    assert result.recall == pytest.approx(0.0)