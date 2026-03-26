"""
evaluate.py
-----------
Evaluation framework for the supplier name deduplication pipeline.

Provides functions to:
  - Score a list of labelled name pairs using cosine similarity
  - Compute precision and recall at a given threshold
  - Sweep across a range of thresholds to produce a precision/recall curve
"""

from dataclasses import dataclass

import numpy as np
from sentence_transformers import SentenceTransformer, util

DEFAULT_MODEL = "all-MiniLM-L6-v2"


@dataclass
class NamePair:
    """A labelled pair of supplier names.

    Attributes:
        name_a: First supplier name (preprocessed).
        name_b: Second supplier name (preprocessed).
        match: True if the two names refer to the same supplier.
    """

    name_a: str
    name_b: str
    match: bool


@dataclass
class ThresholdResult:
    """Precision and recall at a single threshold value.

    Attributes:
        threshold: The cosine similarity threshold used.
        precision: TP / (TP + FP). How many predicted matches were correct.
        recall: TP / (TP + FN). How many true matches were found.
    """

    threshold: float
    precision: float
    recall: float


def score_pairs(
    pairs: list[NamePair],
    model_name: str = DEFAULT_MODEL,
) -> list[tuple[NamePair, float]]:
    """Compute cosine similarity for each name pair.

    Encodes all names in a single batch for efficiency, then looks up
    the similarity for each pair.

    Args:
        pairs: List of labelled NamePair objects.
        model_name: HuggingFace model name for SentenceTransformer.

    Returns:
        List of (NamePair, similarity_score) tuples.
    """
    model = SentenceTransformer(model_name)

    # Collect every unique name across all pairs to avoid re-encoding duplicates
    all_names = list({name for pair in pairs for name in (pair.name_a, pair.name_b)})
    embeddings = model.encode(all_names, convert_to_tensor=True)
    name_to_idx = {name: idx for idx, name in enumerate(all_names)}

    scored = []
    for pair in pairs:
        idx_a = name_to_idx[pair.name_a]
        idx_b = name_to_idx[pair.name_b]
        similarity = float(util.cos_sim(embeddings[idx_a], embeddings[idx_b]))
        scored.append((pair, similarity))

    return scored


def precision_recall_at_threshold(
    scored_pairs: list[tuple[NamePair, float]],
    threshold: float,
) -> ThresholdResult:
    """Compute precision and recall at a single threshold.

    A pair is predicted as a match if its cosine similarity >= threshold.

    Args:
        scored_pairs: Output of score_pairs().
        threshold: Cosine similarity cutoff for a positive prediction.

    Returns:
        ThresholdResult with precision and recall values.
    """
    tp = fp = fn = 0

    for pair, similarity in scored_pairs:
        predicted_match = similarity >= threshold
        if predicted_match and pair.match:
            tp += 1
        elif predicted_match and not pair.match:
            fp += 1
        elif not predicted_match and pair.match:
            fn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    return ThresholdResult(threshold=threshold, precision=precision, recall=recall)


def sweep_thresholds(
    scored_pairs: list[tuple[NamePair, float]],
    thresholds: list[float] | None = None,
) -> list[ThresholdResult]:
    """Compute precision and recall across a range of thresholds.

    Args:
        scored_pairs: Output of score_pairs().
        thresholds: List of threshold values to evaluate. Defaults to
            101 evenly spaced values from 0.0 to 1.0.

    Returns:
        List of ThresholdResult objects, one per threshold value.
    """
    if thresholds is None:
        thresholds = list(np.linspace(0.0, 1.0, 101))

    return [
        precision_recall_at_threshold(scored_pairs, t)
        for t in thresholds
    ]