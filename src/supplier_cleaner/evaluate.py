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
from rapidfuzz import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
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
    """Precision, recall, and F1 at a single threshold value.

    Attributes:
        threshold: The cosine similarity threshold used.
        precision: TP / (TP + FP). How many predicted matches were correct.
        recall: TP / (TP + FN). How many true matches were found.
        f1: Harmonic mean of precision and recall.
    """

    threshold: float
    precision: float
    recall: float
    f1: float


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

def score_pairs_tfidf(
    pairs: list[NamePair],
    ngram_range: tuple[int, int] = (2, 4),
    analyzer: str = "char_wb",
) -> list[tuple[NamePair, float]]:
    """Compute TF-IDF cosine similarity for each name pair.

    Fits a TfidfVectorizer on all unique names using character n-grams,
    then computes cosine similarity between the resulting sparse vectors.

    This provides a simple, interpretable baseline — similarity is driven
    entirely by shared character sequences, with no learned representations.

    Args:
        pairs: List of labelled NamePair objects.
        ngram_range: Min and max character n-gram lengths. (2, 4) generates
            bigrams, trigrams, and 4-grams.
        analyzer: 'char_wb' generates n-grams within word boundaries only.
            'char' ignores word boundaries.

    Returns:
        List of (NamePair, similarity_score) tuples.
    """
    all_names = list({name for pair in pairs for name in (pair.name_a, pair.name_b)})

    vectorizer = TfidfVectorizer(analyzer=analyzer, ngram_range=ngram_range)
    tfidf_matrix = vectorizer.fit_transform(all_names)
    name_to_idx = {name: idx for idx, name in enumerate(all_names)}

    scored = []
    for pair in pairs:
        idx_a = name_to_idx[pair.name_a]
        idx_b = name_to_idx[pair.name_b]
        similarity = float(cosine_similarity(tfidf_matrix[idx_a], tfidf_matrix[idx_b])[0,0])
        scored.append((pair, similarity))

    return scored

def score_pairs_levenshtein(
    pairs: list[NamePair],
) -> list[tuple[NamePair, float]]:
    """Compute normalised Levenshtein similarity for each name pair.

    Uses rapidfuzz's ratio function, which returns a value from 0 to 100
    representing the normalised edit distance similarity. This is divided
    by 100 to produce a score in [0, 1].

    This is the simplest possible string similarity method — it counts
    the minimum number of single-character edits (insertions, deletions,
    substitutions) needed to transform one string into the other, then
    normalises by string length.

    Args:
        pairs: List of labelled NamePair objects.

    Returns:
        List of (NamePair, similarity_score) tuples.
    """
    scored = []
    for pair in pairs:
        similarity = fuzz.ratio(pair.name_a, pair.name_b) / 100
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
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return ThresholdResult(threshold=threshold, precision=precision, recall=recall, f1=f1)


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