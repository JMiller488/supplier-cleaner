"""
grouping.py
-----------
Groups similar supplier names using sentence embeddings and
cosine similarity. Implements complete linkage hierarchical
clustering to produce strict, transitive-free groupings where
every pair of suppliers within a group directly exceeds the
similarity threshold.
"""

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sentence_transformers import SentenceTransformer, util


DEFAULT_MODEL = "all-MiniLM-L6-v2"
DEFAULT_THRESHOLD = 0.69


def group_suppliers(
    names: list[str],
    threshold: float = DEFAULT_THRESHOLD,
    model_name: str = DEFAULT_MODEL,
) -> dict[str, str]:
    """Group similar supplier names using complete linkage hierarchical clustering.

    Two suppliers are placed in the same group if and only if every pair of
    suppliers within that group has a cosine similarity >= threshold. This avoids
    the transitive grouping problem of connected-components approaches, where A
    and C could be grouped via B even if A and C are dissimilar.

    Args:
        names: List of preprocessed supplier name strings.
        threshold: Minimum cosine similarity required between every pair in a group.
        model_name: HuggingFace model name for SentenceTransformer.

    Returns:
        Dict mapping each input name to its canonical group name (alphabetically
        first within the group).
    """
    if len(names) == 1:
        return {names[0]: names[0]}

    model = SentenceTransformer(model_name)
    embeddings = model.encode(names, convert_to_tensor=False)

    cos_sim_matrix = util.cos_sim(embeddings, embeddings).numpy()

    # Convert similarity to distance; use float64 and symmetrise to guard
    # against the small asymmetries introduced by float32 sentence-transformer output.
    dist_matrix = np.clip(1.0 - cos_sim_matrix.astype(np.float64), 0.0, 2.0)
    dist_matrix = (dist_matrix + dist_matrix.T) / 2.0
    np.fill_diagonal(dist_matrix, 0.0)

    condensed = squareform(dist_matrix)
    Z = linkage(condensed, method="complete")

    # In distance space, the cutoff is 1 - threshold.
    # fcluster groups items whose complete-linkage distance <= dist_threshold,
    # which means every pairwise similarity within the group >= threshold.
    dist_threshold = 1.0 - threshold
    labels = fcluster(Z, t=dist_threshold, criterion="distance")

    clusters: dict[int, list[str]] = {}
    for name, label in zip(names, labels):
        clusters.setdefault(int(label), []).append(name)

    groups = {}
    for cluster_names in clusters.values():
        canonical = sorted(cluster_names)[0]
        for name in cluster_names:
            groups[name] = canonical

    return groups
