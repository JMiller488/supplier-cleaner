"""
grouping.py
-----------
Groups similar supplier names using sentence embeddings and
cosine similarity. Implements a connected-components approach
via NetworkX to produce stable, order-independent groupings.
"""

import networkx as nx
from sentence_transformers import SentenceTransformer, util


DEFAULT_MODEL = "all-MiniLM-L6-v2"
DEFAULT_THRESHOLD = 0.69


def build_similarity_graph(
    names: list[str], model: SentenceTransformer, threshold: float = DEFAULT_THRESHOLD
) -> nx.Graph:
    """Build a graph where edges connect names above the similarity threshold.

    Each name is a node. An edge is added between two names if their
    cosine similarity meets or exceeds the threshold.

    Args:
        names: List of preprocessed supplier name strings.
        model: A loaded SentenceTransformer model.
        threshold: Minimum cosine similarity to consider two names equivalent.

    Returns:
        A NetworkX Graph object.
    """
    embeddings = model.encode(names, convert_to_tensor=True)
    cos_sim_matrix = util.cos_sim(embeddings, embeddings)

    graph = nx.Graph()
    graph.add_nodes_from(names)

    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            if cos_sim_matrix[i][j] >= threshold:
                graph.add_edge(names[i], names[j])

    return graph


def group_suppliers(
    names: list[str],
    threshold: float = DEFAULT_THRESHOLD,
    model_name: str = DEFAULT_MODEL,
) -> dict[str, str]:
    """Group similar supplier names using connected components.

    Each connected component in the similarity graph is treated as a
    group. All names in a component are mapped to the same canonical
    name — the first name in the component alphabetically, for stability.

    Args:
        names: List of preprocessed supplier name strings.
        threshold: Minimum cosine similarity to consider two names equivalent.
        model_name: HuggingFace model name for SentenceTransformer.

    Returns:
        Dict mapping each input name to its canonical group name.
    """
    model = SentenceTransformer(model_name)
    graph = build_similarity_graph(names, model, threshold)

    groups = {}
    for component in nx.connected_components(graph):
        canonical = sorted(component)[0]
        for name in component:
            groups[name] = canonical

    return groups
