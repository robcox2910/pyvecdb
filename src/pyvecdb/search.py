"""K-Nearest Neighbors (KNN) search for finding similar vectors.

Imagine you're standing in a huge room full of people, and you want to find
the 5 people standing closest to you. That's KNN search! Instead of people,
we have vectors (lists of numbers), and "close" means "similar."

This module provides the core search function that:

1. Computes the distance from your query vector to every vector in the store
2. Sorts by distance (closest first)
3. Returns the top K results

Example:
    >>> from pyvecdb.store import VectorStore
    >>> from pyvecdb.search import knn_search
    >>> store = VectorStore(dimensions=2)
    >>> store.insert("a", [1.0, 0.0])
    >>> store.insert("b", [0.0, 1.0])
    >>> store.insert("c", [0.9, 0.1])
    >>> results = knn_search(store, query=[1.0, 0.0], k=2)
    >>> [r.record_id for r, _ in results]
    ['a', 'c']

"""

from pyvecdb.distance import cosine_similarity, euclidean_distance, manhattan_distance
from pyvecdb.errors import DimensionMismatchError, EmptyStoreError
from pyvecdb.filtering import MetadataFilter, filter_records
from pyvecdb.record import VectorRecord
from pyvecdb.store import VectorStore

# Supported metric names
METRIC_COSINE = "cosine"
METRIC_EUCLIDEAN = "euclidean"
METRIC_DOT = "dot"
METRIC_MANHATTAN = "manhattan"

VALID_METRICS = {METRIC_COSINE, METRIC_EUCLIDEAN, METRIC_DOT, METRIC_MANHATTAN}


def _compute_score(query: list[float], candidate: list[float], metric: str) -> float:
    """Compute a similarity/distance score between two vectors.

    For cosine and dot product, higher is better (more similar).
    For euclidean and manhattan, lower is better (closer together).

    We negate distance metrics so that sorting by descending score
    always puts the best matches first.

    Args:
        query: The query vector.
        candidate: The candidate vector to compare against.
        metric: One of 'cosine', 'euclidean', 'dot', or 'manhattan'.

    Returns:
        A score where higher means more similar.

    """
    if metric == METRIC_COSINE:
        return cosine_similarity(query, candidate)
    if metric == METRIC_EUCLIDEAN:
        return -euclidean_distance(query, candidate)
    if metric == METRIC_MANHATTAN:
        return -manhattan_distance(query, candidate)
    # dot product
    return sum(q * c for q, c in zip(query, candidate, strict=False))


def knn_search(
    store: VectorStore,
    query: list[float],
    k: int = 5,
    metric: str = METRIC_COSINE,
    metadata_filter: MetadataFilter | None = None,
) -> list[tuple[VectorRecord, float]]:
    """Find the K most similar vectors to a query vector.

    This is the heart of a vector database! It checks every vector in the
    store, computes how similar it is to your query, and returns the top K.

    Args:
        store: The vector store to search.
        query: The query vector to find neighbors for.
        k: How many results to return (default 5).
        metric: Distance metric -- 'cosine', 'euclidean', 'dot', or 'manhattan'.
        metadata_filter: Optional filter to narrow results by metadata.

    Returns:
        A list of (record, score) tuples, sorted by similarity (best first).

    Raises:
        EmptyStoreError: If the store has no records.
        DimensionMismatchError: If the query has wrong dimensions.
        ValueError: If the metric is not recognized or k < 1.

    """
    if store.count == 0:
        msg = "Cannot search an empty store"
        raise EmptyStoreError(msg)

    if len(query) != store.dimensions:
        msg = f"Query has {len(query)} dimensions, store expects {store.dimensions}"
        raise DimensionMismatchError(msg)

    if metric not in VALID_METRICS:
        msg = f"Unknown metric '{metric}'. Choose from: {', '.join(sorted(VALID_METRICS))}"
        raise ValueError(msg)

    min_k = 1
    if k < min_k:
        msg = f"k must be at least {min_k}, got {k}"
        raise ValueError(msg)

    # Get all records, optionally filtered
    candidates = store.list_records()
    if metadata_filter is not None:
        candidates = filter_records(candidates, metadata_filter)

    if not candidates:
        return []

    # Score each candidate
    scored: list[tuple[VectorRecord, float]] = []
    for record in candidates:
        score = _compute_score(query, record.vector, metric)
        scored.append((record, score))

    # Sort by score descending (higher = better match)
    scored.sort(key=lambda pair: pair[1], reverse=True)

    return scored[:k]


def batch_search(
    store: VectorStore,
    queries: list[list[float]],
    k: int = 5,
    metric: str = METRIC_COSINE,
    metadata_filter: MetadataFilter | None = None,
) -> list[list[tuple[VectorRecord, float]]]:
    """Search for multiple query vectors at once.

    Like asking "find the 5 closest pizza shops" for each of your friends
    at the same time!

    Args:
        store: The vector store to search.
        queries: A list of query vectors.
        k: How many results per query (default 5).
        metric: Distance metric to use.
        metadata_filter: Optional metadata filter applied to all queries.

    Returns:
        A list of result lists, one per query.

    """
    return [
        knn_search(store, query=q, k=k, metric=metric, metadata_filter=metadata_filter)
        for q in queries
    ]
