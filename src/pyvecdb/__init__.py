"""pyvecdb -- an educational vector database built from scratch in Python.

A vector database stores vectors (lists of numbers) and finds similar ones.
Think of it like a music app that knows your taste -- it turns songs into
lists of numbers and finds other songs with similar numbers.

This is part of the PyLearn educational platform for kids aged 12+.

Quick Start:
    >>> from pyvecdb import VectorStore, knn_search
    >>> store = VectorStore(dimensions=3)
    >>> store.insert("rock-song", [0.9, 0.1, 0.3], metadata={"genre": "rock"})
    >>> store.insert("pop-song", [0.2, 0.8, 0.4], metadata={"genre": "pop"})
    >>> store.insert("rock-ballad", [0.7, 0.3, 0.5], metadata={"genre": "rock"})
    >>> results = knn_search(store, query=[0.85, 0.15, 0.35], k=2)
    >>> [r.record_id for r, _ in results]
    ['rock-song', 'rock-ballad']
"""

from pyvecdb.distance import (
    cosine_similarity,
    dot_product,
    euclidean_distance,
    manhattan_distance,
)
from pyvecdb.errors import (
    DimensionMismatchError,
    DuplicateIDError,
    EmptyStoreError,
    EmptyVectorError,
    PersistenceError,
    RecordNotFoundError,
    VecDBError,
)
from pyvecdb.filtering import filter_records, matches_filter
from pyvecdb.persistence import load_store, save_store
from pyvecdb.record import VectorRecord
from pyvecdb.search import batch_search, knn_search
from pyvecdb.store import VectorStore
from pyvecdb.vector import add, dot, magnitude, normalize, scale, subtract

__all__ = [
    "DimensionMismatchError",
    "DuplicateIDError",
    "EmptyStoreError",
    "EmptyVectorError",
    "PersistenceError",
    "RecordNotFoundError",
    "VecDBError",
    "VectorRecord",
    "VectorStore",
    "add",
    "batch_search",
    "cosine_similarity",
    "dot",
    "dot_product",
    "euclidean_distance",
    "filter_records",
    "knn_search",
    "load_store",
    "magnitude",
    "manhattan_distance",
    "matches_filter",
    "normalize",
    "save_store",
    "scale",
    "subtract",
]
