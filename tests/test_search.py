"""Tests for KNN search."""

import pytest

from pyvecdb.errors import DimensionMismatchError, EmptyStoreError
from pyvecdb.search import (
    METRIC_COSINE,
    METRIC_DOT,
    METRIC_EUCLIDEAN,
    METRIC_MANHATTAN,
    _compute_score,
    batch_search,
    knn_search,
)
from pyvecdb.store import VectorStore

# Named constants
DIMS_TWO = 2
DIMS_THREE = 3
K_TWO = 2
K_THREE = 3
K_FIVE = 5
EXPECTED_ONE = 1
EXPECTED_TWO = 2
EXPECTED_THREE = 3


def _make_store() -> VectorStore:
    """Create a store with sample vectors for testing."""
    store = VectorStore(dimensions=DIMS_TWO)
    store.insert("north", [0.0, 1.0])
    store.insert("east", [1.0, 0.0])
    store.insert("northeast", [0.7, 0.7])
    store.insert("south", [0.0, -1.0])
    return store


class TestKNNSearch:
    """Tests for KNN search."""

    def test_cosine_search(self) -> None:
        """Find nearest by cosine similarity."""
        store = _make_store()
        results = knn_search(store, query=[0.0, 1.0], k=K_TWO, metric=METRIC_COSINE)
        assert len(results) == K_TWO
        assert results[0][0].record_id == "north"

    def test_euclidean_search(self) -> None:
        """Find nearest by Euclidean distance."""
        store = _make_store()
        results = knn_search(store, query=[1.0, 0.0], k=K_TWO, metric=METRIC_EUCLIDEAN)
        assert len(results) == K_TWO
        assert results[0][0].record_id == "east"

    def test_dot_product_search(self) -> None:
        """Find nearest by dot product."""
        store = _make_store()
        results = knn_search(store, query=[0.0, 1.0], k=K_TWO, metric=METRIC_DOT)
        assert len(results) == K_TWO
        assert results[0][0].record_id == "north"

    def test_manhattan_search(self) -> None:
        """Find nearest by Manhattan distance."""
        store = _make_store()
        results = knn_search(store, query=[1.0, 0.0], k=K_TWO, metric=METRIC_MANHATTAN)
        assert len(results) == K_TWO
        assert results[0][0].record_id == "east"

    def test_k_larger_than_store(self) -> None:
        """Return all records when k is larger than store size."""
        store = _make_store()
        results = knn_search(store, query=[0.5, 0.5], k=K_FIVE * K_FIVE)
        expected_count = 4
        assert len(results) == expected_count

    def test_empty_store_raises(self) -> None:
        """Raise EmptyStoreError when store is empty."""
        store = VectorStore(dimensions=DIMS_TWO)
        with pytest.raises(EmptyStoreError):
            knn_search(store, query=[1.0, 0.0])

    def test_dimension_mismatch_raises(self) -> None:
        """Raise DimensionMismatchError for wrong query dimensions."""
        store = _make_store()
        with pytest.raises(DimensionMismatchError):
            knn_search(store, query=[1.0, 2.0, 3.0])

    def test_invalid_metric_raises(self) -> None:
        """Raise ValueError for unknown metric name."""
        store = _make_store()
        with pytest.raises(ValueError, match="Unknown metric"):
            knn_search(store, query=[1.0, 0.0], metric="invalid")

    def test_k_zero_raises(self) -> None:
        """Raise ValueError when k is less than 1."""
        store = _make_store()
        with pytest.raises(ValueError, match="k must be at least"):
            knn_search(store, query=[1.0, 0.0], k=0)

    def test_with_metadata_filter(self) -> None:
        """Filter results by metadata during search."""
        store = VectorStore(dimensions=DIMS_TWO)
        store.insert("rock1", [0.9, 0.1], metadata={"genre": "rock"})
        store.insert("pop1", [0.8, 0.2], metadata={"genre": "pop"})
        store.insert("rock2", [0.7, 0.3], metadata={"genre": "rock"})
        results = knn_search(
            store,
            query=[1.0, 0.0],
            k=K_FIVE,
            metadata_filter={"genre": "rock"},
        )
        assert len(results) == EXPECTED_TWO
        assert all(r.metadata["genre"] == "rock" for r, _ in results)

    def test_filter_no_matches(self) -> None:
        """Return empty list when filter matches no records."""
        store = _make_store()
        results = knn_search(
            store,
            query=[1.0, 0.0],
            k=K_FIVE,
            metadata_filter={"genre": "jazz"},
        )
        assert results == []


class TestComputeScore:
    """Tests for the internal _compute_score helper."""

    def test_dot_product_checks_dimensions(self) -> None:
        """Dot product scoring rejects mismatched dimensions instead of truncating."""
        with pytest.raises(DimensionMismatchError):
            _compute_score([1.0, 2.0], [1.0, 2.0, 3.0], METRIC_DOT)

    def test_dot_product_value(self) -> None:
        """Dot product scoring returns the correct sum of products."""
        expected = 11.0  # 1*3 + 2*4
        assert _compute_score([1.0, 2.0], [3.0, 4.0], METRIC_DOT) == expected


class TestBatchSearch:
    """Tests for batch search."""

    def test_batch_search(self) -> None:
        """Search for multiple queries at once."""
        store = _make_store()
        queries = [[0.0, 1.0], [1.0, 0.0]]
        results = batch_search(store, queries=queries, k=K_TWO)
        assert len(results) == EXPECTED_TWO
        assert results[0][0][0].record_id == "north"
        assert results[1][0][0].record_id == "east"

    def test_batch_search_with_filter(self) -> None:
        """Batch search with metadata filtering."""
        store = VectorStore(dimensions=DIMS_TWO)
        store.insert("a", [1.0, 0.0], metadata={"type": "x"})
        store.insert("b", [0.0, 1.0], metadata={"type": "y"})
        results = batch_search(
            store,
            queries=[[1.0, 0.0], [0.0, 1.0]],
            k=K_FIVE,
            metadata_filter={"type": "x"},
        )
        assert len(results) == EXPECTED_TWO
        assert len(results[0]) == EXPECTED_ONE
        assert results[0][0][0].record_id == "a"
