"""Tests for distance and similarity metrics."""

import math

import pytest

from pyvecdb.distance import (
    cosine_similarity,
    dot_product,
    euclidean_distance,
    manhattan_distance,
)
from pyvecdb.errors import DimensionMismatchError, EmptyVectorError

# Named constants
UNIT_X = [1.0, 0.0]
UNIT_Y = [0.0, 1.0]
VEC_A = [1.0, 2.0, 3.0]
VEC_B = [4.0, 5.0, 6.0]
IDENTICAL_SIMILARITY = 1.0
PERPENDICULAR_SIMILARITY = 0.0
OPPOSITE_SIMILARITY = -1.0
DOT_RESULT_32 = 32.0
MANHATTAN_RESULT_9 = 9.0
TOLERANCE = 1e-9


class TestCosineSimilarity:
    """Tests for cosine similarity."""

    def test_identical_vectors(self) -> None:
        """Identical vectors have cosine similarity of 1."""
        result = cosine_similarity(UNIT_X, UNIT_X)
        assert result == pytest.approx(IDENTICAL_SIMILARITY, abs=TOLERANCE)

    def test_perpendicular_vectors(self) -> None:
        """Perpendicular vectors have cosine similarity of 0."""
        result = cosine_similarity(UNIT_X, UNIT_Y)
        assert result == pytest.approx(PERPENDICULAR_SIMILARITY, abs=TOLERANCE)

    def test_opposite_vectors(self) -> None:
        """Opposite vectors have cosine similarity of -1."""
        result = cosine_similarity([1.0, 0.0], [-1.0, 0.0])
        assert result == pytest.approx(OPPOSITE_SIMILARITY, abs=TOLERANCE)

    def test_zero_vector_raises(self) -> None:
        """Raise ValueError for zero vector."""
        with pytest.raises(ValueError, match="zero vector"):
            cosine_similarity([0.0, 0.0], UNIT_X)

    def test_dimension_mismatch(self) -> None:
        """Raise DimensionMismatchError for mismatched dimensions."""
        with pytest.raises(DimensionMismatchError):
            cosine_similarity([1.0], [1.0, 2.0])

    def test_empty_vector(self) -> None:
        """Raise EmptyVectorError for empty input."""
        with pytest.raises(EmptyVectorError):
            cosine_similarity([], [1.0])


class TestEuclideanDistance:
    """Tests for Euclidean distance."""

    def test_identical_vectors(self) -> None:
        """Identical vectors have distance 0."""
        result = euclidean_distance(UNIT_X, UNIT_X)
        assert result == pytest.approx(0.0, abs=TOLERANCE)

    def test_known_distance(self) -> None:
        """Compute known 3-4-5 triangle distance."""
        result = euclidean_distance([0.0, 0.0], [3.0, 4.0])
        expected = 5.0
        assert result == pytest.approx(expected, abs=TOLERANCE)

    def test_3d_distance(self) -> None:
        """Compute distance between 3D vectors."""
        result = euclidean_distance(VEC_A, VEC_B)
        expected = math.sqrt(27.0)
        assert result == pytest.approx(expected, abs=TOLERANCE)

    def test_dimension_mismatch(self) -> None:
        """Raise DimensionMismatchError for mismatched dimensions."""
        with pytest.raises(DimensionMismatchError):
            euclidean_distance([1.0], [1.0, 2.0])

    def test_empty_vector(self) -> None:
        """Raise EmptyVectorError for empty input."""
        with pytest.raises(EmptyVectorError):
            euclidean_distance([], [])


class TestDotProduct:
    """Tests for dot product."""

    def test_basic(self) -> None:
        """Compute dot product of two vectors."""
        result = dot_product(VEC_A, VEC_B)
        assert result == pytest.approx(DOT_RESULT_32, abs=TOLERANCE)

    def test_perpendicular(self) -> None:
        """Perpendicular vectors have dot product 0."""
        result = dot_product(UNIT_X, UNIT_Y)
        assert result == pytest.approx(PERPENDICULAR_SIMILARITY, abs=TOLERANCE)

    def test_dimension_mismatch(self) -> None:
        """Raise DimensionMismatchError for mismatched dimensions."""
        with pytest.raises(DimensionMismatchError):
            dot_product([1.0], [1.0, 2.0])

    def test_empty_vector(self) -> None:
        """Raise EmptyVectorError for empty input."""
        with pytest.raises(EmptyVectorError):
            dot_product([], [1.0])


class TestManhattanDistance:
    """Tests for Manhattan distance."""

    def test_identical_vectors(self) -> None:
        """Identical vectors have distance 0."""
        result = manhattan_distance(UNIT_X, UNIT_X)
        assert result == pytest.approx(0.0, abs=TOLERANCE)

    def test_known_distance(self) -> None:
        """Compute known Manhattan distance."""
        result = manhattan_distance(VEC_A, VEC_B)
        assert result == pytest.approx(MANHATTAN_RESULT_9, abs=TOLERANCE)

    def test_dimension_mismatch(self) -> None:
        """Raise DimensionMismatchError for mismatched dimensions."""
        with pytest.raises(DimensionMismatchError):
            manhattan_distance([1.0], [1.0, 2.0])

    def test_empty_vector(self) -> None:
        """Raise EmptyVectorError for empty input."""
        with pytest.raises(EmptyVectorError):
            manhattan_distance([], [])
