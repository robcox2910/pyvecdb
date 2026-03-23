"""Tests for vector operations."""

import math

import pytest

from pyvecdb.errors import DimensionMismatchError, EmptyVectorError
from pyvecdb.vector import add, dot, magnitude, normalize, scale, subtract

# Named constants for test values (PLR2004)
COMPONENT_ONE = 1.0
COMPONENT_TWO = 2.0
COMPONENT_THREE = 3.0
COMPONENT_FOUR = 4.0
COMPONENT_FIVE = 5.0
COMPONENT_SIX = 6.0
SCALE_FACTOR = 2.0
MAGNITUDE_3_4 = 5.0
DOT_PRODUCT_RESULT = 32.0
TOLERANCE = 1e-9


class TestAdd:
    """Tests for vector addition."""

    def test_add_basic(self) -> None:
        """Add two vectors component-wise."""
        result = add([COMPONENT_ONE, COMPONENT_TWO], [COMPONENT_THREE, COMPONENT_FOUR])
        assert result == [COMPONENT_FOUR, COMPONENT_SIX]

    def test_add_dimension_mismatch(self) -> None:
        """Raise DimensionMismatchError for mismatched dimensions."""
        with pytest.raises(DimensionMismatchError):
            add([COMPONENT_ONE], [COMPONENT_ONE, COMPONENT_TWO])

    def test_add_empty_vector(self) -> None:
        """Raise EmptyVectorError for empty input."""
        with pytest.raises(EmptyVectorError):
            add([], [COMPONENT_ONE])

    def test_add_empty_second(self) -> None:
        """Raise EmptyVectorError when second vector is empty."""
        with pytest.raises(EmptyVectorError):
            add([COMPONENT_ONE], [])


class TestSubtract:
    """Tests for vector subtraction."""

    def test_subtract_basic(self) -> None:
        """Subtract two vectors component-wise."""
        result = subtract([COMPONENT_FIVE, COMPONENT_THREE], [COMPONENT_ONE, COMPONENT_TWO])
        assert result == [COMPONENT_FOUR, COMPONENT_ONE]

    def test_subtract_dimension_mismatch(self) -> None:
        """Raise DimensionMismatchError for mismatched dimensions."""
        with pytest.raises(DimensionMismatchError):
            subtract([COMPONENT_ONE, COMPONENT_TWO], [COMPONENT_ONE])

    def test_subtract_empty(self) -> None:
        """Raise EmptyVectorError for empty input."""
        with pytest.raises(EmptyVectorError):
            subtract([], [])


class TestScale:
    """Tests for vector scaling."""

    def test_scale_basic(self) -> None:
        """Scale a vector by a factor."""
        result = scale([COMPONENT_ONE, COMPONENT_TWO, COMPONENT_THREE], SCALE_FACTOR)
        assert result == [SCALE_FACTOR, COMPONENT_FOUR, COMPONENT_SIX]

    def test_scale_zero(self) -> None:
        """Scale by zero produces zero vector."""
        result = scale([COMPONENT_ONE, COMPONENT_TWO], 0.0)
        assert result == [0.0, 0.0]

    def test_scale_empty(self) -> None:
        """Raise EmptyVectorError for empty input."""
        with pytest.raises(EmptyVectorError):
            scale([], SCALE_FACTOR)


class TestDot:
    """Tests for dot product."""

    def test_dot_basic(self) -> None:
        """Compute dot product of two vectors."""
        result = dot(
            [COMPONENT_ONE, COMPONENT_TWO, COMPONENT_THREE],
            [COMPONENT_FOUR, COMPONENT_FIVE, COMPONENT_SIX],
        )
        assert result == pytest.approx(DOT_PRODUCT_RESULT)

    def test_dot_perpendicular(self) -> None:
        """Perpendicular vectors have zero dot product."""
        result = dot([COMPONENT_ONE, 0.0], [0.0, COMPONENT_ONE])
        assert result == pytest.approx(0.0)

    def test_dot_dimension_mismatch(self) -> None:
        """Raise DimensionMismatchError for mismatched dimensions."""
        with pytest.raises(DimensionMismatchError):
            dot([COMPONENT_ONE], [COMPONENT_ONE, COMPONENT_TWO])

    def test_dot_empty(self) -> None:
        """Raise EmptyVectorError for empty input."""
        with pytest.raises(EmptyVectorError):
            dot([], [COMPONENT_ONE])


class TestMagnitude:
    """Tests for vector magnitude."""

    def test_magnitude_3_4(self) -> None:
        """Compute magnitude of a 3-4-5 triangle vector."""
        result = magnitude([COMPONENT_THREE, COMPONENT_FOUR])
        assert result == pytest.approx(MAGNITUDE_3_4)

    def test_magnitude_unit(self) -> None:
        """Unit vector has magnitude 1."""
        result = magnitude([COMPONENT_ONE, 0.0])
        assert result == pytest.approx(COMPONENT_ONE)

    def test_magnitude_empty(self) -> None:
        """Raise EmptyVectorError for empty input."""
        with pytest.raises(EmptyVectorError):
            magnitude([])


class TestNormalize:
    """Tests for vector normalization."""

    def test_normalize_basic(self) -> None:
        """Normalize a vector to unit length."""
        result = normalize([COMPONENT_THREE, COMPONENT_FOUR])
        assert len(result) == 2  # noqa: PLR2004
        assert magnitude(result) == pytest.approx(COMPONENT_ONE, abs=TOLERANCE)

    def test_normalize_preserves_direction(self) -> None:
        """Normalized vector points in the same direction."""
        original = [COMPONENT_THREE, COMPONENT_FOUR]
        result = normalize(original)
        ratio = original[0] / result[0]
        assert original[1] / result[1] == pytest.approx(ratio, abs=TOLERANCE)

    def test_normalize_zero_vector(self) -> None:
        """Raise ValueError for zero vector."""
        with pytest.raises(ValueError, match="Cannot normalize a zero vector"):
            normalize([0.0, 0.0])

    def test_normalize_empty(self) -> None:
        """Raise EmptyVectorError for empty input."""
        with pytest.raises(EmptyVectorError):
            normalize([])

    def test_normalize_already_unit(self) -> None:
        """Normalizing a unit vector returns a unit vector."""
        unit = [COMPONENT_ONE / math.sqrt(SCALE_FACTOR), COMPONENT_ONE / math.sqrt(SCALE_FACTOR)]
        result = normalize(unit)
        assert magnitude(result) == pytest.approx(COMPONENT_ONE, abs=TOLERANCE)
