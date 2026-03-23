"""Vector operations for pyvecdb.

A vector is just a list of numbers -- like coordinates on a treasure map!
If you have a 2D map, each spot is described by two numbers (x, y).
In a vector database, we might use hundreds of numbers to describe
something complex like a song, a picture, or a sentence.

This module provides basic math operations on vectors:

- **Add** two vectors together (combine directions)
- **Subtract** one vector from another (find the difference)
- **Scale** a vector by a number (make it bigger or smaller)
- **Magnitude** -- how long the vector is (distance from the origin)
- **Normalize** -- shrink or stretch to length 1 (keep direction, standard size)
- **Dot product** -- a special multiplication that tells how similar two vectors are

Example:
    >>> from pyvecdb.vector import add, magnitude, normalize
    >>> v = [3.0, 4.0]
    >>> magnitude(v)
    5.0
    >>> normalize(v)
    [0.6, 0.8]

"""

import math

from pyvecdb.errors import DimensionMismatchError, EmptyVectorError


def check_not_empty(v: list[float]) -> None:
    """Raise EmptyVectorError if the vector has no components."""
    if not v:
        msg = "Vector must have at least one component"
        raise EmptyVectorError(msg)


def check_same_dimensions(a: list[float], b: list[float]) -> None:
    """Raise DimensionMismatchError if vectors have different lengths."""
    if len(a) != len(b):
        msg = f"Vector dimensions don't match: {len(a)} vs {len(b)}"
        raise DimensionMismatchError(msg)


def add(a: list[float], b: list[float]) -> list[float]:
    """Add two vectors component by component.

    Like walking 3 steps east and then 2 more steps east -- you end up 5 east!

    Args:
        a: The first vector.
        b: The second vector.

    Returns:
        A new vector where each component is the sum of the inputs.

    Raises:
        DimensionMismatchError: If the vectors have different lengths.
        EmptyVectorError: If either vector is empty.

    """
    check_not_empty(a)
    check_not_empty(b)
    check_same_dimensions(a, b)
    return [x + y for x, y in zip(a, b, strict=False)]


def subtract(a: list[float], b: list[float]) -> list[float]:
    """Subtract vector b from vector a component by component.

    Find the difference between two positions -- like figuring out
    which direction to walk to get from one spot to another.

    Args:
        a: The vector to subtract from.
        b: The vector to subtract.

    Returns:
        A new vector where each component is a[i] - b[i].

    Raises:
        DimensionMismatchError: If the vectors have different lengths.
        EmptyVectorError: If either vector is empty.

    """
    check_not_empty(a)
    check_not_empty(b)
    check_same_dimensions(a, b)
    return [x - y for x, y in zip(a, b, strict=False)]


def scale(v: list[float], factor: float) -> list[float]:
    """Multiply every component of a vector by a number.

    Like zooming in on a map -- everything gets bigger by the same amount.

    Args:
        v: The vector to scale.
        factor: The number to multiply by.

    Returns:
        A new vector with each component multiplied by the factor.

    Raises:
        EmptyVectorError: If the vector is empty.

    """
    check_not_empty(v)
    return [x * factor for x in v]


def dot(a: list[float], b: list[float]) -> float:
    """Compute the dot product of two vectors.

    Multiply matching components and add them up. The dot product tells you
    how much two vectors "agree" -- if they point the same way, it's big
    and positive; if opposite, it's big and negative; if perpendicular, it's zero.

    Args:
        a: The first vector.
        b: The second vector.

    Returns:
        The sum of component-wise products.

    Raises:
        DimensionMismatchError: If the vectors have different lengths.
        EmptyVectorError: If either vector is empty.

    """
    check_not_empty(a)
    check_not_empty(b)
    check_same_dimensions(a, b)
    return sum(x * y for x, y in zip(a, b, strict=False))


def magnitude(v: list[float]) -> float:
    """Compute the length (magnitude) of a vector.

    Think of the Pythagorean theorem in any number of dimensions.
    For [3, 4] the magnitude is 5, just like a 3-4-5 right triangle!

    Args:
        v: The vector to measure.

    Returns:
        The Euclidean length of the vector.

    Raises:
        EmptyVectorError: If the vector is empty.

    """
    check_not_empty(v)
    return math.sqrt(sum(x * x for x in v))


def normalize(v: list[float]) -> list[float]:
    """Scale a vector to have length 1 (unit vector).

    Like keeping the direction on a compass but ignoring how far you walked.
    This is called L2 normalization and is essential for cosine similarity.

    Args:
        v: The vector to normalize.

    Returns:
        A new unit vector pointing in the same direction.

    Raises:
        EmptyVectorError: If the vector is empty.
        ValueError: If the vector has zero magnitude (all zeros).

    """
    check_not_empty(v)
    mag = magnitude(v)
    if mag == 0.0:
        msg = "Cannot normalize a zero vector"
        raise ValueError(msg)
    return [x / mag for x in v]
