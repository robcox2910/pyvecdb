"""Distance and similarity metrics for comparing vectors.

Imagine you and your friend each rate 5 movies from 1 to 10.
Your ratings are a vector like [8, 3, 9, 2, 7] and theirs might be [7, 4, 8, 3, 6].
Distance metrics tell you how similar your taste in movies is!

This module provides four ways to measure how "close" two vectors are:

- **Cosine similarity** -- Are you pointing the same direction? (most popular for AI)
- **Euclidean distance** -- How far apart are you in a straight line?
- **Dot product** -- How much do your ratings "agree"?
- **Manhattan distance** -- How far apart are you walking on a grid?

Example:
    >>> from pyvecdb.distance import cosine_similarity
    >>> cosine_similarity([1.0, 0.0], [1.0, 0.0])
    1.0
    >>> cosine_similarity([1.0, 0.0], [0.0, 1.0])
    0.0

"""

import math

from pyvecdb.vector import check_not_empty, check_same_dimensions


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors.

    Cosine similarity measures the angle between two vectors, ignoring their length.
    It returns a value between -1 (opposite) and 1 (identical direction).
    Think of it like comparing which direction two arrows point, not how long they are.

    Args:
        a: The first vector.
        b: The second vector.

    Returns:
        A float between -1.0 and 1.0 (1.0 means identical direction).

    Raises:
        DimensionMismatchError: If the vectors have different lengths.
        EmptyVectorError: If either vector is empty.
        ValueError: If either vector is a zero vector.

    """
    check_not_empty(a)
    check_not_empty(b)
    check_same_dimensions(a, b)

    dot_product = sum(x * y for x, y in zip(a, b, strict=False))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))

    if mag_a == 0.0 or mag_b == 0.0:
        msg = "Cannot compute cosine similarity with a zero vector"
        raise ValueError(msg)

    return dot_product / (mag_a * mag_b)


def euclidean_distance(a: list[float], b: list[float]) -> float:
    """Compute the Euclidean (straight-line) distance between two vectors.

    This is the "as the crow flies" distance -- the shortest path between
    two points. Like using a ruler to measure between two dots on paper.

    Args:
        a: The first vector.
        b: The second vector.

    Returns:
        A non-negative float (0.0 means the vectors are identical).

    Raises:
        DimensionMismatchError: If the vectors have different lengths.
        EmptyVectorError: If either vector is empty.

    """
    check_not_empty(a)
    check_not_empty(b)
    check_same_dimensions(a, b)

    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b, strict=False)))


def dot_product(a: list[float], b: list[float]) -> float:
    """Compute the dot product of two vectors.

    Multiply each pair of matching numbers and add them all up.
    A bigger result means the vectors "agree" more.

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


def manhattan_distance(a: list[float], b: list[float]) -> float:
    """Compute the Manhattan (taxicab) distance between two vectors.

    Imagine walking through a city grid -- you can only go along streets,
    not cut diagonally through buildings. Manhattan distance is the total
    number of blocks you walk (up/down + left/right).

    Args:
        a: The first vector.
        b: The second vector.

    Returns:
        A non-negative float (0.0 means the vectors are identical).

    Raises:
        DimensionMismatchError: If the vectors have different lengths.
        EmptyVectorError: If either vector is empty.

    """
    check_not_empty(a)
    check_not_empty(b)
    check_same_dimensions(a, b)

    return sum(abs(x - y) for x, y in zip(a, b, strict=False))
