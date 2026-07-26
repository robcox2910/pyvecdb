"""Metadata filtering for vector search results.

Imagine you're searching for the 5 most similar songs to your favorite,
but you only want songs released after 2020. That's metadata filtering!

The metadata is the "extra info" attached to each vector record -- things
like genre, year, artist, rating. Filters let you narrow down results
based on these fields.

Supported filter operators:

- **$eq** (equals): ``{"genre": "rock"}`` -- exact match (plain values are shorthand for $eq)
- **ne** (not equals): ``{"genre": {"$ne": "country"}}``
- **gt / gte / lt / lte**: ``{"year": {"$gt": 2020}}`` -- comparisons
- **in**: ``{"genre": {"$in": ["rock", "pop"]}}`` -- match any in list

Example:
    >>> from pyvecdb.filtering import matches_filter
    >>> metadata = {"genre": "rock", "year": 2023}
    >>> matches_filter(metadata, {"genre": "rock"})
    True
    >>> matches_filter(metadata, {"year": {"$gt": 2020}})
    True

"""

from collections.abc import Mapping

from pyvecdb.record import VectorRecord

# Type alias for metadata values
MetadataValue = str | int | float | bool

# Type alias for a single filter condition
FilterCondition = MetadataValue | dict[str, MetadataValue | list[MetadataValue]]

# Type alias for a complete filter
MetadataFilter = Mapping[str, FilterCondition]


def _compare_values(
    actual: MetadataValue,
    operator: str,
    expected: MetadataValue | list[MetadataValue],
) -> bool:
    """Compare a metadata value using the given operator.

    Args:
        actual: The value from the record's metadata.
        operator: One of '$eq', '$ne', '$gt', '$gte', '$lt', '$lte', '$in'.
        expected: The value (or list) to compare against.

    Returns:
        True if the comparison passes, False otherwise.

    """
    match operator:
        case "$eq":
            return actual == expected
        case "$ne":
            return actual != expected
        case "$in":
            return actual in expected if isinstance(expected, list) else False
        case "$gt" | "$gte" | "$lt" | "$lte":
            if not isinstance(actual, (int, float)) or not isinstance(expected, (int, float)):
                return False
            ops = {
                "$gt": actual > expected,
                "$gte": actual >= expected,
                "$lt": actual < expected,
                "$lte": actual <= expected,
            }
            return ops[operator]
        case _:
            return False


def matches_filter(
    metadata: Mapping[str, MetadataValue],
    filter_dict: MetadataFilter,
) -> bool:
    """Check whether a metadata dict matches all conditions in a filter.

    A filter is a dictionary where each key is a metadata field name. The value
    can be a plain value (shorthand for equals) or a dict of operator-value pairs.

    All conditions must match (AND logic).

    Args:
        metadata: The record's metadata dictionary.
        filter_dict: The filter conditions to check.

    Returns:
        True if all conditions match, False otherwise.

    """
    for field_name, condition in filter_dict.items():
        actual = metadata.get(field_name)
        if actual is None:
            return False

        if isinstance(condition, dict):
            for operator, expected in condition.items():
                if not _compare_values(actual, operator, expected):
                    return False
        elif actual != condition:
            return False

    return True


def filter_records(
    records: list[VectorRecord],
    filter_dict: MetadataFilter,
) -> list[VectorRecord]:
    """Filter a list of vector records by metadata conditions.

    Args:
        records: The records to filter.
        filter_dict: The filter conditions to apply.

    Returns:
        A new list containing only records whose metadata matches all conditions.
        Note: the records themselves are the live objects from the store, so
        editing one here also edits it in the store.

    """
    return [r for r in records if matches_filter(r.metadata, filter_dict)]
