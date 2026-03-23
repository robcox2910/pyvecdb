"""Tests for metadata filtering."""

from pyvecdb.filtering import FilterCondition, filter_records, matches_filter
from pyvecdb.record import VectorRecord

# Named constants
YEAR_2020 = 2020
YEAR_2021 = 2021
YEAR_2022 = 2022
YEAR_2023 = 2023
RATING_HIGH = 9.5
VEC_PLACEHOLDER = [1.0]


class TestMatchesFilter:
    """Tests for the matches_filter function."""

    def test_exact_match(self) -> None:
        """Match when value equals the filter."""
        metadata = {"genre": "rock", "year": YEAR_2023}
        assert matches_filter(metadata, {"genre": "rock"})

    def test_exact_no_match(self) -> None:
        """Reject when value differs."""
        metadata = {"genre": "pop"}
        assert not matches_filter(metadata, {"genre": "rock"})

    def test_missing_field(self) -> None:
        """Reject when field is missing from metadata."""
        metadata = {"genre": "rock"}
        assert not matches_filter(metadata, {"year": YEAR_2023})

    def test_eq_operator(self) -> None:
        """Match with explicit $eq operator."""
        metadata = {"genre": "rock"}
        assert matches_filter(metadata, {"genre": {"$eq": "rock"}})

    def test_ne_operator(self) -> None:
        """Match with $ne operator."""
        metadata = {"genre": "rock"}
        assert matches_filter(metadata, {"genre": {"$ne": "pop"}})

    def test_ne_no_match(self) -> None:
        """Reject with $ne when values are equal."""
        metadata = {"genre": "rock"}
        assert not matches_filter(metadata, {"genre": {"$ne": "rock"}})

    def test_gt_operator(self) -> None:
        """Match with $gt operator."""
        metadata = {"year": YEAR_2023}
        assert matches_filter(metadata, {"year": {"$gt": YEAR_2022}})

    def test_gt_no_match(self) -> None:
        """Reject with $gt when value is not greater."""
        metadata = {"year": YEAR_2020}
        assert not matches_filter(metadata, {"year": {"$gt": YEAR_2023}})

    def test_gte_operator(self) -> None:
        """Match with $gte operator (equal case)."""
        metadata = {"year": YEAR_2023}
        assert matches_filter(metadata, {"year": {"$gte": YEAR_2023}})

    def test_lt_operator(self) -> None:
        """Match with $lt operator."""
        metadata = {"year": YEAR_2020}
        assert matches_filter(metadata, {"year": {"$lt": YEAR_2021}})

    def test_lte_operator(self) -> None:
        """Match with $lte operator (equal case)."""
        metadata = {"year": YEAR_2020}
        assert matches_filter(metadata, {"year": {"$lte": YEAR_2020}})

    def test_in_operator(self) -> None:
        """Match with $in operator."""
        metadata = {"genre": "rock"}
        assert matches_filter(metadata, {"genre": {"$in": ["rock", "pop"]}})

    def test_in_no_match(self) -> None:
        """Reject with $in when value not in list."""
        metadata = {"genre": "jazz"}
        assert not matches_filter(metadata, {"genre": {"$in": ["rock", "pop"]}})

    def test_multiple_conditions_all_match(self) -> None:
        """Match when all conditions pass (AND logic)."""
        metadata = {"genre": "rock", "year": YEAR_2023, "rating": RATING_HIGH}
        filt: dict[str, FilterCondition] = {"genre": "rock", "year": {"$gte": YEAR_2020}}
        assert matches_filter(metadata, filt)

    def test_multiple_conditions_one_fails(self) -> None:
        """Reject when any condition fails."""
        metadata = {"genre": "rock", "year": YEAR_2020}
        filt: dict[str, FilterCondition] = {"genre": "rock", "year": {"$gt": YEAR_2023}}
        assert not matches_filter(metadata, filt)

    def test_boolean_metadata(self) -> None:
        """Match with boolean metadata values."""
        metadata = {"featured": True}
        assert matches_filter(metadata, {"featured": True})

    def test_gt_with_non_numeric_returns_false(self) -> None:
        """Return False for $gt with non-numeric values."""
        metadata = {"name": "alice"}
        assert not matches_filter(metadata, {"name": {"$gt": "bob"}})

    def test_in_with_non_list_returns_false(self) -> None:
        """Return False for $in with a non-list expected value."""
        metadata = {"genre": "rock"}
        assert not matches_filter(metadata, {"genre": {"$in": "rock"}})  # type: ignore[dict-item]


class TestFilterRecords:
    """Tests for filtering a list of records."""

    def test_filter_records(self) -> None:
        """Filter records by metadata."""
        records = [
            VectorRecord(record_id="a", vector=VEC_PLACEHOLDER, metadata={"genre": "rock"}),
            VectorRecord(record_id="b", vector=VEC_PLACEHOLDER, metadata={"genre": "pop"}),
            VectorRecord(record_id="c", vector=VEC_PLACEHOLDER, metadata={"genre": "rock"}),
        ]
        result = filter_records(records, {"genre": "rock"})
        expected_count = 2
        assert len(result) == expected_count
        assert result[0].record_id == "a"
        assert result[1].record_id == "c"

    def test_filter_records_empty_result(self) -> None:
        """Return empty list when no records match."""
        records = [
            VectorRecord(record_id="a", vector=VEC_PLACEHOLDER, metadata={"genre": "rock"}),
        ]
        result = filter_records(records, {"genre": "jazz"})
        assert result == []

    def test_filter_records_empty_filter(self) -> None:
        """Return all records when filter is empty."""
        records = [
            VectorRecord(record_id="a", vector=VEC_PLACEHOLDER, metadata={"genre": "rock"}),
            VectorRecord(record_id="b", vector=VEC_PLACEHOLDER, metadata={"genre": "pop"}),
        ]
        result = filter_records(records, {})
        expected_count = 2
        assert len(result) == expected_count
