"""Tests for VectorRecord."""

import pytest

from pyvecdb.errors import EmptyVectorError
from pyvecdb.record import VectorRecord

# Named constants
DIMS_THREE = 3
COMPONENT_A = 0.1
COMPONENT_B = 0.2
COMPONENT_C = 0.3


class TestVectorRecord:
    """Tests for VectorRecord creation and methods."""

    def test_create_basic(self) -> None:
        """Create a record with ID, vector, and metadata."""
        record = VectorRecord(
            record_id="test-1",
            vector=[COMPONENT_A, COMPONENT_B, COMPONENT_C],
            metadata={"name": "test"},
        )
        assert record.record_id == "test-1"
        assert record.vector == [COMPONENT_A, COMPONENT_B, COMPONENT_C]
        assert record.metadata == {"name": "test"}

    def test_dimensions(self) -> None:
        """Return the number of dimensions."""
        record = VectorRecord(record_id="x", vector=[COMPONENT_A, COMPONENT_B, COMPONENT_C])
        assert record.dimensions == DIMS_THREE

    def test_empty_metadata_default(self) -> None:
        """Default metadata is an empty dict."""
        record = VectorRecord(record_id="x", vector=[COMPONENT_A])
        assert record.metadata == {}

    def test_empty_vector_raises(self) -> None:
        """Raise EmptyVectorError for empty vector."""
        with pytest.raises(EmptyVectorError):
            VectorRecord(record_id="x", vector=[])

    def test_to_dict(self) -> None:
        """Convert a record to a dictionary."""
        record = VectorRecord(
            record_id="abc",
            vector=[COMPONENT_A, COMPONENT_B],
            metadata={"key": "val"},
        )
        data = record.to_dict()
        assert data["record_id"] == "abc"
        assert data["vector"] == [COMPONENT_A, COMPONENT_B]
        assert data["metadata"] == {"key": "val"}

    def test_from_dict(self) -> None:
        """Create a record from a dictionary."""
        data: dict[str, object] = {
            "record_id": "xyz",
            "vector": [COMPONENT_A, COMPONENT_B, COMPONENT_C],
            "metadata": {"genre": "rock"},
        }
        record = VectorRecord.from_dict(data)
        assert record.record_id == "xyz"
        assert record.dimensions == DIMS_THREE
        assert record.metadata["genre"] == "rock"

    def test_from_dict_no_metadata(self) -> None:
        """Create a record from a dict without metadata."""
        data: dict[str, object] = {
            "record_id": "no-meta",
            "vector": [COMPONENT_A],
        }
        record = VectorRecord.from_dict(data)
        assert record.metadata == {}

    def test_from_dict_invalid_vector(self) -> None:
        """Raise TypeError for non-list vector."""
        data: dict[str, object] = {"record_id": "bad", "vector": "not-a-list"}
        with pytest.raises(TypeError, match="Vector must be a list"):
            VectorRecord.from_dict(data)

    def test_from_dict_invalid_metadata_defaults_empty(self) -> None:
        """Default to empty metadata when metadata field is not a dict."""
        data: dict[str, object] = {
            "record_id": "bad",
            "vector": [COMPONENT_A],
            "metadata": "not-a-dict",
        }
        record = VectorRecord.from_dict(data)
        assert record.metadata == {}

    def test_roundtrip(self) -> None:
        """Convert to dict and back produces identical record."""
        original = VectorRecord(
            record_id="rt",
            vector=[COMPONENT_A, COMPONENT_B],
            metadata={"n": 42, "flag": True},
        )
        raw: dict[str, object] = dict(original.to_dict())
        restored = VectorRecord.from_dict(raw)
        assert restored.record_id == original.record_id
        assert restored.vector == original.vector
        assert restored.metadata == original.metadata
