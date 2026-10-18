"""Tests for VectorStore."""

import pytest

from pyvecdb.errors import (
    DimensionMismatchError,
    DuplicateIDError,
    RecordNotFoundError,
)
from pyvecdb.store import VectorStore

# Named constants
DIMS_TWO = 2
DIMS_THREE = 3
VEC_A = [1.0, 2.0]
VEC_B = [3.0, 4.0]
VEC_C = [5.0, 6.0]
VEC_3D = [1.0, 2.0, 3.0]
EXPECTED_COUNT_ZERO = 0
EXPECTED_COUNT_ONE = 1
EXPECTED_COUNT_TWO = 2
EXPECTED_COUNT_THREE = 3


class TestVectorStoreCreation:
    """Tests for creating a VectorStore."""

    def test_create_store(self) -> None:
        """Create an empty store with specified dimensions."""
        store = VectorStore(dimensions=DIMS_THREE)
        assert store.dimensions == DIMS_THREE
        assert store.count == EXPECTED_COUNT_ZERO

    def test_invalid_dimensions(self) -> None:
        """Raise ValueError for zero or negative dimensions."""
        with pytest.raises(ValueError, match="Dimensions must be at least"):
            VectorStore(dimensions=0)


class TestInsert:
    """Tests for inserting records."""

    def test_insert_basic(self) -> None:
        """Insert a record and verify it exists."""
        store = VectorStore(dimensions=DIMS_TWO)
        record = store.insert("a", VEC_A, metadata={"tag": "first"})
        assert record.record_id == "a"
        assert store.count == EXPECTED_COUNT_ONE

    def test_insert_duplicate_id(self) -> None:
        """Raise DuplicateIDError for duplicate IDs."""
        store = VectorStore(dimensions=DIMS_TWO)
        store.insert("a", VEC_A)
        with pytest.raises(DuplicateIDError):
            store.insert("a", VEC_B)

    def test_insert_wrong_dimensions(self) -> None:
        """Raise DimensionMismatchError for wrong dimensions."""
        store = VectorStore(dimensions=DIMS_TWO)
        with pytest.raises(DimensionMismatchError):
            store.insert("a", VEC_3D)

    def test_insert_no_metadata(self) -> None:
        """Insert a record without metadata."""
        store = VectorStore(dimensions=DIMS_TWO)
        record = store.insert("a", VEC_A)
        assert record.metadata == {}


class TestGet:
    """Tests for getting records by ID."""

    def test_get_existing(self) -> None:
        """Get an existing record by ID."""
        store = VectorStore(dimensions=DIMS_TWO)
        store.insert("a", VEC_A, metadata={"x": "y"})
        record = store.get("a")
        assert record.record_id == "a"
        assert record.vector == VEC_A

    def test_get_missing(self) -> None:
        """Raise RecordNotFoundError for missing ID."""
        store = VectorStore(dimensions=DIMS_TWO)
        with pytest.raises(RecordNotFoundError):
            store.get("nonexistent")


class TestUpdate:
    """Tests for updating records."""

    def test_update_vector(self) -> None:
        """Update only the vector of a record."""
        store = VectorStore(dimensions=DIMS_TWO)
        store.insert("a", VEC_A, metadata={"keep": "this"})
        updated = store.update("a", vector=VEC_B)
        assert updated.vector == VEC_B
        assert updated.metadata == {"keep": "this"}

    def test_update_metadata(self) -> None:
        """Update only the metadata of a record."""
        store = VectorStore(dimensions=DIMS_TWO)
        store.insert("a", VEC_A, metadata={"old": "data"})
        updated = store.update("a", metadata={"new": "data"})
        assert updated.vector == VEC_A
        assert updated.metadata == {"new": "data"}

    def test_update_both(self) -> None:
        """Update both vector and metadata."""
        store = VectorStore(dimensions=DIMS_TWO)
        store.insert("a", VEC_A, metadata={"old": "data"})
        updated = store.update("a", vector=VEC_B, metadata={"new": "data"})
        assert updated.vector == VEC_B
        assert updated.metadata == {"new": "data"}

    def test_update_wrong_dimensions(self) -> None:
        """Raise DimensionMismatchError for wrong dimensions on update."""
        store = VectorStore(dimensions=DIMS_TWO)
        store.insert("a", VEC_A)
        with pytest.raises(DimensionMismatchError):
            store.update("a", vector=VEC_3D)

    def test_update_missing(self) -> None:
        """Raise RecordNotFoundError for missing ID."""
        store = VectorStore(dimensions=DIMS_TWO)
        with pytest.raises(RecordNotFoundError):
            store.update("missing", vector=VEC_A)


class TestDelete:
    """Tests for deleting records."""

    def test_delete_existing(self) -> None:
        """Delete an existing record."""
        store = VectorStore(dimensions=DIMS_TWO)
        store.insert("a", VEC_A)
        store.delete("a")
        assert store.count == EXPECTED_COUNT_ZERO
        assert not store.contains("a")

    def test_delete_missing(self) -> None:
        """Raise RecordNotFoundError for missing ID."""
        store = VectorStore(dimensions=DIMS_TWO)
        with pytest.raises(RecordNotFoundError):
            store.delete("missing")


class TestListAndContains:
    """Tests for list_records and contains."""

    def test_list_records(self) -> None:
        """List all records in insertion order."""
        store = VectorStore(dimensions=DIMS_TWO)
        store.insert("a", VEC_A)
        store.insert("b", VEC_B)
        records = store.list_records()
        assert len(records) == EXPECTED_COUNT_TWO
        assert records[0].record_id == "a"
        assert records[1].record_id == "b"

    def test_contains_true(self) -> None:
        """Return True for existing ID."""
        store = VectorStore(dimensions=DIMS_TWO)
        store.insert("a", VEC_A)
        assert store.contains("a")

    def test_contains_false(self) -> None:
        """Return False for missing ID."""
        store = VectorStore(dimensions=DIMS_TWO)
        assert not store.contains("missing")


class TestClear:
    """Tests for clearing the store."""

    def test_clear(self) -> None:
        """Clear removes all records."""
        store = VectorStore(dimensions=DIMS_TWO)
        store.insert("a", VEC_A)
        store.insert("b", VEC_B)
        store.clear()
        assert store.count == EXPECTED_COUNT_ZERO


class TestBatchInsert:
    """Tests for batch_insert."""

    def test_batch_insert(self) -> None:
        """Insert multiple records at once."""
        store = VectorStore(dimensions=DIMS_TWO)
        records = store.batch_insert(
            [
                ("a", VEC_A, None),
                ("b", VEC_B, {"tag": "second"}),
                ("c", VEC_C, None),
            ]
        )
        assert len(records) == EXPECTED_COUNT_THREE
        assert store.count == EXPECTED_COUNT_THREE

    def test_batch_insert_duplicate_in_batch(self) -> None:
        """Raise DuplicateIDError for duplicate IDs in batch."""
        store = VectorStore(dimensions=DIMS_TWO)
        with pytest.raises(DuplicateIDError):
            store.batch_insert(
                [
                    ("a", VEC_A, None),
                    ("a", VEC_B, None),
                ]
            )

    def test_batch_insert_atomic_on_duplicate(self) -> None:
        """A duplicate ID mid-batch leaves the store unchanged (no partial insert)."""
        store = VectorStore(dimensions=DIMS_TWO)
        store.insert("existing", VEC_A)
        with pytest.raises(DuplicateIDError):
            store.batch_insert(
                [
                    ("new1", VEC_B, None),
                    ("existing", VEC_C, None),
                    ("new2", VEC_C, None),
                ]
            )
        assert store.count == EXPECTED_COUNT_ONE
        assert not store.contains("new1")
        assert not store.contains("new2")

    def test_batch_insert_atomic_on_bad_dimensions(self) -> None:
        """A wrong-dimension vector mid-batch leaves the store unchanged."""
        store = VectorStore(dimensions=DIMS_TWO)
        with pytest.raises(DimensionMismatchError):
            store.batch_insert(
                [
                    ("a", VEC_A, None),
                    ("b", VEC_3D, None),
                ]
            )
        assert store.count == EXPECTED_COUNT_ZERO
        assert not store.contains("a")


class TestCopies:
    """Regression: the store never changes behind the caller's back.

    Records handed out are copies, and vectors and metadata handed in are
    copied, so scribbling on either side leaves the store as it was.
    """

    def _store(self) -> VectorStore:
        store = VectorStore(dimensions=DIMS_TWO)
        store.insert("a", [1.0, 2.0], metadata={"genre": "rock"})
        return store

    def _assert_unchanged(self, store: VectorStore) -> None:
        record = store.get("a")
        assert record.vector == [1.0, 2.0]
        assert record.metadata == {"genre": "rock"}

    def test_scribbling_on_a_record_from_get(self) -> None:
        """Changing the vector or metadata get() returned leaves the store alone."""
        store = self._store()
        record = store.get("a")
        record.vector[0] = 99.0
        record.metadata["genre"] = "pop"
        self._assert_unchanged(store)

    def test_scribbling_on_a_record_from_insert(self) -> None:
        """Changing the record insert() returned leaves the store alone."""
        store = VectorStore(dimensions=DIMS_TWO)
        record = store.insert("a", [1.0, 2.0], metadata={"genre": "rock"})
        record.vector.append(99.0)
        record.metadata.clear()
        self._assert_unchanged(store)

    def test_scribbling_on_a_record_from_update(self) -> None:
        """Changing the record update() returned leaves the store alone."""
        store = self._store()
        record = store.update("a", vector=[1.0, 2.0])
        record.vector[1] = 99.0
        record.metadata["genre"] = "pop"
        self._assert_unchanged(store)

    def test_scribbling_on_records_from_list_records(self) -> None:
        """Changing a record list_records() returned leaves the store alone."""
        store = self._store()
        store.list_records()[0].vector[0] = 99.0
        self._assert_unchanged(store)

    def test_scribbling_on_records_from_batch_insert(self) -> None:
        """Changing a record batch_insert() returned leaves the store alone."""
        store = VectorStore(dimensions=DIMS_TWO)
        [record] = store.batch_insert([("a", [1.0, 2.0], {"genre": "rock"})])
        record.vector[0] = 99.0
        self._assert_unchanged(store)

    def test_scribbling_on_what_you_inserted(self) -> None:
        """Changing the vector or metadata you passed to insert() leaves the store alone."""
        store = VectorStore(dimensions=DIMS_TWO)
        vector = [1.0, 2.0]
        metadata: dict[str, str | int | float | bool] = {"genre": "rock"}
        store.insert("a", vector, metadata=metadata)
        vector[0] = 99.0
        metadata["genre"] = "pop"
        self._assert_unchanged(store)

    def test_scribbling_on_what_you_updated_with(self) -> None:
        """Changing the vector or metadata you passed to update() leaves the store alone."""
        store = VectorStore(dimensions=DIMS_TWO)
        store.insert("a", [0.0, 0.0])
        vector = [1.0, 2.0]
        metadata: dict[str, str | int | float | bool] = {"genre": "rock"}
        store.update("a", vector=vector, metadata=metadata)
        vector[0] = 99.0
        metadata["genre"] = "pop"
        self._assert_unchanged(store)

    def test_scribbling_on_what_you_batch_inserted(self) -> None:
        """Changing a vector or metadata you passed to batch_insert() leaves the store alone."""
        store = VectorStore(dimensions=DIMS_TWO)
        vector = [1.0, 2.0]
        metadata: dict[str, str | int | float | bool] = {"genre": "rock"}
        store.batch_insert([("a", vector, metadata)])
        vector[0] = 99.0
        metadata["genre"] = "pop"
        self._assert_unchanged(store)
