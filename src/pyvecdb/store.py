"""Vector store -- a container for vector records.

Think of a ``VectorStore`` like a filing cabinet where every folder
(record) contains a list of numbers (vector) and a label (ID).
The cabinet has one rule: every folder must contain the same number
of pages (dimensions).

The store provides CRUD operations:

- **Create**: Insert a new vector record
- **Read**: Get a record by its ID
- **Update**: Replace a record's vector or metadata
- **Delete**: Remove a record by its ID
- **List**: Get all records in the store

Example:
    >>> from pyvecdb.store import VectorStore
    >>> store = VectorStore(dimensions=3)
    >>> store.insert("song-1", [0.8, 0.2, 0.5], metadata={"genre": "rock"})
    >>> store.get("song-1").metadata["genre"]
    'rock'
    >>> store.count
    1

"""

from pyvecdb.errors import (
    DimensionMismatchError,
    DuplicateIDError,
    RecordNotFoundError,
)
from pyvecdb.record import VectorRecord


class VectorStore:
    """A container that stores vector records with the same dimensionality.

    All vectors in a store must have the same number of dimensions.
    Think of it like a spreadsheet where every row has the same number of columns.

    Attributes:
        dimensions: The number of dimensions each vector must have.

    """

    def __init__(self, dimensions: int) -> None:
        """Create a new empty vector store.

        Args:
            dimensions: The number of dimensions for vectors in this store.

        Raises:
            ValueError: If dimensions is less than 1.

        """
        min_dims = 1
        if dimensions < min_dims:
            msg = f"Dimensions must be at least {min_dims}, got {dimensions}"
            raise ValueError(msg)
        self._dimensions = dimensions
        self._records: dict[str, VectorRecord] = {}

    @property
    def dimensions(self) -> int:
        """Return the dimensionality of this store."""
        return self._dimensions

    @property
    def count(self) -> int:
        """Return the number of records in this store."""
        return len(self._records)

    def _check_dimensions(self, vector: list[float]) -> None:
        """Raise DimensionMismatchError if the vector has the wrong length."""
        if len(vector) != self._dimensions:
            msg = f"Expected {self._dimensions} dimensions, got {len(vector)}"
            raise DimensionMismatchError(msg)

    def insert(
        self,
        record_id: str,
        vector: list[float],
        metadata: dict[str, str | int | float | bool] | None = None,
    ) -> VectorRecord:
        """Insert a new vector record into the store.

        Args:
            record_id: A unique identifier for this record.
            vector: The vector (list of floats) to store.
            metadata: Optional key-value pairs of extra information.

        Returns:
            The newly created VectorRecord.

        Raises:
            DuplicateIDError: If a record with this ID already exists.
            DimensionMismatchError: If the vector has the wrong number of dimensions.

        """
        if record_id in self._records:
            msg = f"Record with ID '{record_id}' already exists"
            raise DuplicateIDError(msg)
        self._check_dimensions(vector)
        record = VectorRecord(
            record_id=record_id,
            vector=list(vector),
            metadata=dict(metadata) if metadata is not None else {},
        )
        self._records[record_id] = record
        return record

    def get(self, record_id: str) -> VectorRecord:
        """Get a record by its ID.

        Args:
            record_id: The ID of the record to retrieve.

        Returns:
            The matching VectorRecord.

        Raises:
            RecordNotFoundError: If no record with this ID exists.

        """
        if record_id not in self._records:
            msg = f"No record found with ID '{record_id}'"
            raise RecordNotFoundError(msg)
        return self._records[record_id]

    def update(
        self,
        record_id: str,
        vector: list[float] | None = None,
        metadata: dict[str, str | int | float | bool] | None = None,
    ) -> VectorRecord:
        """Update an existing record's vector and/or metadata.

        Args:
            record_id: The ID of the record to update.
            vector: New vector (optional -- keeps old if not provided).
            metadata: New metadata (optional -- keeps old if not provided).

        Returns:
            The updated VectorRecord.

        Raises:
            RecordNotFoundError: If no record with this ID exists.
            DimensionMismatchError: If the new vector has wrong dimensions.

        """
        record = self.get(record_id)
        if vector is not None:
            self._check_dimensions(vector)
            record.vector = list(vector)
        if metadata is not None:
            record.metadata = dict(metadata)
        return record

    def delete(self, record_id: str) -> None:
        """Remove a record from the store.

        Args:
            record_id: The ID of the record to delete.

        Raises:
            RecordNotFoundError: If no record with this ID exists.

        """
        if record_id not in self._records:
            msg = f"No record found with ID '{record_id}'"
            raise RecordNotFoundError(msg)
        del self._records[record_id]

    def list_records(self) -> list[VectorRecord]:
        """Return all records in the store as a list.

        Returns:
            A list of all VectorRecord objects in insertion order.

        """
        return list(self._records.values())

    def contains(self, record_id: str) -> bool:
        """Check whether a record with the given ID exists.

        Args:
            record_id: The ID to check for.

        Returns:
            True if the record exists, False otherwise.

        """
        return record_id in self._records

    def clear(self) -> None:
        """Remove all records from the store."""
        self._records.clear()

    def batch_insert(
        self,
        records: list[tuple[str, list[float], dict[str, str | int | float | bool] | None]],
    ) -> list[VectorRecord]:
        """Insert multiple records at once.

        Args:
            records: A list of (record_id, vector, metadata) tuples.

        Returns:
            A list of the newly created VectorRecord objects.

        Raises:
            DuplicateIDError: If any ID already exists or is duplicated in the batch.
            DimensionMismatchError: If any vector has wrong dimensions.

        """
        results: list[VectorRecord] = []
        for record_id, vector, metadata in records:
            results.append(self.insert(record_id, vector, metadata))
        return results
