"""Vector record -- a vector with an ID and extra info.

Think of a ``VectorRecord`` like a trading card:

- The **ID** is the card number (unique identifier)
- The **vector** is the card's stats (a list of numbers)
- The **metadata** is the extra info printed on the card (name, team, year)

When you search a vector database, you get back records -- so you know
*which* item matched, not just *where* it is in space.

Example:
    >>> from pyvecdb.record import VectorRecord
    >>> card = VectorRecord(
    ...     record_id="card-42",
    ...     vector=[0.8, 0.2, 0.5],
    ...     metadata={"name": "Pikachu", "type": "electric"},
    ... )
    >>> card.record_id
    'card-42'
    >>> card.dimensions
    3

"""

from dataclasses import dataclass, field

from pyvecdb.errors import EmptyVectorError


@dataclass
class VectorRecord:
    """A vector paired with an ID and optional metadata.

    Attributes:
        record_id: A unique string identifier for this record.
        vector: The list of floats representing this record's position in space.
        metadata: Optional key-value pairs with extra info about this record.

    """

    record_id: str
    vector: list[float]
    metadata: dict[str, str | int | float | bool] = field(default_factory=lambda: {})

    def __post_init__(self) -> None:
        """Validate the record after creation."""
        if not self.vector:
            msg = "Vector must have at least one component"
            raise EmptyVectorError(msg)

    @property
    def dimensions(self) -> int:
        """Return the number of dimensions in this record's vector."""
        return len(self.vector)

    def to_dict(self) -> dict[str, str | list[float] | dict[str, str | int | float | bool]]:
        """Convert this record to a plain dictionary for serialization.

        Returns:
            A dictionary with 'record_id', 'vector', and 'metadata' keys.

        """
        return {
            "record_id": self.record_id,
            "vector": list(self.vector),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> VectorRecord:
        """Create a VectorRecord from a dictionary.

        Args:
            data: A dictionary with 'record_id', 'vector', and optionally 'metadata'.

        Returns:
            A new VectorRecord instance.

        """
        record_id = str(data["record_id"])
        raw_vector = data["vector"]
        if not isinstance(raw_vector, list):
            msg = "Vector must be a list"
            raise TypeError(msg)
        vector = [float(x) for x in raw_vector]  # pyright: ignore[reportUnknownVariableType, reportUnknownArgumentType]
        raw_metadata = data.get("metadata")
        if not isinstance(raw_metadata, dict):
            raw_metadata = {}
        metadata: dict[str, str | int | float | bool] = {}
        for k, v in raw_metadata.items():  # pyright: ignore[reportUnknownVariableType]
            if isinstance(v, (str, int, float, bool)):
                metadata[str(k)] = v  # pyright: ignore[reportUnknownArgumentType]
        return cls(record_id=record_id, vector=vector, metadata=metadata)
