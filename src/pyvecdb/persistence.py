"""Save and load vector stores to/from JSON files.

Think of persistence like saving your game -- you can close the program,
come back later, and pick up right where you left off. Without persistence,
all your vectors would vanish when the program stops!

The JSON format looks like this:

.. code-block:: json

    {
        "dimensions": 3,
        "records": [
            {
                "record_id": "song-1",
                "vector": [0.8, 0.2, 0.5],
                "metadata": {"genre": "rock"}
            }
        ]
    }

Example:
    >>> from pyvecdb.store import VectorStore
    >>> from pyvecdb.persistence import save_store, load_store
    >>> store = VectorStore(dimensions=2)
    >>> store.insert("a", [1.0, 2.0])
    >>> save_store(store, "/tmp/my_vectors.json")
    >>> loaded = load_store("/tmp/my_vectors.json")
    >>> loaded.count
    1

"""

import json
from pathlib import Path
from typing import cast

from pyvecdb.atomic import atomic_write_text
from pyvecdb.errors import PersistenceError
from pyvecdb.record import VectorRecord
from pyvecdb.store import VectorStore

# Every real save file has these top-level keys; anything else isn't one of ours.
_REQUIRED_KEYS = frozenset({"dimensions", "records"})


def store_to_dict(
    store: VectorStore,
) -> dict[str, int | list[dict[str, str | list[float] | dict[str, str | int | float | bool]]]]:
    """Convert a VectorStore to a plain dictionary.

    Args:
        store: The vector store to serialize.

    Returns:
        A dictionary with 'dimensions' and 'records' keys.

    """
    return {
        "dimensions": store.dimensions,
        "records": [r.to_dict() for r in store.list_records()],
    }


def dict_to_store(data: dict[str, object]) -> VectorStore:
    """Create a VectorStore from a dictionary.

    Args:
        data: A dictionary with 'dimensions' and 'records' keys.

    Returns:
        A new VectorStore populated with the records.

    """
    dimensions = int(data["dimensions"])  # type: ignore[arg-type]
    store = VectorStore(dimensions=dimensions)
    raw_records = data.get("records", [])
    if not isinstance(raw_records, list):
        msg = "Records must be a list"
        raise TypeError(msg)
    for raw in raw_records:  # pyright: ignore[reportUnknownVariableType]
        if not isinstance(raw, dict):
            msg = "Each record must be a dict"
            raise TypeError(msg)
        rec_data: dict[str, object] = {str(k): v for k, v in raw.items()}  # pyright: ignore[reportUnknownVariableType, reportUnknownArgumentType]
        record = VectorRecord.from_dict(rec_data)
        store.insert(record.record_id, record.vector, record.metadata)
    return store


def save_store(store: VectorStore, path: str | Path) -> None:
    """Save a vector store to a JSON file.

    Args:
        store: The vector store to save.
        path: The file path to write to.

    """
    file_path = Path(path)
    data = store_to_dict(store)
    atomic_write_text(file_path, json.dumps(data, indent=2))


def load_store(path: str | Path) -> VectorStore:
    """Load a vector store from a JSON file.

    Args:
        path: The file path to read from.

    Returns:
        A new VectorStore populated with the saved records.

    Raises:
        PersistenceError: If the file is missing, unreadable, not valid JSON,
            or not a valid save file.

    """
    file_path = Path(path)
    try:
        data: object = json.loads(file_path.read_text(encoding="utf-8"))
    except OSError as exc:
        msg = f"Could not read {file_path}: {exc}"
        raise PersistenceError(msg) from exc
    except json.JSONDecodeError as exc:
        msg = f"Invalid JSON in {file_path}: {exc}"
        raise PersistenceError(msg) from exc
    if not isinstance(data, dict):
        msg = f"{file_path} is not a valid save file: the top level must be an object"
        raise PersistenceError(msg)
    saved: dict[str, object] = {str(k): v for k, v in cast("dict[object, object]", data).items()}
    missing = _REQUIRED_KEYS - saved.keys()
    if missing:
        msg = f"{file_path} is not a valid save file: missing {sorted(missing)}"
        raise PersistenceError(msg)
    try:
        return dict_to_store(saved)
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        msg = f"{file_path} is not a valid save file: {exc}"
        raise PersistenceError(msg) from exc
