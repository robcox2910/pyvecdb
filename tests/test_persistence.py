"""Tests for persistence (save/load)."""

import json
import tempfile
from pathlib import Path

import pytest

from pyvecdb.persistence import dict_to_store, load_store, save_store, store_to_dict
from pyvecdb.store import VectorStore

# Named constants
DIMS_TWO = 2
DIMS_THREE = 3
VEC_A = [1.0, 2.0]
VEC_B = [3.0, 4.0]
EXPECTED_COUNT_TWO = 2


class TestStoreToDict:
    """Tests for store_to_dict."""

    def test_basic(self) -> None:
        """Convert a store to a dictionary."""
        store = VectorStore(dimensions=DIMS_TWO)
        store.insert("a", VEC_A, metadata={"x": "y"})
        data = store_to_dict(store)
        assert data["dimensions"] == DIMS_TWO
        records = data["records"]
        assert isinstance(records, list)
        assert len(records) == 1

    def test_empty_store(self) -> None:
        """Convert an empty store to a dictionary."""
        store = VectorStore(dimensions=DIMS_THREE)
        data = store_to_dict(store)
        assert data["dimensions"] == DIMS_THREE
        records = data["records"]
        assert isinstance(records, list)
        assert len(records) == 0


class TestDictToStore:
    """Tests for dict_to_store."""

    def test_basic(self) -> None:
        """Create a store from a dictionary."""
        data: dict[str, object] = {
            "dimensions": DIMS_TWO,
            "records": [
                {"record_id": "a", "vector": VEC_A, "metadata": {"x": "y"}},
                {"record_id": "b", "vector": VEC_B, "metadata": {}},
            ],
        }
        store = dict_to_store(data)
        assert store.dimensions == DIMS_TWO
        assert store.count == EXPECTED_COUNT_TWO

    def test_invalid_records_type(self) -> None:
        """Raise TypeError when records is not a list."""
        data: dict[str, object] = {"dimensions": DIMS_TWO, "records": "bad"}
        with pytest.raises(TypeError, match="Records must be a list"):
            dict_to_store(data)

    def test_invalid_record_type(self) -> None:
        """Raise TypeError when a record is not a dict."""
        data: dict[str, object] = {"dimensions": DIMS_TWO, "records": ["bad"]}
        with pytest.raises(TypeError, match="Each record must be a dict"):
            dict_to_store(data)


class TestSaveAndLoad:
    """Tests for save_store and load_store."""

    def test_roundtrip(self) -> None:
        """Save and load a store with records."""
        store = VectorStore(dimensions=DIMS_TWO)
        store.insert("a", VEC_A, metadata={"genre": "rock"})
        store.insert("b", VEC_B, metadata={"genre": "pop"})

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.json"
            save_store(store, path)

            loaded = load_store(path)
            assert loaded.dimensions == DIMS_TWO
            assert loaded.count == EXPECTED_COUNT_TWO
            assert loaded.get("a").vector == VEC_A
            assert loaded.get("a").metadata["genre"] == "rock"

    def test_save_creates_valid_json(self) -> None:
        """Save creates valid JSON file."""
        store = VectorStore(dimensions=DIMS_TWO)
        store.insert("x", VEC_A)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.json"
            save_store(store, path)

            raw = json.loads(path.read_text(encoding="utf-8"))
            assert "dimensions" in raw
            assert "records" in raw

    def test_load_invalid_json_root(self) -> None:
        """Raise TypeError when JSON root is not an object."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "bad.json"
            path.write_text("[1, 2, 3]", encoding="utf-8")
            with pytest.raises(TypeError, match="JSON root must be an object"):
                load_store(path)

    def test_save_load_string_path(self) -> None:
        """Save and load using string paths."""
        store = VectorStore(dimensions=DIMS_TWO)
        store.insert("a", VEC_A)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "test.json")
            save_store(store, path)

            loaded = load_store(path)
            assert loaded.count == 1
