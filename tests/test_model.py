"""Model check -- VectorStore must behave exactly like a plain Python dict.

A property test checks one rule for many inputs. A *model check* goes
further: Hypothesis invents a random *story* -- insert this, update that,
delete the other, search, clear everything -- and plays it on two things at
once:

- the real ``VectorStore``;
- a **model**: a plain ``dict`` from each ID to its vector and metadata,
  which we already trust to be right.

After every step the two must agree. If they ever disagree, Hypothesis
shrinks the story to the shortest list of steps that still goes wrong.

The model is deliberately *not* a copy of ``VectorStore``'s code. It is the
simplest thing that is obviously correct, so when the two disagree, the
store is the one to suspect. Searches are checked by measuring every vector
in the dict by hand (``how_far`` in ``tests/strategies.py``) and sorting.

See ``docs/property-testing.md`` for the walk-through.
"""

import unittest
from typing import cast

import pytest
from hypothesis import settings
from hypothesis import strategies as st
from hypothesis.stateful import Bundle, RuleBasedStateMachine, invariant, precondition, rule

from pyvecdb.errors import (
    DimensionMismatchError,
    DuplicateIDError,
    EmptyStoreError,
    RecordNotFoundError,
)
from pyvecdb.filtering import MetadataValue
from pyvecdb.search import knn_search
from pyvecdb.store import VectorStore
from tests.strategies import (
    any_metadata,
    how_far,
    record_ids,
    search_metrics,
    search_vectors,
)

# Every story uses a 2D store: a flat map, like a real pizza-shop map.
DIMENSIONS = 2

# Batches stay short: three records show every all-or-nothing case, and big ones only slow the run.
MOST_IN_A_BATCH = 3

Metadata = dict[str, MetadataValue]


def map_spots() -> st.SearchStrategy[list[float]]:
    """Make a 2D vector of small whole numbers: a spot on the map."""
    return search_vectors(DIMENSIONS)


def wrong_sized_vectors() -> st.SearchStrategy[list[float]]:
    """Make a vector the store must refuse: any length except 2."""
    return st.one_of(search_vectors(1), search_vectors(3))


def maybe_metadata() -> st.SearchStrategy[Metadata | None]:
    """Make new metadata, or None for "leave the metadata alone"."""
    return st.one_of(st.none(), any_metadata())


Record = tuple[str, list[float]]
Batch = list[Record]


def good_records() -> st.SearchStrategy[Batch]:
    """Make a short list of (ID, vector) pairs with brand-new IDs and the right-sized vectors."""
    return st.lists(st.tuples(record_ids(), map_spots()), max_size=MOST_IN_A_BATCH)


def mixed_batches(ids_to_use: st.SearchStrategy[str]) -> st.SearchStrategy[Batch]:
    """Make a short batch of IDs from ``ids_to_use``, with the odd wrong-sized vector."""
    return st.lists(
        st.tuples(ids_to_use, st.one_of(map_spots(), wrong_sized_vectors())),
        max_size=MOST_IN_A_BATCH,
    )


@st.composite
def batches_with_one_bad_record(draw: st.DrawFn, bad: st.SearchStrategy[Record]) -> Batch:
    """Make a batch of good records with one record drawn from ``bad`` somewhere in it."""
    # A random batch rarely puts the bad record *after* good ones, and that is the case
    # where a missing check leaves half a batch behind instead of refusing all of it.
    before, after = draw(good_records()), draw(good_records())
    return [*before, draw(bad), *after]


@st.composite
def batches_repeating_an_id(draw: st.DrawFn) -> Batch:
    """Make a batch of good records where one ID comes twice."""
    # Likewise, a fresh ID repeated inside one batch is rare by luck alone.
    batch = draw(good_records().filter(bool))
    which = draw(st.integers(min_value=0, max_value=len(batch) - 1))
    return [*batch, (batch[which][0], draw(map_spots()))]


# --- The machine: one story, played on the store and on a dict ----------------


class VectorStoreMachine(RuleBasedStateMachine):
    """Play random inserts, updates, deletes, reads and searches on a VectorStore and a dict."""

    # IDs we've used before. Brand-new random IDs almost never repeat, so
    # rules also pick from here to read, update or delete a record that
    # really is (or was) in the store.
    ids = Bundle("ids")

    def __init__(self) -> None:
        """Start with an empty 2D store and an empty dict."""
        super().__init__()
        self.store = VectorStore(dimensions=DIMENSIONS)
        self.model: dict[str, tuple[list[float], Metadata]] = {}

    @rule(
        target=ids,
        record_id=st.one_of(ids, record_ids()),
        vector=map_spots(),
        metadata=any_metadata(),
    )
    def insert(self, record_id: str, vector: list[float], metadata: Metadata) -> str:
        """Insert in both, or get DuplicateIDError if the dict already has that ID."""
        if record_id in self.model:
            with pytest.raises(DuplicateIDError):
                self.store.insert(record_id, vector, metadata)
            return record_id
        mine, my_metadata = list(vector), dict(metadata)
        record = self.store.insert(record_id, mine, my_metadata)
        assert (record.record_id, record.vector, record.metadata) == (record_id, vector, metadata)
        self.model[record_id] = (vector, metadata)
        # The store keeps its own copy: scribbling on ours afterwards changes nothing.
        mine.append(99.0)
        my_metadata["scribble"] = True
        return record_id

    @rule(record_id=st.one_of(ids, record_ids()), vector=wrong_sized_vectors())
    def insert_wrong_size(self, record_id: str, vector: list[float]) -> None:
        """Refuse a vector of the wrong size (or the ID, if it's already taken)."""
        expected = DuplicateIDError if record_id in self.model else DimensionMismatchError
        with pytest.raises(expected):
            self.store.insert(record_id, vector)

    @rule(
        record_id=st.one_of(ids, record_ids()),
        vector=st.one_of(st.none(), map_spots()),
        metadata=maybe_metadata(),
    )
    def update(self, record_id: str, vector: list[float] | None, metadata: Metadata | None) -> None:
        """Update in both: None leaves that part alone; a missing ID is RecordNotFoundError."""
        if record_id not in self.model:
            with pytest.raises(RecordNotFoundError):
                self.store.update(record_id, vector, metadata)
            return
        old_vector, old_metadata = self.model[record_id]
        self.model[record_id] = (
            old_vector if vector is None else vector,
            old_metadata if metadata is None else metadata,
        )
        mine = None if vector is None else list(vector)
        updated = self.store.update(record_id, mine, metadata)
        if mine is not None:
            # The store keeps its own copy: scribbling on ours afterwards changes nothing.
            mine.append(99.0)
        assert (updated.record_id, updated.vector, updated.metadata) == (
            record_id,
            *self.model[record_id],
        )

    @rule(record_id=st.one_of(ids, record_ids()))
    def delete(self, record_id: str) -> None:
        """Delete from both; a missing ID is RecordNotFoundError."""
        if record_id not in self.model:
            with pytest.raises(RecordNotFoundError):
                self.store.delete(record_id)
            return
        self.store.delete(record_id)
        del self.model[record_id]

    @rule(record_id=st.one_of(ids, record_ids()))
    def get(self, record_id: str) -> None:
        """Read one record: the dict's vector and metadata, or RecordNotFoundError."""
        assert self.store.contains(record_id) == (record_id in self.model)
        if record_id not in self.model:
            with pytest.raises(RecordNotFoundError):
                self.store.get(record_id)
            return
        record = self.store.get(record_id)
        assert record.record_id == record_id
        assert (record.vector, record.metadata) == self.model[record_id]

    @rule(record_id=ids, query=map_spots())
    def scribble_on_returned_records(self, record_id: str, query: list[float]) -> None:
        """Scribble on records that get() and knn_search() hand back: the store must not change."""
        if record_id not in self.model:
            return
        handed_back = [self.store.get(record_id)]
        handed_back += [
            record for record, _score in knn_search(self.store, query, k=1, metric="euclidean")
        ]
        for record in handed_back:
            record.vector.append(99.0)
            record.metadata["scribble"] = True
        # The invariant below then checks every record against the dict.

    @precondition(lambda self: bool(self.model))
    @rule(
        before=good_records(),
        which=st.integers(min_value=0),
        vector=map_spots(),
        after=good_records(),
    )
    def batch_insert_over_a_used_id(
        self, before: Batch, which: int, vector: list[float], after: Batch
    ) -> None:
        """Insert a batch with an ID the store really holds, after other good records."""
        # Picked from the dict, so the ID is certainly taken. A missing "already in the
        # store" check would insert `before` and only then fail, leaving half a batch.
        taken = list(self.model)[which % len(self.model)]
        self.insert_batch([*before, (taken, vector), *after])

    @rule(
        batch=st.one_of(
            mixed_batches(st.one_of(ids, record_ids())),
            batches_with_one_bad_record(st.tuples(record_ids(), wrong_sized_vectors())),
            batches_repeating_an_id(),
        )
    )
    def batch_insert(self, batch: Batch) -> None:
        """Insert a whole batch, or -- if any one record is bad -- none of it."""
        self.insert_batch(batch)

    def insert_batch(self, batch: Batch) -> None:
        """Insert ``batch`` in the store and the dict, or check the store refuses all of it."""
        ids_in_batch = [record_id for record_id, _vector in batch]
        all_good = (
            len(set(ids_in_batch)) == len(ids_in_batch)
            and not any(record_id in self.model for record_id in ids_in_batch)
            and all(len(vector) == DIMENSIONS for _record_id, vector in batch)
        )
        if not all_good:
            with pytest.raises((DuplicateIDError, DimensionMismatchError)):
                self.store.batch_insert([(record_id, vector, None) for record_id, vector in batch])
            return
        inserted = self.store.batch_insert(
            [(record_id, vector, None) for record_id, vector in batch]
        )
        assert [(record.record_id, record.vector, record.metadata) for record in inserted] == [
            (record_id, vector, {}) for record_id, vector in batch
        ]
        for record_id, vector in batch:
            self.model[record_id] = (vector, {})

    @rule(query=map_spots(), k=st.integers(min_value=1, max_value=5), metric=search_metrics())
    def search(self, query: list[float], k: int, metric: str) -> None:
        """Search: the k nearest by measuring every vector in the dict, or EmptyStoreError."""
        if not self.model:
            with pytest.raises(EmptyStoreError):
                knn_search(self.store, query, k=k, metric=metric)
            return
        found = [record.record_id for record, _score in knn_search(self.store, query, k, metric)]
        every_distance = sorted(how_far(vector, query, metric) for vector, _ in self.model.values())
        assert len(set(found)) == len(found)
        assert [how_far(self.model[shop][0], query, metric) for shop in found] == every_distance[:k]

    @rule()
    def clear(self) -> None:
        """Empty both."""
        self.store.clear()
        self.model.clear()

    @invariant()
    def same_records_in_the_same_order(self) -> None:
        """After every step, the store lists exactly the dict's records, oldest first."""
        assert self.store.count == len(self.model)
        listed = self.store.list_records()
        assert [record.record_id for record in listed] == list(self.model)
        assert [(record.vector, record.metadata) for record in listed] == list(self.model.values())


# why: 12 steps catch every bug we planted in the store; 50 (the default) took about
# 5.6 s, nearly all of the model's time. Each short story is also cheaper to shrink.
VectorStoreMachine.TestCase.settings = settings(stateful_step_count=12)
# pytest runs this like any other test class: one run plays 100 random stories.
# (Hypothesis doesn't say what type ``TestCase`` is, so ``cast`` tells pyright.)
TestVectorStoreModel = cast("type[unittest.TestCase]", VectorStoreMachine.TestCase)
