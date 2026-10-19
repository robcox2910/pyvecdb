"""Property-based tests -- rules that must hold for *every* input.

An ordinary test checks one example someone thought of: "put a pizza shop
at [1, 1] and one at [10, 10], search from [2, 2], and the first one comes
back first". A property test states a rule instead, such as "whatever shops
are on the map, the search finds the same nearest ones as measuring every
shop by hand", and lets Hypothesis invent hundreds of maps to try to break
it. If one breaks the rule, Hypothesis *shrinks* it down to the smallest
input that still fails, and shows you that.

Each test below says its rule in plain English in its docstring. The
strategies at the top are recipes for making random inputs; each one is
named for what it makes.

The step-by-step check of ``VectorStore`` against a plain dict lives in
``tests/test_model.py``. See ``docs/property-testing.md`` for the walk-through.
"""

import json
import math
import operator
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import cast

from hypothesis import given
from hypothesis import strategies as st

from pyvecdb.distance import dot_product, euclidean_distance, manhattan_distance
from pyvecdb.errors import PersistenceError
from pyvecdb.filtering import FilterCondition, MetadataFilter, MetadataValue
from pyvecdb.persistence import load_store, save_store
from pyvecdb.search import batch_search, knn_search
from pyvecdb.store import VectorStore
from tests.strategies import (
    any_metadata,
    how_far,
    record_ids,
    search_metrics,
    search_vectors,
)

# --- Strategies: recipes for random inputs -----------------------------------
# record_ids(), search_vectors(), any_metadata() and search_metrics() live in
# tests/strategies.py, shared with the model check.

# Every search map in this file has this many dimensions (a 3D map).
MAP_DIMENSIONS = 3

# A saved store has between 1 and this many dimensions, and at most this
# many records. Bigger stores are slow to invent and save, and find the same
# bugs; a failure shrinks to a small store anyway.
MOST_SAVED_DIMENSIONS = 5
MOST_SAVED_RECORDS = 5

# The longest JSON list or object the junk-file strategy makes. Bigger ones
# are slow to invent, and a broken file shrinks to a small one anyway.
MOST_JUNK_ITEMS = 3

Records = dict[str, tuple[list[float], dict[str, MetadataValue]]]


def shops_on_a_map() -> st.SearchStrategy[dict[str, list[float]]]:
    """Make a map of at least one shop: each ID with a 3D vector of small whole numbers."""
    return st.dictionaries(record_ids(), search_vectors(MAP_DIMENSIONS), min_size=1)


def where_you_stand() -> st.SearchStrategy[list[float]]:
    """Make a query vector: a 3D spot of small whole numbers to search from."""
    return search_vectors(MAP_DIMENSIONS)


def how_many_to_find() -> st.SearchStrategy[int]:
    """Make k, how many results to ask for: from 1 to more than there are shops."""
    return st.integers(min_value=1, max_value=12)


def any_coordinate() -> st.SearchStrategy[float]:
    """Make any float at all: decimals, huge, tiny, -0.0, infinity, even NaN."""
    return st.floats()


@st.composite
def saved_stores(draw: st.DrawFn) -> tuple[int, Records]:
    """Make the contents of one store: its dimensions, and records of any floats and metadata."""
    dimensions = draw(st.integers(min_value=1, max_value=MOST_SAVED_DIMENSIONS))
    vectors = st.lists(any_coordinate(), min_size=dimensions, max_size=dimensions)
    records = draw(
        st.dictionaries(
            record_ids(), st.tuples(vectors, any_metadata()), max_size=MOST_SAVED_RECORDS
        )
    )
    return dimensions, records


def number_vectors(dimensions: int) -> st.SearchStrategy[list[float]]:
    """Make a vector of ``dimensions`` floats: any number, including huge ones and infinity."""
    # NaN ("not a number") is left out of the *inputs*: a vector with a NaN
    # in it has no real distance to anything.
    return st.lists(st.floats(allow_nan=False), min_size=dimensions, max_size=dimensions)


@st.composite
def pairs_of_vectors(draw: st.DrawFn) -> tuple[list[float], list[float]]:
    """Make two vectors of the same length (1 to 5 numbers each)."""
    dimensions = draw(st.integers(min_value=1, max_value=MOST_SAVED_DIMENSIONS))
    return draw(number_vectors(dimensions)), draw(number_vectors(dimensions))


def distance_functions() -> st.SearchStrategy[Callable[[list[float], list[float]], float]]:
    """Pick one of the measures that work on any two vectors, even all-zero ones."""
    # cosine_similarity refuses a vector of all zeros, so it isn't here.
    return st.sampled_from([euclidean_distance, manhattan_distance, dot_product])


def any_bytes() -> st.SearchStrategy[bytes]:
    """Make any run of raw bytes, including the empty one and ones that aren't text."""
    return st.binary()


def lists_and_objects_of(
    children: st.SearchStrategy[object],
) -> st.SearchStrategy[list[object] | dict[str, object]]:
    """Make a short JSON list, or a short JSON object, holding values made by ``children``."""
    return st.one_of(
        st.lists(children, max_size=MOST_JUNK_ITEMS),
        st.dictionaries(st.text(), children, max_size=MOST_JUNK_ITEMS),
    )


def any_json() -> st.SearchStrategy[object]:
    """Make any small JSON value: null, true/false, numbers, text, lists and objects."""
    return st.recursive(
        st.one_of(st.none(), st.booleans(), st.integers(), st.floats(), st.text()),
        lists_and_objects_of,
        max_leaves=MOST_JUNK_ITEMS,
    )


def any_number() -> st.SearchStrategy[int | float]:
    """Make any JSON number: a whole number of any size, or any float."""
    return st.one_of(st.integers(), st.floats())


def save_shaped_json() -> st.SearchStrategy[dict[str, object]]:
    """Make JSON that looks like a save file on the outside, with anything at all inside."""
    # Random JSON almost never has both "dimensions" and "records", so the
    # loader would refuse it at the door. This gets past the door.
    # Small whole numbers and lists of numbers turn up often, so the junk gets
    # past the first few checks and reaches the deeper ones too.
    junk_vector = st.one_of(any_json(), st.lists(any_number(), max_size=MOST_JUNK_ITEMS))
    junk_record = st.fixed_dictionaries(
        {"record_id": any_json(), "vector": junk_vector},
        optional={"metadata": any_json()},
    )
    return st.fixed_dictionaries(
        {
            "dimensions": st.one_of(st.integers(min_value=0, max_value=3), any_json()),
            "records": st.one_of(any_json(), st.lists(junk_record, max_size=MOST_JUNK_ITEMS)),
        }
    )


# --- Helpers -----------------------------------------------------------------


def store_of_shops(shops: dict[str, list[float]]) -> VectorStore:
    """Return a 3D VectorStore holding every shop in ``shops``."""
    store = VectorStore(dimensions=MAP_DIMENSIONS)
    for shop_id, vector in shops.items():
        store.insert(shop_id, vector)
    return store


# --- Filtered search: shops with a colour and a size --------------------------
# Metadata for the filter properties is a small, fixed shape so that filters
# match some shops and miss others, instead of almost never matching.
COLOURS = ("red", "green", "blue")
MOST_SIZE = 3


def tagged_shops() -> st.SearchStrategy[dict[str, tuple[list[float], dict[str, MetadataValue]]]]:
    """Make at least one shop: each ID with a 3D spot and maybe a colour and a size."""
    fields: dict[str, st.SearchStrategy[MetadataValue]] = {
        "colour": st.sampled_from(COLOURS),
        "size": st.integers(0, MOST_SIZE),
    }
    # Most shops have both tags; some miss one or both, and must never match.
    tags = st.one_of(
        st.fixed_dictionaries(fields),
        st.fixed_dictionaries(fields),
        st.fixed_dictionaries({}, optional=fields),
    )
    return st.dictionaries(
        record_ids(), st.tuples(search_vectors(MAP_DIMENSIONS), tags), min_size=1, max_size=5
    )


def conditions_on(values: st.SearchStrategy[MetadataValue]) -> st.SearchStrategy[FilterCondition]:
    """Make one condition: a plain value, one operator, two operators, or "in a list"."""
    operators = st.sampled_from(["$eq", "$ne", "$gt", "$gte", "$lt", "$lte"])
    conditions = st.one_of(
        values,
        st.dictionaries(operators, values, min_size=1, max_size=1),
        st.dictionaries(operators, values, min_size=1, max_size=1),
        st.dictionaries(operators, values, min_size=2, max_size=2),
        st.fixed_dictionaries({"$in": st.lists(values, max_size=3)}),
    )
    # Hypothesis builds dicts pyright can't see as FilterCondition (dict values are invariant).
    return cast("st.SearchStrategy[FilterCondition]", conditions)


def filter_conditions() -> st.SearchStrategy[MetadataFilter]:
    """Make a filter on colour and/or size, mostly with the right kind of value for the field."""
    colour = st.sampled_from(COLOURS)
    size = st.integers(-1, MOST_SIZE + 1)
    # Now and then the wrong kind (a colour asked to be "greater than" 3) must match nothing.
    on_size = conditions_on(st.one_of(size, size, size, colour))
    on_colour = conditions_on(st.one_of(colour, colour, colour, size))
    return st.one_of(
        st.fixed_dictionaries({"size": on_size}),
        st.fixed_dictionaries({"colour": on_colour}),
        st.fixed_dictionaries({"colour": on_colour, "size": on_size}),
    )


def passes(actual: MetadataValue, operator_name: str, expected: object) -> bool:
    """Say whether one value passes one operator: written out by hand, not PyVecDB's code."""
    if operator_name == "$eq":
        return actual == expected
    if operator_name == "$ne":
        return actual != expected
    if operator_name == "$in":
        return isinstance(expected, list) and actual in expected
    # The four comparisons only make sense between two numbers.
    if not isinstance(actual, int) or not isinstance(expected, int):
        return False
    return {
        "$gt": actual > expected,
        "$gte": actual >= expected,
        "$lt": actual < expected,
        "$lte": actual <= expected,
    }[operator_name]


def shop_passes(tags: dict[str, MetadataValue], wanted: MetadataFilter) -> bool:
    """Say whether a shop's tags pass every condition; a missing field never passes."""
    for field, condition in wanted.items():
        if field not in tags:
            return False
        operators = condition if isinstance(condition, dict) else {"$eq": condition}
        if not all(passes(tags[field], op, expected) for op, expected in operators.items()):
            return False
    return True


def load_or_refuse(path: Path) -> None:
    """Load the store at ``path``, accepting only a PersistenceError refusal.

    A PersistenceError is caught and ignored: refusing a bad file is fine. Any
    *other* exception escapes and fails the test, and that is the whole check.
    """
    try:
        load_store(path)
    except PersistenceError:
        return


# --- Search: the nearest shops, the same as measuring every one by hand -------


@given(
    shops=shops_on_a_map(), query=where_you_stand(), k=how_many_to_find(), metric=search_metrics()
)
def test_top_k_matches_a_brute_force_sort(
    shops: dict[str, list[float]], query: list[float], k: int, metric: str
) -> None:
    """knn_search finds k shops exactly as near as the k nearest found by measuring every shop."""
    results = knn_search(store_of_shops(shops), query, k=k, metric=metric)
    found = [record.record_id for record, _score in results]
    every_distance = sorted(how_far(vector, query, metric) for vector in shops.values())
    assert len(set(found)) == len(found)
    # Compare distances, not IDs: two shops the same distance away are a tie,
    # and either one may come first.
    assert [how_far(shops[shop_id], query, metric) for shop_id in found] == every_distance[:k]


@given(
    shops=tagged_shops(),
    wanted=filter_conditions(),
    query=where_you_stand(),
    k=how_many_to_find(),
    metric=search_metrics(),
)
def test_filtered_top_k_matches_a_brute_force_sort_of_the_matching_shops(
    shops: dict[str, tuple[list[float], dict[str, MetadataValue]]],
    wanted: MetadataFilter,
    query: list[float],
    k: int,
    metric: str,
) -> None:
    """A filtered knn_search finds the nearest shops among only those whose tags match."""
    store = VectorStore(dimensions=MAP_DIMENSIONS)
    for shop_id, (vector, tags) in shops.items():
        store.insert(shop_id, vector, tags)
    results = knn_search(store, query, k=k, metric=metric, metadata_filter=wanted)
    found = [record.record_id for record, _score in results]
    matching = {s: v for s, (v, tags) in shops.items() if shop_passes(tags, wanted)}
    every_distance = sorted(how_far(vector, query, metric) for vector in matching.values())
    assert len(set(found)) == len(found)
    assert set(found) <= set(matching)
    assert [how_far(matching[shop_id], query, metric) for shop_id in found] == every_distance[:k]


@given(
    shops=tagged_shops(),
    wanted=st.one_of(st.none(), filter_conditions()),
    queries=st.lists(where_you_stand(), max_size=4),
    k=how_many_to_find(),
    metric=search_metrics(),
)
def test_batch_search_is_one_knn_search_per_query(
    shops: dict[str, tuple[list[float], dict[str, MetadataValue]]],
    wanted: MetadataFilter | None,
    queries: list[list[float]],
    k: int,
    metric: str,
) -> None:
    """batch_search gives, for each query, exactly what knn_search gives for it alone."""
    store = VectorStore(dimensions=MAP_DIMENSIONS)
    for shop_id, (vector, tags) in shops.items():
        store.insert(shop_id, vector, tags)
    batched = batch_search(store, queries, k=k, metric=metric, metadata_filter=wanted)
    alone = [knn_search(store, q, k=k, metric=metric, metadata_filter=wanted) for q in queries]
    assert len(batched) == len(queries)
    assert [[(r.record_id, score) for r, score in each] for each in batched] == [
        [(r.record_id, score) for r, score in each] for each in alone
    ]


@given(
    sizes=st.dictionaries(record_ids(), st.integers(0, MOST_SIZE), min_size=1, max_size=6),
    operator_name=st.sampled_from(["$gt", "$gte", "$lt", "$lte"]),
    line=st.integers(0, MOST_SIZE),
    query=where_you_stand(),
)
def test_size_comparisons_keep_exactly_the_right_shops(
    sizes: dict[str, int], operator_name: str, line: int, query: list[float]
) -> None:
    """Asking for every shop with a size over/under a line returns exactly those, ties on the line included or not."""
    compare = {"$gt": operator.gt, "$gte": operator.ge, "$lt": operator.lt, "$lte": operator.le}
    store = VectorStore(dimensions=MAP_DIMENSIONS)
    for shop_id, size in sizes.items():
        store.insert(shop_id, [0.0] * MAP_DIMENSIONS, {"size": size})
    results = knn_search(
        store,
        query,
        k=len(sizes),
        metric="manhattan",
        metadata_filter={"size": {operator_name: line}},
    )
    found = {record.record_id for record, _score in results}
    assert found == {s for s, size in sizes.items() if compare[operator_name](size, line)}


@given(shops=shops_on_a_map(), query=where_you_stand(), metric=search_metrics())
def test_scores_are_the_real_distances(
    shops: dict[str, list[float]], query: list[float], metric: str
) -> None:
    """Every score knn_search reports is that shop's real distance (made negative for distances)."""
    results = knn_search(store_of_shops(shops), query, k=len(shops), metric=metric)
    for record, score in results:
        distance = how_far(shops[record.record_id], query, metric)
        # how_far gives squared distance for euclidean, so take its square root.
        # For dot it gives minus the dot product, so minus that is the dot product.
        assert score == (-math.sqrt(distance) if metric == "euclidean" else -distance)


@given(pair=pairs_of_vectors(), distance=distance_functions())
def test_distance_is_the_same_both_ways(
    pair: tuple[list[float], list[float]],
    distance: Callable[[list[float], list[float]], float],
) -> None:
    """Measuring from a to b gives exactly the same answer as measuring from b to a."""
    a, b = pair
    forward, backward = distance(a, b), distance(b, a)
    # Huge numbers can add up to infinity minus infinity, which is NaN, and
    # NaN is never equal to anything, not even itself. So two NaNs count as
    # the same answer; any other pair must be exactly equal.
    assert forward == backward or (math.isnan(forward) and math.isnan(backward))


# --- Save and load: whatever you save comes back -----------------------------


@given(contents=saved_stores())
def test_save_then_load_gives_back_the_same_store(contents: tuple[int, Records]) -> None:
    """Loading a saved store gives back the same dimensions and records, in the same order."""
    dimensions, records = contents
    store = VectorStore(dimensions=dimensions)
    for record_id, (vector, metadata) in records.items():
        store.insert(record_id, vector, metadata)
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "store.json"
        save_store(store, path)
        loaded = load_store(path)
    assert loaded.dimensions == dimensions
    # ``repr`` spells out every ID, number and metadata value, and tells
    # True from 1 and 1 from 1.0. It also writes NaN as ``nan`` both times,
    # where ``==`` would say NaN isn't even equal to itself.
    assert repr(loaded.list_records()) == repr(store.list_records())


# --- Loading junk: a bad save file is refused, never a crash ------------------


@given(junk=any_bytes())
def test_any_bytes_on_disk_load_or_raise_persistence_error(junk: bytes) -> None:
    """Whatever bytes are in the file, loading gives a store or a PersistenceError, never another error."""
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "store.json"
        path.write_bytes(junk)
        load_or_refuse(path)


@given(document=save_shaped_json())
def test_any_save_shaped_json_load_or_raise_persistence_error(document: dict[str, object]) -> None:
    """Whatever is inside a save-shaped JSON file, loading gives a store or a PersistenceError."""
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "store.json"
        path.write_text(json.dumps(document), encoding="utf-8")
        load_or_refuse(path)
