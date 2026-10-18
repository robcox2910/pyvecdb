# Property Testing (The Nearest Shop Really Is the Nearest)

## The Pizza Scout

Remember the [pizza shop problem](concepts/search.md)? You stand somewhere
in town, and PyVecDB finds the 5 pizza shops closest to you.

How would you check that it works? The usual way: put three shops on a
map, stand at one spot, ask for the 2 closest, and check you get the two
you expected. That's an **example test**. It's useful, but it only tells
you about the one map you thought of.

Now imagine a pizza scout. All day long the scout draws new maps --
one shop, forty shops, shops piled on the same corner, shops with no name
at all, shops named in emoji -- stands somewhere random, asks the app for
the nearest few, and then does it the slow way: walks to *every* shop with
a measuring wheel, writes the distances down, and sorts them. For each map,
the scout checks a single rule:

> The shops the app finds are exactly as near as the nearest shops on my list.

That rule is a **property**. It doesn't say which shop should win on any
one map. It says something that must be true for *every* map.

When the scout finds a map that breaks the rule, they don't just report
"something went wrong on that huge map from this morning". They rub out
shops one at a time, and move the rest closer to the middle, checking it
still goes wrong each time, until the map is as simple as it can be. That
boiling-down is called **shrinking**, and it's the best part: the smallest
failing map usually points straight at the bug.

In Python, the scout is a library called
[Hypothesis](https://hypothesis.readthedocs.io/). PyVecDB's property tests
live in `tests/test_properties.py`, next to the ordinary example tests, and
the model check (more on that below) is in `tests/test_model.py`.

## A Property from PyVecDB

Here's the scout's rule as a test, with the recipes it uses:

```python
def whole_coordinates() -> st.SearchStrategy[float]:
    """Make one small whole number, written as a float, such as -3.0 or 7.0."""
    return st.integers(SMALLEST_COORDINATE, LARGEST_COORDINATE).map(float)


def shops_on_a_map() -> st.SearchStrategy[dict[str, list[float]]]:
    """Make a map of at least one shop: each ID with a 3D vector of small whole numbers."""
    return st.dictionaries(record_ids(), search_vectors(MAP_DIMENSIONS), min_size=1)


def where_you_stand() -> st.SearchStrategy[list[float]]:
    """Make a query vector: a 3D spot of small whole numbers to search from."""
    return search_vectors(MAP_DIMENSIONS)


def how_many_to_find() -> st.SearchStrategy[int]:
    """Make k, how many results to ask for: from 1 to more than there are shops."""
    return st.integers(min_value=1, max_value=12)


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
```

Line by line:

- **The functions at the top** are **strategies**: recipes for making
  random inputs. `shops_on_a_map()` makes a dictionary from shop IDs to
  3D spots. The IDs come from `record_ids()`, which is `st.text()`: any
  string at all, including `""`, `" "` and `"🍕"`. Each spot is three
  numbers from `whole_coordinates()`. Each recipe is a named function so
  the test reads like a sentence.
- **Why whole numbers?** Adding and multiplying numbers like `3.0` and
  `-7.0` is exact on a computer, so the scout's answer and the app's
  answer can be compared with `==`. With messy decimals the two could
  differ in the last digit, and we'd need a "close enough" fudge that
  might hide a real mistake. Numbers run from -10 to 10, so ties (two
  shops the same distance away) happen all the time. That's on purpose.
- **`search_metrics()`** picks `"euclidean"` (straight line),
  `"manhattan"` (city blocks) or `"dot"` (agreement) from
  [the distance page](concepts/distance.md). Cosine similarity divides by
  square roots, so its scores aren't exact, and it refuses an all-zero
  vector, so it isn't in this test.
- **`@given(...)`** tells Hypothesis to run the test many times (100 in a
  normal run), each time with a fresh map, spot, `k` and metric.
- **`store_of_shops(shops)`** is a small helper that makes a 3D
  `VectorStore` and inserts every shop.
- **`how_far`** (in `tests/strategies.py`) is the measuring wheel. It is
  *not* PyVecDB's code: it works in whole numbers, so nothing is ever
  rounded, and it says "smaller is nearer" for every metric. `sorted(...)`
  then lines up every shop's distance, nearest first. That's the scout's
  list.
- **The first `assert`** says no shop comes back twice.
- **The second `assert`** is the rule itself. It compares *distances*,
  not names. If two shops tie for nearest, the app may pick either one,
  and both answers are right. Comparing IDs would fail on a correct app;
  comparing distances accepts either winner but still catches a wrong one.

Notice what the test *doesn't* say. It never says "from `[2, 2]` the
nearest is `slice-heaven`". It doesn't need to know any answers. It only
needs a rule that's always true.

**A search that does nothing -- hands back an empty list every time --
fails this straight away**, because the scout's list always has at least
one shop on it.

PyVecDB has five more properties:

| Rule | Test |
|---|---|
| Every score `knn_search` reports is that shop's real distance (made negative for distances) | `test_scores_are_the_real_distances` |
| Measuring from a to b gives exactly the same answer as measuring from b to a | `test_distance_is_the_same_both_ways` |
| Loading a saved store gives back the same dimensions and records, in the same order | `test_save_then_load_gives_back_the_same_store` |
| Whatever bytes are in the file, loading gives a store or a `PersistenceError`, never another error | `test_any_bytes_on_disk_load_or_raise_persistence_error` |
| Whatever is inside a save-shaped JSON file, loading gives a store or a `PersistenceError` | `test_any_save_shaped_json_load_or_raise_persistence_error` |

The last two are about damaged save files. A program using PyVecDB writes
`except PersistenceError:` to handle one; if some other error sneaks out,
that program crashes instead.

## Checking Against a Model

A property checks one action. But a store is something you change step by
step: insert, update, delete, read, search, clear. Bugs often hide in the
*order* of steps -- updating a record you just deleted, or inserting an ID
twice.

So the scout has a second trick. They carry a notebook and write down
every shop and its spot by hand. After every single change to the app,
they compare the app with the notebook. If the two ever disagree,
something's wrong with the app.

That notebook is a **model**: something simple that we already trust.
For `VectorStore`, the model is a plain Python `dict` from each ID to its
vector and metadata. Here's the heart of `tests/test_model.py`:

```python
class VectorStoreMachine(RuleBasedStateMachine):
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
        return record_id

    @invariant()
    def same_records_in_the_same_order(self) -> None:
        """After every step, the store lists exactly the dict's records, oldest first."""
        assert self.store.count == len(self.model)
        listed = self.store.list_records()
        assert [record.record_id for record in listed] == list(self.model)
        assert [(record.vector, record.metadata) for record in listed] == list(self.model.values())


# why: 20 steps find the same bugs; 50 (the default) is most of the run time
VectorStoreMachine.TestCase.settings = settings(stateful_step_count=20)
# pytest runs this like any other test class: one run plays 100 random stories.
TestVectorStoreModel = cast("type[unittest.TestCase]", VectorStoreMachine.TestCase)
```

Piece by piece:

- **`RuleBasedStateMachine`** is Hypothesis's tool for testing things
  that change step by step. Hypothesis makes up a random **story** of
  steps and plays it.
- **`self.store` and `self.model`** are the app and the notebook. Every
  story starts with both empty.
- **Each `@rule`** is one kind of step. It does the same thing to the
  store and to the dict, and checks what the store hands back: `insert`
  must return a record with exactly the ID, vector and metadata it was
  given. The real file has rules to insert, insert a vector of the wrong
  size, update, delete, get, insert a batch (all of it or none of it),
  search, and clear. The real `insert` also scribbles on its own copy of
  the vector afterwards, to check the store kept a copy of its own.
- **`ids = Bundle("ids")`** is a bag of IDs used earlier in the story.
  Two random strings hardly ever match, so without it the story would
  almost never update or delete a record that is really there. `insert`
  puts its ID in the bag (`target=ids`). It picks its ID with
  `st.one_of(ids, record_ids())`: an old one from the bag *or* a brand-new
  one, so it tries both a fresh ID and one that's already taken.
- **The `@invariant()`** runs after every single step. It checks the
  count, the IDs in order (oldest first, like a `dict`), and every vector
  and metadata.
- **`TestVectorStoreModel = ...TestCase`** turns the machine into a test
  that pytest can find and run. One run plays 100 random stories, each up
  to 20 steps long.

The `search` rule uses the same measuring wheel as the property above:
it measures every vector in the notebook and sorts.

The `dict` is a good model because it's obviously right and it's *not*
`VectorStore`'s code. If we built the model by copying `VectorStore`, any
bug would be in both, and they'd agree perfectly about the wrong answer.

## Try It Yourself

Let's break saving on purpose and watch the pizza scout find it.

Open `src/pyvecdb/persistence.py`, find `store_to_dict`, and make it write
the records backwards:

```python
        "records": [r.to_dict() for r in reversed(store.list_records())],  # was: store.list_records()
```

Run the property tests:

```bash
uv run pytest tests/test_properties.py tests/test_model.py --no-cov
```

Hypothesis reports this (trimmed to the interesting part):

```
contents = (1, {'': ([0.0], {}), '0': ([0.0], {})})

    @given(contents=saved_stores())
    def test_save_then_load_gives_back_the_same_store(contents: tuple[int, Records]) -> None:
        """Loading a saved store gives back the same dimensions and records, in the same order."""
>       assert repr(loaded.list_records()) == repr(store.list_records())
E       AssertionError: assert '[VectorRecor...metadata={})]' == '[VectorRecor...metadata={})]'
E
E         - [VectorRecord(record_id='', vector=[0.0], metadata={}), VectorRecord(record_id='0', vector=[0.0], metadata={})]
E         + [VectorRecord(record_id='0', vector=[0.0], metadata={}), VectorRecord(record_id='', vector=[0.0], metadata={})]
E       Failing test case: test_save_then_load_gives_back_the_same_store(
E           # The test always failed when commented parts were varied together.
E           contents=(1, {'': ([0.0], {}), '0': ([0.0], {})}),
E       )
```

Look at how small that is. A one-dimensional store, two records named
`''` and `'0'`, both sitting at `[0.0]`, no metadata. You can't swap the
order of fewer than two records, so this is the smallest store that can
show the bug. Every record still comes back, with the right vector and
metadata -- just in the wrong order. Why does order matter? When two shops
tie, `knn_search` lists the older one first, so a store that loaded
backwards would quietly change search results.

Only the round-trip catches it. The other properties never save a file,
and the junk-file properties only care that loading doesn't crash.

**The example tests all still pass. Only the property notices.** The
example round-trip loads two records and checks them by ID with
`loaded.get("a")`, which finds a record wherever it sits in the list.

Put the line back to `store.list_records()` and run the tests again.
They pass.

!!! tip "Older Hypothesis versions"
    Older versions of Hypothesis print `Falsifying example:` instead of
    `Failing test case:`. It means the same thing.

## A Bug We Found

When we first wrote `test_distance_is_the_same_both_ways`, it failed on
PyVecDB's real code -- no deliberate bug needed:

```
E   OverflowError: (34, 'Result too large')
E   Failing test case: test_distance_is_the_same_both_ways(
E       pair=([0.0], [1.3407807929942597e+154]),
E       distance=euclidean_distance,
E   )
```

The rule was about symmetry, but the test never got that far: measuring
from `[0.0]` to one very big number crashed.

Here's why. The biggest float Python has is about `1.8e308`, and
`1.34e154` squared is just past it. The old code squared each difference
with `** 2`, and in Python `**` *raises* `OverflowError` when the answer
is too big. Plain `*` gives `inf` (infinity) instead -- and that's what
`magnitude()`, `manhattan_distance` and `dot_product` already did. So
`euclidean_distance` was the odd one out.

The example tests used friendly vectors like `[3.0, 4.0]`. Nobody thinks
to measure a distance of 10^154. The scout didn't need to think of it:
`st.floats()` tries huge numbers all the time, and shrinking walked it
down to the smallest one that breaks.

The fix was one line:

```python
    return math.sqrt(sum((x - y) * (x - y) for x, y in zip(a, b, strict=False)))  # was: (x - y) ** 2
```

The shrunk input is now an ordinary example test in
`tests/test_distance.py`, so the bug can't sneak back in.

The junk-file properties found a second bug. A save file holding one byte,
`0x80` (not text at all), made `load_store` crash with
`UnicodeDecodeError`, and a save-shaped file with an empty vector,
`{"dimensions": 1, "records": [{"record_id": null, "vector": []}]}`,
crashed with `EmptyVectorError`. Every damaged file now gets a
`PersistenceError` naming it, and the shrunk files are pinned in
`tests/test_persistence.py`.

## Run It Yourself

Run just the property tests and the model check (100 examples each, the
same every time):

```bash
uv run pytest tests/test_properties.py tests/test_model.py --no-cov
```

The `--no-cov` turns off the coverage check, which expects the *whole*
test suite to run and would fail on two files.

Run them the long way (2,000 random examples each, different every run)
to hunt for rarer bugs. It takes about 4 minutes:

```bash
uv run pytest tests/test_properties.py tests/test_model.py --no-cov --hypothesis-profile=thorough
```

The normal run uses a fixed seed, so it tries the same inputs every time
and a passing build means the same thing on every computer. The thorough
run is random, so each time you run it the scout draws brand-new maps.
Both settings live in `conftest.py` at the top of the project.

## Your Turn

Write one new property for PyVecDB. What should the distance from a
vector to *itself* always be, for `euclidean_distance` and
`manhattan_distance`? What's the strategy that makes the vector (think
about which floats make sense)? What's the `assert`? Run it and watch it
pass. Then break `manhattan_distance` on purpose and watch the scout find
it.
