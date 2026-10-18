"""Strategies -- recipes for random IDs, vectors and metadata, shared by the property tests.

A strategy tells Hypothesis how to invent one kind of input. Each one here
is a named function, named for what it makes, with a one-line docstring.
``tests/test_properties.py`` and ``tests/test_model.py`` both use them.

The brute-force "how far away is it?" helpers that both files check search
results against live here too, so there is exactly one copy of them.
"""

from hypothesis import strategies as st

from pyvecdb.filtering import MetadataValue

# Search vectors are made of whole numbers between these two, written as
# floats (3.0, -7.0). Adding and multiplying small whole numbers is exact, so
# the brute-force answer below and the store's answer can be compared with
# ``==``, with no "close enough" fudge that could hide a real mistake.
SMALLEST_COORDINATE = -10
LARGEST_COORDINATE = 10

# The metrics whose scores are exact for whole-number vectors. Cosine
# similarity divides by square roots, so two shops that are really the same
# "distance" away can come out one rounding step apart; it also refuses a
# vector of all zeros. It's left out here, not hidden: see the docs page.
EXACT_METRICS = ("euclidean", "manhattan", "dot")


def record_ids() -> st.SearchStrategy[str]:
    """Make any record ID at all: empty, spaces, emoji, accents, quotes, the lot."""
    return st.text()


def whole_coordinates() -> st.SearchStrategy[float]:
    """Make one small whole number, written as a float, such as -3.0 or 7.0."""
    return st.integers(SMALLEST_COORDINATE, LARGEST_COORDINATE).map(float)


def search_vectors(dimensions: int) -> st.SearchStrategy[list[float]]:
    """Make a vector of exactly ``dimensions`` small whole numbers."""
    return st.lists(whole_coordinates(), min_size=dimensions, max_size=dimensions)


def metadata_values() -> st.SearchStrategy[MetadataValue]:
    """Make one metadata value: text, a whole number, a decimal, or True/False."""
    # The allowed types are pyvecdb.filtering.MetadataValue. NaN ("not a
    # number") is never equal to itself, so ``==`` could never confirm it came
    # back; that's a rule of floats, not of PyVecDB. Infinity stays in.
    return st.one_of(st.text(), st.integers(), st.floats(allow_nan=False), st.booleans())


def any_metadata() -> st.SearchStrategy[dict[str, MetadataValue]]:
    """Make the extra info for one record: a few text keys, each with a metadata value."""
    return st.dictionaries(st.text(), metadata_values(), max_size=3)


def search_metrics() -> st.SearchStrategy[str]:
    """Pick one of the metrics whose scores are exact for whole-number vectors."""
    return st.sampled_from(EXACT_METRICS)


# --- Brute force: the obviously-right way to rank vectors --------------------


def how_far(vector: list[float], query: list[float], metric: str) -> int:
    """Return how far ``vector`` is from ``query`` as a whole number: smaller is a better match.

    This is deliberately *not* the store's code. It works in whole numbers,
    so nothing is ever rounded: squared straight-line distance for
    ``euclidean`` (squaring keeps the same order and needs no square root),
    blocks walked for ``manhattan``, and minus the dot product for ``dot``
    (a bigger dot product is a better match, so we flip its sign).
    """
    pairs = [(round(v), round(q)) for v, q in zip(vector, query, strict=True)]
    if metric == "euclidean":
        return sum((v - q) * (v - q) for v, q in pairs)
    if metric == "manhattan":
        return sum(abs(v - q) for v, q in pairs)
    return -sum(v * q for v, q in pairs)
