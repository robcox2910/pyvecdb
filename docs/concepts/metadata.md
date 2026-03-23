# Metadata: Extra Info Like Toppings on Each Pizza Shop

## What is Metadata?

Every vector record can carry extra information called **metadata**. It's like the details on a restaurant listing:

- The **vector** tells you WHERE it is (position in taste-space)
- The **metadata** tells you WHAT it is (name, genre, year, rating, etc.)

```python
from pyvecdb import VectorStore

store = VectorStore(dimensions=3)
store.insert(
    "song-42",
    vector=[0.9, 0.1, 0.8],
    metadata={
        "title": "Thunder",
        "artist": "Imagine Dragons",
        "genre": "rock",
        "year": 2017,
        "bpm": 168,
    },
)
```

## Why Metadata Matters

Without metadata, search results would just be lists of numbers -- not very helpful! Metadata lets you:

1. **Display results**: Show the song title instead of `[0.9, 0.1, 0.8]`
2. **Filter results**: "Find similar songs, but only rock songs from 2020+"

## Metadata Filtering

Filtering lets you narrow search results by metadata values. It's like adding checkboxes to your search: "Only show me vegetarian pizza shops within 2 miles."

### Simple Equality

```python
from pyvecdb import knn_search

# Only find rock songs
results = knn_search(store, query=my_vector, k=5, metadata_filter={"genre": "rock"})
```

### Comparison Operators

```python
# Songs after 2020
results = knn_search(store, query=my_vector, k=5, metadata_filter={"year": {"$gt": 2020}})

# Songs from 2020 or later
results = knn_search(store, query=my_vector, k=5, metadata_filter={"year": {"$gte": 2020}})

# Songs before 2020
results = knn_search(store, query=my_vector, k=5, metadata_filter={"year": {"$lt": 2020}})
```

### The $in Operator

```python
# Rock OR pop songs
results = knn_search(
    store,
    query=my_vector,
    k=5,
    metadata_filter={"genre": {"$in": ["rock", "pop"]}},
)
```

### Multiple Conditions (AND)

All conditions must match:

```python
# Rock songs after 2020
results = knn_search(
    store,
    query=my_vector,
    k=5,
    metadata_filter={
        "genre": "rock",
        "year": {"$gt": 2020},
    },
)
```

## Supported Operators

| Operator | Meaning | Example |
|----------|---------|---------|
| (none) | Equals | `{"genre": "rock"}` |
| `$eq` | Equals | `{"genre": {"$eq": "rock"}}` |
| `$ne` | Not equals | `{"genre": {"$ne": "country"}}` |
| `$gt` | Greater than | `{"year": {"$gt": 2020}}` |
| `$gte` | Greater or equal | `{"year": {"$gte": 2020}}` |
| `$lt` | Less than | `{"year": {"$lt": 2020}}` |
| `$lte` | Less or equal | `{"year": {"$lte": 2020}}` |
| `$in` | In list | `{"genre": {"$in": ["rock", "pop"]}}` |

## Metadata Types

Metadata values can be:
- **Strings**: `"rock"`, `"Imagine Dragons"`
- **Integers**: `2017`, `168`
- **Floats**: `4.5`, `0.95`
- **Booleans**: `True`, `False`
