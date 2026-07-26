# pyvecdb

An educational vector database library built from scratch in Python. Part of the **PyLearn** platform for kids aged 12+.

## What is a Vector Database?

A vector database stores **vectors** (lists of numbers) and finds **similar** ones. Think of it like a music app that turns songs into lists of numbers and finds other songs with similar numbers!

## Features

- **Vector Operations** -- add, subtract, scale, normalize, dot product, magnitude
- **Vector Store** -- CRUD operations for vector records with dimensionality checks
- **Distance Metrics** -- cosine similarity, Euclidean, Manhattan, dot product
- **KNN Search** -- find the K most similar vectors to a query
- **Metadata Filtering** -- filter results by metadata (eq, ne, gt, gte, lt, lte, in)
- **Batch Operations** -- insert and search multiple vectors at once
- **Persistence** -- save/load to JSON files
- **Pure Python** -- no external dependencies

## Quick Start

```python
from pyvecdb import VectorStore, knn_search

# Create a store for 3-dimensional vectors
store = VectorStore(dimensions=3)

# Insert some songs as vectors (energy, happiness, danceability)
store.insert("thunder", [0.9, 0.6, 0.7], metadata={"genre": "rock", "year": 2017})
store.insert("lullaby", [0.1, 0.8, 0.2], metadata={"genre": "folk", "year": 2020})
store.insert("party-rock", [0.95, 0.9, 0.99], metadata={"genre": "pop", "year": 2011})
store.insert("ballad", [0.3, 0.4, 0.1], metadata={"genre": "rock", "year": 2019})

# Find the 2 most similar songs to a high-energy query
results = knn_search(store, query=[0.85, 0.7, 0.8], k=2)
for record, score in results:
    print(f"{record.record_id}: {score:.3f}")
# party-rock: 0.998
# thunder: 0.995

# Filter by metadata -- only rock songs
results = knn_search(
    store,
    query=[0.85, 0.7, 0.8],
    k=2,
    metadata_filter={"genre": "rock"},
)
```

## Installation

```bash
git clone https://github.com/robcox2910/pyvecdb.git
cd pyvecdb
uv sync --all-extras
```

## Development

```bash
uv run ruff check src/ tests/      # Lint
uv run ruff format src/ tests/      # Format
uv run pyright src tests            # Type check
uv run pytest                       # Test (90%+ coverage)
```

## Related Projects

| Project | Description |
|---------|-------------|
| [pydb](https://github.com/robcox2910/pydb) | Educational relational database |
| [pykv](https://github.com/robcox2910/pykv) | Educational key-value store |
| [pymq](https://github.com/robcox2910/pymq) | Educational message queue |
| [pynet](https://github.com/robcox2910/pynet) | Educational networking library |
| [pyweb](https://github.com/robcox2910/pyweb) | Educational web framework |
| [pycrypt](https://github.com/robcox2910/pycrypt) | Educational cryptography library |
| [pygit](https://github.com/robcox2910/pygit) | Educational Git implementation |
| [pysearch](https://github.com/robcox2910/pysearch) | Educational search engine |
| [pebble-lang](https://github.com/robcox2910/pebble-lang) | Educational programming language |
| [py-os](https://github.com/robcox2910/py-os) | Educational operating system concepts |
| [pystack](https://github.com/robcox2910/pystack) | Unified platform integrating all projects |

## License

MIT
