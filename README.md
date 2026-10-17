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

## Releasing

Other projects in the series ([PyStack](https://github.com/robcox2910/pystack)) use a pinned release of this one,
not whatever is on `main`. To publish a change for them:

1. Merge the change to `main`.
2. Tag it: `git tag -a vX.Y.Z -m vX.Y.Z && git push origin vX.Y.Z`.
   A change to the public API bumps the minor version (`v0.1.0` → `v0.2.0`);
   anything else bumps the patch (`v0.1.0` → `v0.1.1`).
3. In each project that uses it, edit the tag in `[tool.uv.sources]`, run
   `uv lock --upgrade-package <name>`, and open a pull request so CI checks
   the pair together.

<!-- series:start -->
## Related Projects

This project is one of a series that builds real systems "from scratch in Python" to teach how they work. See the [series home](https://robcox2910.github.io/) for the full story.

| Project | What it teaches | Repository | Docs |
| --- | --- | --- | --- |
| PyOS | Operating systems | [py-os](https://github.com/robcox2910/py-os) | [Docs](https://robcox2910.github.io/py-os/) |
| Pebble | Compilers and programming languages | [pebble-lang](https://github.com/robcox2910/pebble-lang) | [Docs](https://robcox2910.github.io/pebble-lang/) |
| PyDB | Relational databases | [pydb](https://github.com/robcox2910/pydb) | [Docs](https://robcox2910.github.io/pydb/) |
| PyKV | Key-value stores | [pykv](https://github.com/robcox2910/pykv) | [Docs](https://robcox2910.github.io/pykv/) |
| PyDocDB | Document databases | [pydoc-db](https://github.com/robcox2910/pydoc-db) | [Docs](https://robcox2910.github.io/pydoc-db/) |
| PyGraphDB | Graph databases | [pygraphdb](https://github.com/robcox2910/pygraphdb) | [Docs](https://robcox2910.github.io/pygraphdb/) |
| PyTSDB | Time-series databases | [pytsdb](https://github.com/robcox2910/pytsdb) | [Docs](https://robcox2910.github.io/pytsdb/) |
| **PyVecDB** | **Vector databases** | [pyvecdb](https://github.com/robcox2910/pyvecdb) | [Docs](https://robcox2910.github.io/pyvecdb/) |
| PyColDB | Column-family databases | [pycoldb](https://github.com/robcox2910/pycoldb) | [Docs](https://robcox2910.github.io/pycoldb/) |
| PyCrypt | Cryptography | [pycrypt](https://github.com/robcox2910/pycrypt) | [Docs](https://robcox2910.github.io/pycrypt/) |
| PyNet | Computer networking | [pynet](https://github.com/robcox2910/pynet) | [Docs](https://robcox2910.github.io/pynet/) |
| PyWeb | HTTP web servers | [pyweb](https://github.com/robcox2910/pyweb) | [Docs](https://robcox2910.github.io/pyweb/) |
| PyGit | Version control | [pygit](https://github.com/robcox2910/pygit) | [Docs](https://robcox2910.github.io/pygit/) |
| PySearch | Search engines | [pysearch](https://github.com/robcox2910/pysearch) | [Docs](https://robcox2910.github.io/pysearch/) |
| PyMQ | Message queues | [pymq](https://github.com/robcox2910/pymq) | [Docs](https://robcox2910.github.io/pymq/) |
| PyLLM | Language models (the brain) | [pyllm](https://github.com/robcox2910/pyllm) | [Docs](https://robcox2910.github.io/pyllm/) |
| PyStack | Full-stack integration | [pystack](https://github.com/robcox2910/pystack) | [Docs](https://robcox2910.github.io/pystack/) |
<!-- series:end -->

<!-- docs:start -->
## Documentation

Full docs (with kid-friendly explanations of every idea): https://robcox2910.github.io/pyvecdb/
<!-- docs:end -->

## License

MIT
