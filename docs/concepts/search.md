# Search: Finding the 5 Closest Pizza Shops

## The Pizza Shop Problem

You're hungry and want pizza. There are 100 pizza shops in your city. You want to find the 5 closest ones to where you're standing.

That's **K-Nearest Neighbors (KNN)** search!
- **K** = how many results you want (5 pizza shops)
- **Nearest** = closest by some distance measure
- **Neighbors** = the other vectors in the database

## How KNN Search Works

### The Simple (Brute Force) Way

1. Measure the distance from YOU to every single pizza shop
2. Sort all distances from shortest to longest
3. Take the first K results

That's exactly what pyvecdb does! It's called "brute force" because we check every single vector. It's simple and correct.

```python
from pyvecdb import VectorStore, knn_search

store = VectorStore(dimensions=2)
store.insert("pizza-palace", [3.0, 4.0])
store.insert("slice-heaven", [1.0, 1.0])
store.insert("crust-kingdom", [10.0, 10.0])

# Find the 2 closest to my location [2.0, 2.0]
results = knn_search(store, query=[2.0, 2.0], k=2, metric="euclidean")
for record, score in results:
    print(f"{record.record_id}: distance = {-score:.2f}")
```

### Choosing a Metric

You can search using any of the four distance metrics:

```python
# Default: cosine similarity (best for AI)
knn_search(store, query=[1.0, 0.0], k=5, metric="cosine")

# Euclidean: straight-line distance
knn_search(store, query=[1.0, 0.0], k=5, metric="euclidean")

# Manhattan: city-block distance
knn_search(store, query=[1.0, 0.0], k=5, metric="manhattan")

# Dot product: agreement score
knn_search(store, query=[1.0, 0.0], k=5, metric="dot")
```

## Batch Search

Need to search for multiple things at once? Use batch search:

```python
from pyvecdb import batch_search

results = batch_search(store, queries=[[1.0, 0.0], [0.0, 1.0]], k=3)
# results[0] = top 3 for first query
# results[1] = top 3 for second query
```

## Performance Note

Our brute-force approach checks every vector for every search. For small databases (up to ~10,000 vectors), this is plenty fast. Real production vector databases use clever tricks like indexes and approximate search to handle millions of vectors!
