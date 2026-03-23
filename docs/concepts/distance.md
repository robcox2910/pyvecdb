# Distance: How Far Apart Are Two Spots?

## The Big Idea

If vectors are points on a map, **distance** tells you how far apart two points are. There are different ways to measure "how far" -- just like there are different ways to get across town!

## Four Ways to Measure Distance

### 1. Euclidean Distance (Straight Line)

This is the "as the crow flies" distance -- a straight line between two points. It uses the Pythagorean theorem extended to any number of dimensions.

```python
from pyvecdb import euclidean_distance
euclidean_distance([0.0, 0.0], [3.0, 4.0])  # 5.0
```

**When to use**: When you care about the actual magnitude of differences.

### 2. Cosine Similarity (Direction Only)

Cosine similarity ignores how long the vectors are and only checks if they point the same direction. It returns a number from -1 to 1:
- **1.0** = pointing the same way (identical taste!)
- **0.0** = perpendicular (completely unrelated)
- **-1.0** = pointing opposite ways (opposite taste!)

```python
from pyvecdb import cosine_similarity
cosine_similarity([1.0, 0.0], [1.0, 0.0])  # 1.0 (same!)
cosine_similarity([1.0, 0.0], [0.0, 1.0])  # 0.0 (unrelated)
```

**When to use**: Most popular for AI embeddings. This is the default in pyvecdb!

### 3. Manhattan Distance (City Blocks)

Imagine walking through a city grid -- you can only go along streets, not diagonally through buildings. Manhattan distance counts the total blocks walked.

```python
from pyvecdb import manhattan_distance
manhattan_distance([0.0, 0.0], [3.0, 4.0])  # 7.0 (3 + 4 blocks)
```

**When to use**: When dimensions are independent (like different categories).

### 4. Dot Product (Agreement Score)

The dot product multiplies matching components and adds them up. Bigger numbers mean more "agreement."

```python
from pyvecdb import dot_product
dot_product([1.0, 2.0, 3.0], [4.0, 5.0, 6.0])  # 32.0
```

**When to use**: When vector length matters (not just direction).

## Which One Should I Use?

| Metric | Best For | Range |
|--------|----------|-------|
| Cosine | AI embeddings, text | -1 to 1 |
| Euclidean | Physical distance | 0 to infinity |
| Manhattan | Grid-like data | 0 to infinity |
| Dot Product | When magnitude matters | -infinity to infinity |

For most AI applications, **cosine similarity** is the go-to choice!
