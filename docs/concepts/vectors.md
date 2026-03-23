# Vectors: Coordinates on a Treasure Map

## What is a Vector?

A **vector** is just a list of numbers. That's it!

```python
my_vector = [3.0, 4.0]
```

Think of it like coordinates on a treasure map. The first number says "go 3 steps east" and the second says "go 4 steps north." Together, they point to the treasure!

## Why Lists of Numbers?

In real life, we describe things with words: "This song is energetic and happy." But computers work better with numbers. So we convert descriptions into numbers:

| Song | Energy | Happiness | Danceability |
|------|--------|-----------|--------------|
| "Thunder" | 0.9 | 0.6 | 0.7 |
| "Lullaby" | 0.1 | 0.8 | 0.2 |
| "Party Rock" | 0.95 | 0.9 | 0.99 |

Each row is a 3-dimensional vector (3 numbers = 3 dimensions).

## Vector Operations

### Magnitude (Length)

How far is the treasure from where you're standing? For `[3, 4]`, it's 5 -- the famous 3-4-5 triangle! (Pythagorean theorem)

```python
from pyvecdb import magnitude
magnitude([3.0, 4.0])  # 5.0
```

### Normalization (Unit Vector)

Normalization shrinks (or stretches) a vector so its length is exactly 1, while keeping the same direction. Like saying "I'm heading northeast" without saying how far.

```python
from pyvecdb import normalize
normalize([3.0, 4.0])  # [0.6, 0.8]
```

### Addition and Subtraction

Adding vectors is like following two sets of directions one after another. Subtracting finds the difference between two positions.

```python
from pyvecdb import add, subtract
add([1.0, 2.0], [3.0, 4.0])       # [4.0, 6.0]
subtract([5.0, 3.0], [1.0, 1.0])  # [4.0, 2.0]
```

## Dimensions

The number of values in a vector is its **dimensionality**:
- 2D: `[x, y]` -- a point on a flat map
- 3D: `[x, y, z]` -- a point in a room
- 768D: A sentence encoded by an AI model!

In pyvecdb, all vectors in a store must have the same number of dimensions. You can't mix 2D and 3D vectors -- that would be like comparing a flat map to a 3D globe!
