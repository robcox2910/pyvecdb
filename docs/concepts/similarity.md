# Similarity: Finding Your Musical Twin

## What is Similarity?

**Similarity** is the opposite of distance. If two vectors are very close together (low distance), they are very similar (high similarity). If they're far apart, they're not similar at all.

Think of it like finding your "musical twin" -- the person in your school whose taste in music is most like yours!

## How It Works

### Step 1: Rate Your Favorites

Imagine everyone rates their favorite music on three scales:

| Person | Energy | Happiness | Danceability |
|--------|--------|-----------|--------------|
| You | 0.9 | 0.7 | 0.8 |
| Alice | 0.85 | 0.75 | 0.7 |
| Bob | 0.1 | 0.9 | 0.2 |
| Carol | 0.8 | 0.6 | 0.9 |

### Step 2: Compute Similarity

Using cosine similarity:
- You vs Alice: **0.998** (almost identical -- musical twins!)
- You vs Carol: **0.986** (pretty similar)
- You vs Bob: **0.745** (quite different)

### Step 3: Rank by Similarity

Your top matches: Alice > Carol > Bob

## Similarity vs Distance

| Concept | Meaning | "Best" Value |
|---------|---------|-------------|
| Similarity | How alike | Higher is better |
| Distance | How different | Lower is better |

In pyvecdb, the search function always returns results sorted with the "best match first" -- whether that means highest similarity or lowest distance.

## Why Cosine Similarity is King

For AI embeddings, cosine similarity is the most popular metric because:

1. **Scale-invariant**: A review with 5 sentences and one with 50 sentences can still be compared fairly
2. **Bounded**: Always between -1 and 1, easy to interpret
3. **Fast**: Simple math, even for high-dimensional vectors
