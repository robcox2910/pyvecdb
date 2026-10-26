# What is a Vector Database?

## The "Finding Similar Songs" Analogy

Imagine you just discovered an amazing song. You love everything about it -- the beat, the energy, the mood. Now you want to find MORE songs just like it. How would a computer do that?

### Step 1: Turn Songs into Numbers

First, we describe each song as a list of numbers. Imagine rating each song on a few scales:

- **Energy**: 0 (calm) to 1 (pumped up)
- **Happiness**: 0 (sad) to 1 (happy)
- **Danceability**: 0 (sit still) to 1 (can't stop moving)

Your favorite song might be: `[0.9, 0.7, 0.8]` -- high energy, pretty happy, very danceable.

That list of numbers is called a **vector**. It's like a point in "taste-space" -- a magical 3D room where similar songs sit close together!

### Step 2: Store Lots of Vectors

A **vector database** stores thousands (or millions!) of these song vectors. Each one has:
- An **ID** (like "song-42")
- A **vector** (the list of numbers)
- **Metadata** (extra info like the artist name, genre, year)

### Step 3: Find the Closest Ones

When you search, you give the database your favorite song's vector and say "find me the 5 most similar." It measures the "distance" between your song and every other song in the database, then returns the 5 closest ones.

That's called **K-Nearest Neighbors (KNN)** search -- "K" is the number of results you want.

## Why Do Vector Databases Matter?

Vector databases are everywhere in modern AI:

- **Music apps** find similar songs
- **Image search** finds similar photos
- **ChatGPT** retrieves relevant documents
- **Recommendation systems** suggest products you'll like

AI models (like the ones behind ChatGPT) turn text, images, and audio into vectors called **embeddings**. A vector database stores and searches these embeddings super fast.

## What You'll Learn

By building pyvecdb, you'll understand:

1. How vectors represent real-world things as numbers
2. How different distance metrics measure similarity
3. How KNN search finds the best matches
4. How metadata filtering narrows results
5. How to persist data to disk

Let's get started!

## Install

```bash
pip install fromscratch-pyvecdb
```

Or get the whole series at once with `pip install fromscratch-pystack`. To read the code or run the tests, clone it from [GitHub](https://github.com/robcox2910/pyvecdb).
