# N-gram Language Models

## Overview

This project was developed as part of the **Text Analytics** course (2025-26) and covers two exercises on N-gram Language Models:

- **Exercise 2** — Levenshtein Distance: implementation of the edit distance algorithm and vocabulary-based fuzzy retrieval
- **Exercise 3** — N-gram Language Models: bigram and trigram models trained on a real-world corpus, evaluated with perplexity, and used for sentence completion

**Authors:** Stefanatou Gerasimina, Perikou Aikaterini, Lamprinou Evdoxia

---

## Exercise 2 — Levenshtein Distance

### Part (i) — Algorithm Implementation

Implements the classic **dynamic programming** algorithm for computing the minimum edit distance between two strings. Allowed operations: insertion, deletion, substitution (each with cost 1, except substitution which costs 2 for distinct characters).

- Bottom-up DP table construction
- Returns both the full distance matrix and the final distance value
- Includes a visualisation function that prints the DP table in readable tabular format

**Test cases:**

| Word 1 | Word 2 | Distance |
|---|---|---|
| colour | color | 1 |
| dog | cat | 6 |
| Rain | raining | 5 |
| hello | Hello | 2 |
| python | java | 10 |

Note: the algorithm is **case-sensitive** — uppercase and lowercase characters are treated as distinct.

### Part (ii) — Vocabulary Retrieval

Extends the algorithm to fuzzy vocabulary search: given a query word and a distance threshold, retrieves all vocabulary words within that distance, sorted by ascending distance.

- Vocabulary derived from the **Brown Corpus** (NLTK), keeping words with frequency ≥ 10
- Query word is lowercased before matching (case-insensitive retrieval)

**Example results for query = 'colour':**

| Threshold | Words found |
|---|---|
| d ≤ 1 | 1 (color) |
| d ≤ 2 | 2 (color, colors) |
| d ≤ 3 | 7 (color, colors, court, our, lou, cloud, colored) |
| d ≤ 5 | 125 |

Results were validated against the reference tool at http://www.let.rug.nl/~kleiweg/lev/.

---

## Exercise 3 — N-gram Language Models

### Corpus

- **Source:** Brown Corpus, `mystery` category (detective & crime fiction)
- **Size:** 3,886 sentences
- **Split:** 80% train (3,108 sentences) / 20% test (778 sentences)

### Preprocessing

- **Vocabulary:** words appearing ≥ 10 times in the training set → 528 unique words
- **OOV handling:** all rare words replaced with `*UNK*` token
- **Boundary tokens:**
  - Bigram: 1× `*start*` at sentence beginning, 1× `*end*` at sentence end
  - Trigram: 2× `*start*` at sentence beginning, 1× `*end*` at sentence end

### Models

Both models use **Laplace smoothing** (α = 1.0) to avoid zero probabilities for unseen n-grams.

**Bigram:**
$$P(w_i \mid w_{i-1}) = \frac{C(w_{i-1}, w_i) + \alpha}{C(w_{i-1}) + \alpha \cdot |V|}$$

**Trigram:**
$$P(w_i \mid w_{i-2}, w_{i-1}) = \frac{C(w_{i-2}, w_{i-1}, w_i) + \alpha}{C(w_{i-2}, w_{i-1}) + \alpha \cdot |V|}$$

### Evaluation Results

| Model | Set | Cross-Entropy | Perplexity |
|---|---|---|---|
| Bigram | Train | 5.464 | 44.137 |
| Bigram | Test | 5.579 | 47.803 |
| Trigram | Train | 6.527 | 92.251 |
| Trigram | Test | 6.944 | 123.164 |

**Key findings:**
- The bigram model generalizes better (small train/test gap), because bigram statistics are stable across the corpus
- The trigram model overfits more due to data sparsity — 79.2% of trigrams appear only once in training
- Despite richer context, the trigram model underperforms the bigram model, highlighting the fundamental trade-off: higher-order models need substantially more data

### Sentence Completion

Three decoding strategies were implemented and compared:

| Method | Description | Characteristics |
|---|---|---|
| **Greedy** | Always selects the most probable next word | Deterministic, fast, but repetitive |
| **Beam Search** | Keeps top-3 candidate sequences at each step | More stable, globally better sequences |
| **Top-K Sampling** | Samples from top-5 candidates with probability weighting | Most diverse, but less coherent |

**General observations:**
- Beam search produced the most balanced results overall
- Trigram model generates more fluent local phrases (e.g. "at the same time") but becomes unstable under Top-K sampling in sparse contexts
- OOV words in the input significantly degrade generation quality
- Inputs matching the mystery corpus domain produce noticeably better completions

---

## Dependencies

```
nltk
collections
math
```

---

## Notes

- A vocabulary threshold of ≥ 10 was chosen to balance coverage vs. sparsity. Experiments with lower thresholds (≥ 2, ≥ 5) showed that greedy decoding is largely unaffected, beam search shows minor differences, while Top-K sampling becomes more diverse but less stable with larger vocabularies.
- The `(*start*, *start*)` bigram count for the trigram model must be manually set to the number of training sentences, as this pair does not naturally appear in the corpus.
