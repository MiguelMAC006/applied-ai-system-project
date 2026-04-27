# Music Recommender with RAG

A command-line music recommender that uses **Retrieval-Augmented Generation (RAG)** to turn natural-language queries into ranked song recommendations from a local catalog.

---

## What It Does

You type a free-text query like:

```
"I want upbeat pop music for the gym"
```

The system:
1. **Retrieves** semantically relevant songs from `data/songs.csv` using TF-IDF cosine similarity.
2. **Parses** your query into structured preferences (genre, mood, energy, valence, etc.) using keyword mappings.
3. **Scores** only the retrieved candidates using a weighted feature-similarity formula.
4. **Returns** ranked recommendations with a per-song explanation.

Retrieval directly narrows the scoring universe — songs that are semantically irrelevant to your query cannot appear in the final output.

---

## How RAG Changes Recommendations

Without RAG, every song in the catalog is scored against the user's numeric preferences. With RAG:

- TF-IDF selects a **candidate pool** (default: 10 songs) whose text representation best matches the query.
- Only those candidates are scored, so a "gym workout" query surfaces high-energy/pop songs at retrieval time — before any numeric scoring begins.
- This means **rare or borderline songs** that happen to score well numerically but are semantically unrelated to the query are excluded.

---

## Project Structure

```
.
├── src/
│   ├── main.py          — CLI entry point; RAG mode or demo mode
│   ├── recommender.py   — scoring logic, load_songs, Song/UserProfile dataclasses
│   └── retrieval.py     — TF-IDF retrieval, query parser, recommend_from_query (RAG)
├── data/
│   └── songs.csv        — 18-song catalog with audio features
├── tests/
│   └── test_recommender.py  — 19 tests covering retrieval, RAG pipeline, guardrails
├── logs/                — auto-created; recommender.log written here at runtime
├── requirements.txt
├── model_card.md
└── reflection.md
```

---

## Setup

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # macOS/Linux
.venv\Scripts\activate           # Windows

# 2. Install dependencies
pip install -r requirements.txt
```

---

## Running the App

**RAG mode — natural-language query:**

```bash
python -m src.main "I want upbeat happy pop music for the gym"
python -m src.main "Give me chill lofi music for studying"
python -m src.main "Something dark and moody for a late night drive"
```

**Demo mode — runs six built-in taste profiles:**

```bash
python -m src.main
```

### Sample Output (RAG mode)

```
========================================================
  QUERY: I want upbeat happy pop music for the gym
========================================================

  Retrieved 10 candidate(s) from catalog:
    • [ 5] Gym Hero (pop, intense)
    • [ 1] Sunrise City (pop, happy)
    • [10] Rooftop Lights (indie pop, happy)
    • [18] Pulse Drop (edm, euphoric)
    ...

  Top 5 Recommendation(s):

  #1  Sunrise City  —  Neon Echo
       Genre: pop  |  Mood: happy
       Score: 6.17 / 7.75  [################....]
       Why:
         • genre match (pop) +1.0
         • mood match (happy) +1.0
         • energy similarity +2.76
         • valence similarity +0.96
         • tempo similarity +0.45
```

---

## Running Tests

```bash
pytest
```

All 19 tests should pass. The suite covers:
- `retrieve_songs()` returns only catalog songs
- `recommend_from_query()` returns ranked tuples
- Empty query does not crash (retrieve or recommend)
- Final recommendations are restricted to catalog songs
- A workout/pop query returns a high-energy or pop song at the top
- Explanation strings are non-empty
- `parse_query_preferences()` maps gym → high energy, lofi → low energy, etc.
- Numeric preferences are clamped to [0, 1]
- Full catalog loads correctly (18 songs)
- End-to-end RAG pipeline on real catalog

---

## Guardrails and Logging

**Guardrails:**
- Empty query → safe fallback (returns first k songs), no crash
- Missing or malformed CSV fields → row is skipped with a warning
- All numeric preferences clamped to valid ranges ([0,1] or [40,220] BPM)
- Recommendations are always a subset of `data/songs.csv` — no hallucinated songs
- TF-IDF failure (e.g., missing scikit-learn) → fallback to first k songs

**Logging:**
- Each run logs to `logs/recommender.log` (auto-created)
- Log entries include: original query, parsed preferences, retrieved song IDs, final recommendation IDs, any warnings or fallbacks
- Console shows only WARNING+ level messages so normal output stays clean

---

## Scoring Algorithm

Maximum possible score: **7.75 points**

| Signal | Weight | Type |
|---|---|---|
| Genre match | +1.00 | Binary (exact string) |
| Mood match | +1.00 | Binary (exact string) |
| Energy similarity | up to +3.00 | `3 × (1 - |diff|)` |
| Valence similarity | up to +1.00 | `1 × (1 - |diff|)` |
| Danceability similarity | up to +0.75 | `0.75 × (1 - |diff|)` |
| Acousticness similarity | up to +0.50 | `0.5 × (1 - |diff|)` |
| Tempo similarity | up to +0.50 | `0.5 × (1 - |diff|/120)` |

---

## Limitations

- Catalog is 18 songs — niche genres have one representative each.
- Genre and mood matching is exact string comparison; "indie pop" ≠ "pop".
- Query parsing is keyword-based; complex or ambiguous phrasing may not map cleanly.
- No behavioral data — the system cannot learn from listening history.
- Not intended for production use; built for educational exploration of RAG concepts.
