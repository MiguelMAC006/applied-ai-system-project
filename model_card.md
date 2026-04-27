# Model Card: AuraPlay 2.0 (RAG Edition)

---

## 1. Model Name

**AuraPlay 2.0**

An upgrade of AuraPlay 1.0 that adds a Retrieval-Augmented Generation (RAG) layer. Instead of scoring every song against a numeric profile, the system first retrieves a semantically relevant candidate pool using TF-IDF, then scores only those candidates.

---

## 2. Goal / Task

The system takes a **natural-language query** (e.g., "upbeat pop music for the gym") and returns five ranked song recommendations from a small local catalog (`data/songs.csv`).

It does not use neural language models, streaming APIs, or behavioral history. All retrieval and scoring is local and deterministic.

---

## 3. How It Works

### RAG Pipeline

1. **Retrieval (TF-IDF):** Each song is converted into a text document that includes its genre, mood, title, artist, and derived descriptors (e.g., "high energy", "gym", "acoustic"). A TF-IDF vectorizer indexes these documents. At query time, cosine similarity ranks the catalog and returns the top candidates (default: 10).

2. **Query Parsing (keyword mapping):** A rule-based parser maps recognized words to structured preferences:
   - "gym", "workout" → `energy: 0.9`, `tempo_bpm: 130`
   - "chill", "lofi", "study" → `energy: 0.3`, `tempo_bpm: 75`
   - "happy", "upbeat" → `valence: 0.8`
   - "sad", "dark" → `valence: 0.2`
   - "acoustic", "folk" → `acousticness: 0.85`
   - "electronic", "edm" → `acousticness: 0.1`
   - Genre and mood words found in the catalog are also detected.

3. **Scoring (weighted proximity):** Only the retrieved candidates are scored. The same formula from AuraPlay 1.0 applies (max 7.75 pts), covering genre/mood exact match and energy, valence, danceability, acousticness, and tempo similarity.

4. **Ranking:** Candidates are sorted by score descending. Top-k are returned with per-reason explanations.

**Why retrieval matters:** Songs semantically irrelevant to the query (e.g., a folk/sad song for a gym query) are excluded before scoring begins. This is a direct improvement over the v1 approach of scoring all 18 songs blindly.

---

## 4. Data

- **Catalog:** 18 songs in `data/songs.csv`
- **Features per song:** id, title, artist, genre, mood, energy, tempo_bpm, valence, danceability, acousticness
- **Genres:** pop, lofi, rock, ambient, synthwave, jazz, indie pop, hip-hop, r&b, classical, country, metal, reggae, folk, edm
- **Moods:** happy, chill, intense, relaxed, focused, moody, nostalgic, romantic, melancholic, uplifting, aggressive, peaceful, sad, euphoric

**Dataset limitations:**
- 18 songs is a very small catalog. Real streaming services have millions.
- 13 of 15 genres have exactly one song — genre-specific retrieval has no diversity.
- No energy values exist between 0.45 and 0.65 — moderate-energy queries have no close matches.
- All songs were synthesized for this simulation; they do not reflect real listening behavior or cultural diversity.
- This system does **not** use real streaming data, user history, or external APIs.

---

## 5. Algorithm Summary

**Retrieval (TF-IDF):**
- Each song → text document with genre, mood, title, artist + feature-derived terms
- `TfidfVectorizer` + cosine similarity selects top-k candidates
- Falls back to first k songs if query is empty or vectorizer fails

**Scoring (max 7.75 pts):**

| Signal | Weight | Type |
|---|---|---|
| Genre match | +1.00 | Binary |
| Mood match | +1.00 | Binary |
| Energy | up to +3.00 | Continuous |
| Valence | up to +1.00 | Continuous |
| Danceability | up to +0.75 | Continuous |
| Acousticness | up to +0.50 | Continuous |
| Tempo | up to +0.50 | Continuous |

---

## 6. Observed Behavior and Biases

**Keyword parsing is literal.** A query like "not acoustic" will still trigger `acousticness: 0.85` if the word "acoustic" is present. Negation is not handled. Users who phrase preferences with negatives ("not too energetic") may get incorrect preference mappings.

**Small catalog amplifies filter bubbles.** With 18 songs, TF-IDF retrieval frequently surfaces the same songs (e.g., Gym Hero, Sunrise City for any high-energy query). Adding more songs per genre would reduce repetition.

**Genre/mood labels dominate.** The scoring formula awards +2.0 pts for genre+mood matches vs. a maximum of +5.75 for all numeric features combined. A labeled match can override superior audio-feature alignment.

**Energy dead zone persists.** No songs have energy 0.45–0.65. Mid-energy queries still find no close matches.

**No behavioral signal.** The system cannot distinguish between a first-time listener and a power user. It also cannot detect when a song has been over-recommended.

---

## 7. Guardrails

- Empty query → safe fallback, no crash
- Malformed CSV rows → skipped with a warning log
- All parsed numeric preferences clamped to [0, 1] (tempo: [40, 220])
- TF-IDF failure (import error, edge-case corpus) → falls back to first k catalog songs
- All returned songs are guaranteed to be from `data/songs.csv` — no hallucinated results

---

## 8. Evaluation

**Automated tests (19 total):**
- Retrieval returns only catalog songs
- RAG pipeline returns ranked tuples
- Empty query does not crash retrieve or recommend
- Final recommendations restricted to catalog
- Workout/pop query surfaces high-energy or pop song at rank 1
- Explanation strings are non-empty
- Preference parsing maps gym → high energy, chill → low energy, etc.
- Numeric preferences are clamped
- Full CSV loads correctly (18 songs)
- End-to-end RAG on real catalog

**Manual spot-checks:**

| Query | Expected top result | Actual top result |
|---|---|---|
| "upbeat happy pop music for the gym" | Sunrise City (pop, happy) | Sunrise City ✓ |
| "chill lofi music for studying" | Library Rain or Midnight Coding | Library Rain ✓ |

---

## 9. Intended vs. Non-Intended Use

**Intended:**
- Learning how RAG combines retrieval and scoring
- Exploring how keyword parsing affects recommendations
- Educational exploration of content-based filtering

**Not intended for:**
- Real music recommendations for real users
- Large catalogs (TF-IDF on 18 docs is fast; scale changes the tradeoffs)
- Production deployment without additional guardrails, fairness review, and a much larger catalog
- Any claim that the catalog represents diverse musical taste or cultural listening habits

---

## 10. Ideas for Improvement

1. **Handle negation in query parsing.** "not acoustic" should set `acousticness: 0.1`, not 0.85.
2. **Expand the catalog.** Even 100 songs would allow TF-IDF to produce meaningfully different retrievals per query.
3. **Add genre families.** "indie pop" and "pop" should share partial credit rather than being treated as completely different labels.
4. **Use a real embedding model.** Replacing TF-IDF with a sentence transformer would enable semantic retrieval that handles synonyms and paraphrasing.

---

## 11. Personal Reflection

Adding the RAG layer revealed how much the *framing* of a query changes what the system considers. In v1, a "gym" profile and a "high-energy pop" profile with identical numeric features produced identical results because every song was scored the same way. With TF-IDF retrieval, a query containing the word "gym" now surfaces Gym Hero before scoring begins — the word itself is meaningful signal.

The biggest surprise was how well a keyword-based query parser works on a constrained catalog. "chill lofi study music" reliably retrieves the three lofi songs and scores them highest, even though the parser is just checking token membership in predefined sets. Simplicity is competitive when the domain is narrow.

The key limitation that RAG does not solve is the small catalog. With only one or two songs per genre, retrieval diversity is inherently limited regardless of how good the retrieval is.
