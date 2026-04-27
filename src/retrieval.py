"""
RAG retrieval layer for the music recommender.

Pipeline:
  1. retrieve_songs()         — TF-IDF cosine similarity over song documents
  2. parse_query_preferences() — keyword mapping: query → structured prefs dict
  3. recommend_from_query()   — retrieve → parse → score → rank
"""

import logging
from typing import Optional

from src.recommender import score_song

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Document builder
# ---------------------------------------------------------------------------

def _build_document(song: dict) -> str:
    """Convert a song dict into a searchable text string for TF-IDF indexing."""
    parts = [
        song.get("genre", ""),
        song.get("mood", ""),
        song.get("title", ""),
        song.get("artist", ""),
    ]

    energy = float(song.get("energy", 0.5))
    if energy >= 0.8:
        parts += ["high", "energy", "intense", "energetic", "upbeat", "powerful"]
    elif energy <= 0.4:
        parts += ["low", "energy", "calm", "chill", "relaxed", "gentle"]

    valence = float(song.get("valence", 0.5))
    if valence >= 0.7:
        parts += ["happy", "bright", "positive", "uplifting", "joyful"]
    elif valence <= 0.35:
        parts += ["sad", "dark", "moody", "melancholic", "somber"]

    danceability = float(song.get("danceability", 0.5))
    if danceability >= 0.75:
        parts += ["danceable", "dance", "groove", "club", "party"]

    acousticness = float(song.get("acousticness", 0.5))
    if acousticness >= 0.7:
        parts += ["acoustic", "organic", "folk", "natural", "unplugged"]
    elif acousticness <= 0.2:
        parts += ["electronic", "synth", "produced", "digital"]

    tempo = float(song.get("tempo_bpm", 100))
    if tempo >= 130:
        parts += ["fast", "uptempo", "workout", "gym", "run"]
    elif tempo <= 80:
        parts += ["slow", "study", "background", "lofi"]

    return " ".join(parts)


# ---------------------------------------------------------------------------
# TF-IDF retrieval
# ---------------------------------------------------------------------------

def retrieve_songs(query: str, songs: list[dict], k: int = 8) -> list[dict]:
    """
    Retrieve top-k songs from the catalog using TF-IDF cosine similarity.

    Guardrails:
    - Empty query → returns first k songs (safe fallback).
    - Any sklearn error → returns first k songs (safe fallback).
    - Results are always a subset of the supplied songs list.
    """
    if not query or not query.strip():
        logger.warning("Empty query received; returning first %d songs as fallback", k)
        return songs[:k]

    if not songs:
        logger.warning("Empty song catalog; nothing to retrieve")
        return []

    k = min(k, len(songs))

    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        documents = [_build_document(s) for s in songs]
        vectorizer = TfidfVectorizer(stop_words="english")
        tfidf_matrix = vectorizer.fit_transform(documents)
        query_vec = vectorizer.transform([query.lower()])
        scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
        top_indices = scores.argsort()[::-1][:k]

        retrieved = [songs[int(i)] for i in top_indices]
        logger.info(
            "TF-IDF retrieved %d songs for query %r → IDs: %s",
            len(retrieved),
            query,
            [s["id"] for s in retrieved],
        )
        return retrieved

    except Exception as exc:
        logger.error("TF-IDF retrieval failed (%s); returning first %d songs", exc, k)
        return songs[:k]


# ---------------------------------------------------------------------------
# Keyword tables for natural-language preference parsing
# ---------------------------------------------------------------------------

_HIGH_ENERGY = {"gym", "workout", "hype", "energetic", "intense", "pump", "power", "run", "sprint", "hard"}
_LOW_ENERGY  = {"chill", "study", "lofi", "relax", "calm", "sleep", "quiet", "background", "gentle", "soft"}

_HIGH_VALENCE = {"happy", "bright", "upbeat", "positive", "joyful", "cheerful", "fun", "uplifting"}
_LOW_VALENCE  = {"sad", "dark", "moody", "melancholic", "somber", "depressing", "gloomy"}

_HIGH_DANCE = {"dance", "club", "party", "danceable", "groove", "floor", "bop"}

_HIGH_ACOUSTIC = {"acoustic", "folk", "organic", "unplugged", "natural"}
_LOW_ACOUSTIC  = {"electronic", "edm", "synth", "digital", "produced", "electro"}

_HIGH_TEMPO = {"fast", "quick", "uptempo", "gym", "workout", "run", "sprint"}
_LOW_TEMPO  = {"slow", "chill", "relax", "study", "lofi"}

_KNOWN_GENRES = {
    "pop", "lofi", "rock", "ambient", "synthwave", "jazz",
    "hip-hop", "r&b", "classical", "country", "metal",
    "reggae", "folk", "edm", "indie",
}

_KNOWN_MOODS = {
    "happy", "chill", "intense", "relaxed", "focused", "moody",
    "nostalgic", "romantic", "melancholic", "uplifting", "aggressive",
    "peaceful", "sad", "euphoric",
}


def parse_query_preferences(query: str) -> dict:
    """
    Convert a natural-language music query into a structured preferences dict
    compatible with score_song().

    Guardrails:
    - Empty/blank query → returns {}.
    - All numeric values are clamped to [0, 1] (tempo_bpm to [40, 220]).
    """
    if not query or not query.strip():
        return {}

    tokens = set(query.lower().split())
    q = query.lower()
    prefs: dict = {}

    if tokens & _HIGH_ENERGY:
        prefs["energy"] = 0.9
    elif tokens & _LOW_ENERGY:
        prefs["energy"] = 0.3

    if tokens & _HIGH_VALENCE:
        prefs["valence"] = 0.8
    elif tokens & _LOW_VALENCE:
        prefs["valence"] = 0.2

    if tokens & _HIGH_DANCE:
        prefs["danceability"] = 0.85

    if tokens & _HIGH_ACOUSTIC:
        prefs["acousticness"] = 0.85
    elif tokens & _LOW_ACOUSTIC:
        prefs["acousticness"] = 0.1

    if tokens & _HIGH_TEMPO:
        prefs["tempo_bpm"] = 130.0
    elif tokens & _LOW_TEMPO:
        prefs["tempo_bpm"] = 75.0

    # Multi-word genres first, then single-token match
    if "hip hop" in q or "hip-hop" in q:
        prefs["genre"] = "hip-hop"
    elif "indie pop" in q:
        prefs["genre"] = "indie pop"
    elif "r&b" in q or "rnb" in q:
        prefs["genre"] = "r&b"
    else:
        for g in _KNOWN_GENRES:
            if g in tokens:
                prefs["genre"] = g
                break

    for mood in _KNOWN_MOODS:
        if mood in tokens:
            prefs["mood"] = mood
            break

    # Clamp numeric values into valid ranges
    for key in ("energy", "valence", "danceability", "acousticness"):
        if key in prefs:
            prefs[key] = min(1.0, max(0.0, prefs[key]))
    if "tempo_bpm" in prefs:
        prefs["tempo_bpm"] = min(220.0, max(40.0, prefs["tempo_bpm"]))

    return prefs


# ---------------------------------------------------------------------------
# RAG recommendation pipeline
# ---------------------------------------------------------------------------

def recommend_from_query(
    query: str,
    songs: list[dict],
    k: int = 5,
) -> list[tuple[dict, float, str]]:
    """
    RAG-style recommendation:

    1. Retrieve a pool of semantically relevant candidates via TF-IDF.
    2. Parse the natural-language query into structured preferences.
    3. Score ONLY the retrieved candidates (retrieval narrows the scoring universe).
    4. Return top-k ranked (song, score, explanation) tuples.

    Guardrails:
    - Empty query → safe fallback, no crash.
    - Failed retrieval → fallback to first k catalog songs.
    - All returned songs are guaranteed to be from the supplied songs list.
    - If no preferences can be parsed, candidates are returned in retrieval order.
    """
    logger.info("recommend_from_query | query=%r | k=%d | catalog_size=%d", query, k, len(songs))

    # Step 1: Retrieve a semantically relevant candidate pool
    candidate_k = min(len(songs), max(k * 2, 8))
    candidates = retrieve_songs(query, songs, k=candidate_k)

    if not candidates:
        logger.warning("No candidates retrieved; falling back to first %d catalog songs", k)
        candidates = songs[:k]

    logger.info("Candidate pool IDs: %s", [s["id"] for s in candidates])

    # Step 2: Parse query into structured preferences
    prefs = parse_query_preferences(query)
    logger.info("Parsed preferences: %s", prefs)

    # Step 3: If no preferences parsed, return candidates in retrieval order
    if not prefs:
        logger.info("No preferences parsed; returning candidates in retrieval order")
        return [
            (song, 0.0, "retrieved by semantic similarity")
            for song in candidates[:k]
        ]

    # Step 4: Score only the retrieved candidates
    scored: list[tuple[dict, float, str]] = []
    for song in candidates:
        song_score, reasons = score_song(prefs, song)
        explanation = " · ".join(reasons) if reasons else "retrieved by semantic similarity"
        scored.append((song, song_score, explanation))

    scored.sort(key=lambda x: x[1], reverse=True)
    final = scored[:k]

    logger.info("Final recommendation IDs: %s", [s[0]["id"] for s in final])
    return final
