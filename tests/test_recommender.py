"""Tests for the music recommender RAG pipeline."""

import pytest
from src.recommender import Song, UserProfile, Recommender, load_songs, score_song
from src.retrieval import retrieve_songs, parse_query_preferences, recommend_from_query

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

SAMPLE_SONGS = [
    {
        "id": 1, "title": "Pump It Up", "artist": "Test Artist",
        "genre": "pop", "mood": "happy",
        "energy": 0.92, "tempo_bpm": 132, "valence": 0.85,
        "danceability": 0.88, "acousticness": 0.05,
    },
    {
        "id": 2, "title": "Chill Rain", "artist": "Ambient Artist",
        "genre": "lofi", "mood": "chill",
        "energy": 0.35, "tempo_bpm": 75, "valence": 0.58,
        "danceability": 0.55, "acousticness": 0.82,
    },
    {
        "id": 3, "title": "Metal Storm", "artist": "Heavy Artist",
        "genre": "metal", "mood": "intense",
        "energy": 0.97, "tempo_bpm": 165, "valence": 0.28,
        "danceability": 0.50, "acousticness": 0.08,
    },
    {
        "id": 4, "title": "Acoustic Morning", "artist": "Folk Artist",
        "genre": "folk", "mood": "relaxed",
        "energy": 0.30, "tempo_bpm": 70, "valence": 0.65,
        "danceability": 0.42, "acousticness": 0.90,
    },
]


def make_small_recommender() -> Recommender:
    songs = [
        Song(id=1, title="Test Pop Track", artist="Test Artist",
             genre="pop", mood="happy", energy=0.8, tempo_bpm=120,
             valence=0.9, danceability=0.8, acousticness=0.2),
        Song(id=2, title="Chill Lofi Loop", artist="Test Artist",
             genre="lofi", mood="chill", energy=0.4, tempo_bpm=80,
             valence=0.6, danceability=0.5, acousticness=0.9),
    ]
    return Recommender(songs)


# ---------------------------------------------------------------------------
# Recommender class tests (fixed from original)
# ---------------------------------------------------------------------------

def test_recommend_returns_songs_sorted_by_score():
    user = UserProfile(
        favorite_genre="pop",
        favorite_mood="happy",
        target_energy=0.8,
        target_valence=0.85,
        target_acousticness=0.1,
        target_tempo_bpm=120.0,
        target_danceability=0.8,
    )
    rec = make_small_recommender()
    results = rec.recommend(user, k=2)

    assert len(results) == 2
    assert results[0].genre == "pop"
    assert results[0].mood == "happy"


def test_explain_recommendation_returns_non_empty_string():
    user = UserProfile(
        favorite_genre="pop",
        favorite_mood="happy",
        target_energy=0.8,
        target_valence=0.85,
        target_acousticness=0.1,
        target_tempo_bpm=120.0,
        target_danceability=0.8,
    )
    rec = make_small_recommender()
    song = rec.songs[0]

    explanation = rec.explain_recommendation(user, song)
    assert isinstance(explanation, str)
    assert explanation.strip() != ""


# ---------------------------------------------------------------------------
# retrieve_songs tests
# ---------------------------------------------------------------------------

def test_retrieve_songs_returns_catalog_songs():
    results = retrieve_songs("upbeat pop gym workout", SAMPLE_SONGS, k=3)
    catalog_ids = {s["id"] for s in SAMPLE_SONGS}
    for song in results:
        assert song["id"] in catalog_ids


def test_retrieve_songs_returns_at_most_k():
    results = retrieve_songs("music", SAMPLE_SONGS, k=2)
    assert len(results) <= 2


def test_retrieve_songs_never_returns_outside_catalog():
    results = retrieve_songs("energetic happy dance music", SAMPLE_SONGS, k=4)
    catalog_ids = {s["id"] for s in SAMPLE_SONGS}
    for song in results:
        assert song["id"] in catalog_ids


def test_empty_query_does_not_crash_retrieve():
    results = retrieve_songs("", SAMPLE_SONGS, k=3)
    assert isinstance(results, list)
    assert len(results) <= len(SAMPLE_SONGS)


# ---------------------------------------------------------------------------
# recommend_from_query (RAG pipeline) tests
# ---------------------------------------------------------------------------

def test_recommend_from_query_returns_ranked():
    results = recommend_from_query("upbeat pop gym", SAMPLE_SONGS, k=2)
    assert len(results) > 0
    assert len(results) <= 2
    for song, score, explanation in results:
        assert isinstance(song, dict)
        assert isinstance(score, float)
        assert isinstance(explanation, str)


def test_empty_query_does_not_crash_recommend():
    results = recommend_from_query("", SAMPLE_SONGS, k=3)
    assert isinstance(results, list)
    catalog_ids = {s["id"] for s in SAMPLE_SONGS}
    for song, score, explanation in results:
        assert song["id"] in catalog_ids


def test_recommendations_restricted_to_catalog():
    catalog_ids = {s["id"] for s in SAMPLE_SONGS}
    results = recommend_from_query("happy energetic music", SAMPLE_SONGS, k=4)
    for song, score, explanation in results:
        assert song["id"] in catalog_ids, f"Song ID {song['id']} not in catalog"


def test_upbeat_pop_workout_returns_high_energy_or_pop():
    """A gym/workout/pop query should surface high-energy or pop songs near the top."""
    results = recommend_from_query("upbeat happy pop workout music", SAMPLE_SONGS, k=3)
    assert len(results) > 0
    top_song, _score, _expl = results[0]
    assert top_song["energy"] >= 0.6 or top_song["genre"] == "pop"


def test_explanation_strings_are_non_empty():
    results = recommend_from_query("chill lofi study music", SAMPLE_SONGS, k=2)
    for song, score, explanation in results:
        assert explanation.strip() != ""


# ---------------------------------------------------------------------------
# parse_query_preferences tests
# ---------------------------------------------------------------------------

def test_parse_gym_query_sets_high_energy_and_genre():
    prefs = parse_query_preferences("gym workout upbeat pop")
    assert prefs.get("energy", 0) >= 0.8
    assert prefs.get("genre") == "pop"


def test_parse_chill_query_sets_low_energy_and_genre():
    prefs = parse_query_preferences("chill lofi study music")
    assert prefs.get("energy", 1.0) <= 0.5
    assert prefs.get("genre") == "lofi"


def test_parse_empty_query_returns_empty_dict():
    assert parse_query_preferences("") == {}
    assert parse_query_preferences("   ") == {}


def test_numeric_prefs_are_clamped():
    prefs = parse_query_preferences("happy energetic dance music")
    for key in ("energy", "valence", "danceability", "acousticness"):
        if key in prefs:
            assert 0.0 <= prefs[key] <= 1.0


def test_sad_query_sets_low_valence():
    prefs = parse_query_preferences("sad dark melancholic music")
    assert prefs.get("valence", 1.0) <= 0.4


def test_acoustic_query_sets_high_acousticness():
    prefs = parse_query_preferences("acoustic folk organic music")
    assert prefs.get("acousticness", 0.0) >= 0.7


# ---------------------------------------------------------------------------
# Integration: load from actual CSV
# ---------------------------------------------------------------------------

def test_load_songs_returns_full_catalog():
    songs = load_songs("data/songs.csv")
    assert len(songs) == 18
    assert all("title" in s for s in songs)
    assert all("energy" in s for s in songs)


def test_rag_pipeline_on_real_catalog():
    songs = load_songs("data/songs.csv")
    results = recommend_from_query("upbeat happy pop workout music", songs, k=5)
    assert len(results) > 0
    catalog_ids = {s["id"] for s in songs}
    for song, score, explanation in results:
        assert song["id"] in catalog_ids
