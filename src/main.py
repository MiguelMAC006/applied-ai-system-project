"""
Music Recommender — command-line entry point.

Usage:
    python -m src.main "I want upbeat pop music for the gym"
    python -m src.main                        # runs built-in demo profiles
"""

import logging
import sys
from pathlib import Path

from src.recommender import load_songs, recommend_songs
from src.retrieval import recommend_from_query

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def setup_logging() -> None:
    log_dir = Path(__file__).parent.parent / "logs"
    log_dir.mkdir(exist_ok=True)

    root = logging.getLogger()
    root.setLevel(logging.INFO)

    fh = logging.FileHandler(log_dir / "recommender.log")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s"))

    # Only warnings and above go to the console so output stays clean
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARNING)
    ch.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))

    root.addHandler(fh)
    root.addHandler(ch)


# ---------------------------------------------------------------------------
# Demo profiles (used when no CLI argument is given)
# ---------------------------------------------------------------------------

HIGH_ENERGY_POP = {
    "name": "High-Energy Pop",
    "genre": "pop", "mood": "happy",
    "energy": 0.92, "valence": 0.88,
    "acousticness": 0.05, "tempo_bpm": 130.0, "danceability": 0.90,
}

CHILL_LOFI = {
    "name": "Chill Lofi",
    "genre": "lofi", "mood": "chill",
    "energy": 0.35, "valence": 0.58,
    "acousticness": 0.80, "tempo_bpm": 76.0, "danceability": 0.58,
}

DEEP_INTENSE_ROCK = {
    "name": "Deep Intense Rock",
    "genre": "rock", "mood": "intense",
    "energy": 0.93, "valence": 0.38,
    "acousticness": 0.07, "tempo_bpm": 155.0, "danceability": 0.62,
}

CONFLICTING_ENERGY_SAD = {
    "name": "Conflicting — High Energy + Sad Mood",
    "genre": "folk", "mood": "sad",
    "energy": 0.90, "valence": 0.10,
    "acousticness": 0.85, "tempo_bpm": 100.0, "danceability": 0.30,
}

METAL_BUT_ACOUSTIC = {
    "name": "Conflicting — Metal Genre + Max Acousticness",
    "genre": "metal", "mood": "aggressive",
    "energy": 0.97, "valence": 0.25,
    "acousticness": 0.95, "tempo_bpm": 170.0, "danceability": 0.50,
}

ALL_EXTREMES = {
    "name": "Edge Case — All Parameters at Maximum",
    "genre": "edm", "mood": "euphoric",
    "energy": 1.00, "valence": 1.00,
    "acousticness": 0.00, "tempo_bpm": 200.0, "danceability": 1.00,
}

PROFILES = [
    HIGH_ENERGY_POP, CHILL_LOFI, DEEP_INTENSE_ROCK,
    CONFLICTING_ENERGY_SAD, METAL_BUT_ACOUSTIC, ALL_EXTREMES,
]


# ---------------------------------------------------------------------------
# Output formatters
# ---------------------------------------------------------------------------

MAX_SCORE = 7.75
SEP = "-" * 56


def print_rag_recommendations(query: str, candidates: list, results: list) -> None:
    print("\n" + "=" * 56)
    print(f"  QUERY: {query}")
    print("=" * 56)

    print(f"\n  Retrieved {len(candidates)} candidate(s) from catalog:")
    for s in candidates:
        print(f"    • [{s['id']:2}] {s['title']} ({s['genre']}, {s['mood']})")

    print(f"\n  Top {len(results)} Recommendation(s):")
    for rank, (song, score, explanation) in enumerate(results, start=1):
        bar_filled = round((score / MAX_SCORE) * 20) if score > 0 else 0
        bar = "#" * bar_filled + "." * (20 - bar_filled)
        print(f"\n  #{rank}  {song['title']}  —  {song['artist']}")
        print(f"       Genre: {song['genre']}  |  Mood: {song['mood']}")
        print(f"       Score: {score:.2f} / {MAX_SCORE:.2f}  [{bar}]")
        print(f"       Why:")
        for reason in explanation.split(" · "):
            print(f"         • {reason}")
        print(SEP)


def print_recommendations(user_prefs: dict, recommendations: list) -> None:
    print("\n" + "=" * 56)
    print(f"  PROFILE: {user_prefs['name']}")
    print("=" * 56)

    for rank, (song, score, explanation) in enumerate(recommendations, start=1):
        bar_filled = round((score / MAX_SCORE) * 20)
        bar = "#" * bar_filled + "." * (20 - bar_filled)
        print(f"\n  #{rank}  {song['title']}  —  {song['artist']}")
        print(f"       Genre: {song['genre']}  |  Mood: {song['mood']}")
        print(f"       Score: {score:.2f} / {MAX_SCORE:.2f}  [{bar}]")
        print(f"       Why:")
        for reason in explanation.split(" · "):
            print(f"         • {reason}")
        print(SEP)


# ---------------------------------------------------------------------------
# Runner helpers
# ---------------------------------------------------------------------------

def _load_catalog() -> list:
    csv_path = Path(__file__).parent.parent / "data" / "songs.csv"
    return load_songs(str(csv_path))


def run_rag_mode(query: str) -> None:
    songs = _load_catalog()
    from src.retrieval import retrieve_songs, parse_query_preferences
    candidate_k = min(len(songs), max(10, 5 * 2))
    candidates = retrieve_songs(query, songs, k=candidate_k)
    results = recommend_from_query(query, songs, k=5)
    print_rag_recommendations(query, candidates, results)


def run_demo_mode() -> None:
    songs = _load_catalog()
    for profile in PROFILES:
        recommendations = recommend_songs(profile, songs, k=5)
        print_recommendations(profile, recommendations)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    setup_logging()
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        run_rag_mode(query)
    else:
        run_demo_mode()


if __name__ == "__main__":
    main()
